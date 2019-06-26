/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;

std::default_random_engine gen;


inline double multi_gaussian(double pr_x, double pr_y, double ob_x, double ob_y, double std_x, double std_y){
    return ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(pr_x-ob_x,2)/(2*pow(std_x, 2)) + (pow(pr_y-ob_y,2)/(2*pow(std_y, 2))) ) );
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 101;  // TODO: Set the number of particles

    //define normal distributions for sensor noise
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for(int i=0;i<num_particles;i++){
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    normal_distribution<double> delta_x(0, std_pos[0]);
    normal_distribution<double> delta_y(0, std_pos[1]);
    normal_distribution<double> delta_theta(0, std_pos[2]);

    //update positions
    for(int i=0;i<num_particles;i++) {
        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else{
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }

        //add noise
        particles[i].x += delta_x(gen);
        particles[i].y += delta_y(gen);
        particles[i].theta += delta_theta(gen);
    }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
    int num_obs = observations.size();
    int num_pred = predicted.size();
    for(int i=0;i<num_obs;i++){
        double x1 = observations[i].x;
        double y1 = observations[i].y;
        double x2 = predicted[0].x;
        double y2 = predicted[0].y;
        double min_dis = dist(x1, y1, x2, y2);
        observations[i].id = predicted[0].id;
        for(int j=1;j<num_pred;j++){
            x2 = predicted[j].x;
            y2 = predicted[j].y;
            double dis = dist(x1, y1, x2, y2);
            if(dis< min_dis){
                min_dis = dis;
                observations[i].id = predicted[j].id;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    int num_obs = observations.size();
    int num_landmarks = map_landmarks.landmark_list.size();
    for(int i=0;i<num_particles;i++){
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;
        vector<LandmarkObs> landmarks;
        for(int j=0;j<num_landmarks;j++){
            double l_x = map_landmarks.landmark_list[j].x_f;
            double l_y = map_landmarks.landmark_list[j].y_f;
            int l_id = map_landmarks.landmark_list[j].id_i;

            if((fabs(p_x - l_x)<sensor_range) && fabs(p_y - l_y)< sensor_range){
                landmarks.push_back(LandmarkObs({l_id, l_x, l_y}));
            }
        }
        //Transform the observed landmarks from vehicle coordinates to map coordinates
        vector<LandmarkObs> transformed_landmarks;
        for(int k=0;k<num_obs;k++){
            double t_x = cos(p_theta)*observations[k].x - sin(p_theta)*observations[k].y + p_x;
            double t_y = sin(p_theta)*observations[k].x + cos(p_theta)*observations[k].y + p_y;
            transformed_landmarks.push_back(LandmarkObs({observations[k].id, t_x, t_y}));
        }

        dataAssociation(landmarks, transformed_landmarks);

        particles[i].weight = 1.0;

        for(int j=0;j<transformed_landmarks.size();j++){
            double trans_x = transformed_landmarks[j].x;
            double trans_y = transformed_landmarks[j].y;
            double related_x, related_y;
            for(int k=0; k<landmarks.size();k++ ){
                if(landmarks[k].id == transformed_landmarks[j].id){
                    related_x = landmarks[k].x;
                    related_y = landmarks[k].y;
                }
            }
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];
            double obs_w = multi_gaussian(related_x,related_y,trans_x,trans_y,std_x,std_y);

            particles[i].weight *= obs_w;
        }
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    vector<Particle> new_particles;
    vector<double> weights;
    for(int i=0; i<num_particles; i++){
        weights.push_back(particles[i].weight);
    }
    std::uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);

    double max_weight = *max_element(weights.begin(), weights.end());

    std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;
    for(int i=0;i<num_particles;i++){
        beta += unirealdist(gen) * 2.0;
        while(beta > weights[index]){
            beta -= weights[index];
            index = (index+1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
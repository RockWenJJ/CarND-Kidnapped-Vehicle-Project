# Overview
This repository contains all the codes needed to complete the final project for Kidnapped Vehicle project.

## Project Introduction
In this project I have implemented a 2 dimensional particle filter in C++. The particle filter has be given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter could get observation and control data. The filter would then predict which particle has the largest possibility to represent the real location of the vehicle.

## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for `uWebSocketIO` is complete, the main program can be built and ran by doing the following from the project top directory.

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./particle_filter
```

On the other hand, you should turn on the Term 2 Simulator to evaluate the fulfilled particle filter. The result below seems pretty well.

![](kidnapped_vehicle.gif)

Here is the main protocol that `main.cpp` uses for `uWebSocketIO` in communicating with the simulator.

**INPUT: values provided by the simulator to the c++ program**

* sense noisy position data from the simulator

```
["sense_x"]
["sense_y"]
["sense_theta"]
```

* get the previous velocity and yaw rate to predict the particle's transitioned state

```
["previous_velocity"]
["previous_yawrate"]
```

* receive noisy observation data from the simulator, in a respective list of x/y values

```
["sense_observations_x"]
["sense_observations_y"]
```

**OUTPUT: values provided by the c++ program to the simulator**

* best particle values used for calculating the error evaluation

```
["best_particle_x"]
["best_particle_y"]
["best_particle_theta"]
```

* Optional message data used for debugging particle's sensing and associations for respective (x,y) sensed positions ID label

```
["best_particle_associations"]
```

* for respective (x,y) sensed positions

```
["best_particle_sense_x"] <= list of sensed x positions
["best_particle_sense_y"] <= list of sensed y positions
```

# Implementing the Particle Filter
The directory structure of this repository is as follows:

```
root
|   set_git.sh
|   clean.sh
|   CMakeLists.txt
|   README.md
|   run.sh
|   install-mac.sh
|   install-ubuntu.sh
|   kidnapped_vehicle.gif
|
|___data
|   |   
|   |   map_data.txt
|   
|   
|___src
    |   helper_functions.h
    |   main.cpp
    |   map.h
    |   particle_filter.cpp
    |   particle_filter.h
    |   json.hpp
```

## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory.

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns

1. x position
2. y position
3. landmark id

### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.

# Introduction

Perspective-n-Point(PnP) is the problem of estimating the pose of a calibrated camera given a set of n 3D points in the world and their corresponding 2D projections in the image.(https://en.wikipedia.org/wiki/Perspective-n-Point)

We extend this methods to 3D-2D lines, which is called as Perspective-n-Line(PnL). Moreover, we can combine 3D-2D point pair and 3D-2D line pair together to solve the camera pose. These are implemented in pnpl.cpp(h) as a graph optimization and solved by g2o.

The pnp.cpp(h) contains close form solutions for camera position with known camera attitude(rotation). Note that we need at least 2 point pair in pnpTrans() while at least 3 line pair in pnlTrans().

# Prerequisites
Prerequisites needed for compiling PnPL using c++:
- OpenCV (http://opencv.org)
- g2o (https://github.com/RainerKuemmerle/g2o)

Note that we oply use opencv data structures to store datas. You can change the variable type in codes if you really don't want to use opencv.

# How to run
- Linux
```
mkdir build
cd build
cmake ..
make
```
- Windows
You can use CMake-GUI to build your project.

We provide two demos to show how to use,
1. demo_pnpl solves camera position and rotation (6DoF) with point pairs and line pairs, it needs g2o.
2. demo_pnptrans solves only camera position(3DoF) with point pairs and line pairs in close form.

# Licence
The source code is released under GPLv3 license.

Any problem, please contact maoshuyuan123@gmail.com

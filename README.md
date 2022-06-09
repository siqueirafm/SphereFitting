# SphereFitting  

A class to fit a sphere to a set of points in space. Unit tests for the class are provided.

## INTRODUCTION  

The code is organized as follows:  

* headers        - subdirectory containing the HPP source files  
* include        - subdirectory where gtest headers files will be after code is built (initially empty)  
* lib            - subdirectory where gtest lib files will be after code is built (initially empty)  
* LICENSE.md     - license file  
* README.md      - this file  
* sources        - subdirectory containing the CPP source files  
* tst            - subdirectory where the executable with the unit tests will be after the code is built  
* CMakeLists.txt - Input to the CMake build system  

The fiiting algorithms implemented in the class come from the following papers:

Samuel M. Thomas and Y. T. Chan  
A simple approach for the estimation of circular arc center and radius.  
Computer Vision, Graphics, and Image Processing, 45, p. 362-370, 1989.  

I. D. Coope  
Circle Fitting by Linear and Nonlinear Least Squares.  
Journal of Optimization Theory and Applications, 76(2), p. 381-388, 1993.  

Gabor Lukács, Ralph Martin, and Dave Marshall.  
Faithful Least-SquaresFitting  of Spheres, Cylinders, Cones and Tori for Reliable Segmentation,  
Proceedings of the ECCV'98, LNCS, v. 1406,  Springer, Berlin, Heidelberg, 1998.  

## INSTALLATION  

You need to install CMake 3.15 or higher and Eigen 3.3 or higher prior to building the code.
CMake will then try to find Eigen automatically. If it fails to do so, please provide the path to the Eigen library using CMake variable.
CMake will also download and install the GTest library automatically from [here](https://github.com/google/googletest/).  

To install, follow the following steps:

* Enter directory SphereFitting  
* Run cmake -S . -B build
* Run cmake --build build --config Release
* Run cmake --install build --prefix [full path to your directory SphereFitting]

If all goes well, then you should see an executable inside subdirectory tst.  

If CMake fails to find Eigen, then provide the path to the Eigen directory:

* Run cmake -S . -B build -DEIGEN3_INCLUDE_DIR=[full path to the Eigen library]

##  LAST UPDATE

May 29, 2022

## CONTACT

If you run  into trouble compiling or using the library, please email me at:

mfsiqueira@gmail.com

Have fun!

Marcelo Siqueira

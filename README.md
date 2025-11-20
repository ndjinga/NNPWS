## NNPWS

This project provides a class implementing the [IAPWS-97](https://iapws.org/documents/release/IF97-Rev) functions using neural networks instead of polynomials.

## Introduction

Numerical simulations of water and steam require thermodynamics function that can be costly in terms of computational time.
The International Association for the Properties of Water and Steam (IAPWS) did an extensive survey of experimental data and proposed functions aimed at an industrial use in 1997.
The original [IAPWS-97](https://iapws.org/documents/release/IF97-Rev) functions use polynomials of very large degree. Examples of packages implementing these functions are [CoolProp](https://coolprop.org/) (C++), [FreeSteam](https://github.com/qingfengxia/freesteam)  (C++) and [iapws](https://iapws.readthedocs.io) (Python).  
We provide a new implementation of these functions based on neural networks to improve the performance. In order to facilitate their use in numerical simulation, the implementation allows pressure-temperature (P,T) as well as (P,h) pressure-enthalpy couples as input parameters.  
We also  provide error and computational time estimates.

## Installation

```bash
git clone https://github.com/ndjinga/NNPWS NNPWS_SRC
mkdir build
cd build
cmake ../NNPWS_SRC -DCMAKE_INSTALL_PREFIX=../path/to/install/folder -DNNPWS_WITH_PYTHON=ON
make
make test 
make install 
```

## Prerequisites
+ CMake, mandatory. Package 'cmake' on Fedora and Ubuntu
+ python (optional), required if '-DNNPWS_WITH_PYTHON=ON'. Package 'python-devel' on Fedora, 'python-dev' on Ubuntu
+ SWIG (optional), required if '-DNNPWS_WITH_PYTHON=ON'. Package 'swig' on Fedora and Ubuntu
+ matplotlib (optional), for curves plotting in tests. Package 'python-matplotlib' on Fedora and Ubuntu


## Notebook & examples



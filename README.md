## NNPWS

This project provides a class implementing the [IAPWS-97](https://iapws.org/documents/release/IF97-Rev) functions using neural networks.

## Introduction

Simulation using water ans steam required thermodynamics function that can be costly in terms of computational time.
The original [IAPWS-97](https://iapws.org/documents/release/IF97-Rev) functions use polynomials of very large degree. 
We provide an implementation based on neural networks to improve the performance.
We provide error and computational time estimates.

## Installation

```bash
mkdir build
cd build
cmake ../NNPWS -DCMAKE_INSTALL_PREFIX=../path/to/install/folder -DNNPWS_WITH_PYTHON=ON
make
make test 
make install 
```

## Prerequisites


## Notebook & examples



# Cholesky decompositions for sparse kernels in Gaussian Processes

This repository is the implementation section of my thesis.
The [GPR class (GP.py)](GP.py) implements a Gaussian Process Regression class based on a custom banded Cholesky decomposition in [JAX](https://github.com/google/jax).
The code is benchmarked on a number of different examples to measure the effectiveness of sparse Cholesky implementations in Gaussian processes.
Locally supported covariance functions ("sparse kernels") are used in this context.

## Motivation
Using covariance tapering in conjunction with special sparse Cholesky factorisation algorithms, it may be possible to reduce the time complexity of larger GP models.
The effect of which is supposed to be investigated on real datasets.

## Features
- Gaussian Process Regression in JAX
  - Banded implementation using pure JAX
  - Sparse implementation using a custom primitive based on [Eigen](https://eigen.tuxfamily.org).
  - Optimizing the marginal loglikelihood using [Optax](https://github.com/deepmind/optax)
  - Optimizing the marginal loglikelihood using [JAXopt](https://github.com/google/jaxopt)
- Kernels
  - Including covariance functions with local support, e.g. the [Wendland functions](http://www.math.iit.edu/~fass/603_ch4.pdf)
- Benchmarks based on real data

## Requirements

The project runs on Python 3.11.

To install requirements:

```bash
pip install -r requirements.txt
```

Additionally, the package `liesel-sparse` is required for the sparse algorithms. Currently only available at [liesel-devs/liesel-sparse](https://github.com/liesel-devs/liesel-sparse).
A virtual environment is recommended. The required datasets are included in the repository.

## Datasets
`data` contains the following processed datasets and other relevant files:
- Metz, J. and Ammer, C., Dendrometer data of trees, neighbor, 1 mip, 2012-2013, Dataset. Published. Version 12, DatasetId: 17766, 2018. Available: https://www.bexis.uni-jena.de/ddm/data/Showdata/17766.
- Pieter, T. and Keeling, R., Trends in atmospheric carbon dioxide, National Oceanic and Atmospheric Administration/Global Monitoring Laboratory, Feb. 2023. Available: https://gml.noaa.gov/ccgg/trends/data.html.

## Code example
```python3
def kernel_(s, l, x, y):
    return MaternKernel32(s, l, x, y) * WendlandTapering(3, 8, x, y)

gpr = GPR(X_train, y_train, kernel_, params = jnp.array([37**2, 3]), eps = 0.01)
gpr.fit(X_train, y_train)
mean_pred = gpr.predict(X_test, False)
```

## Results
The thesis shows that the sparse kernels can make GPs more scalable if only the Cholesky decomposition of sparse algorithms is considered.
In a general sense, the sparse implementations are slower than a pure JAX version.
This is due to the overhead of the algorithms, in particular the conversion between different matrix formats.
The compactly supported covariance functions achieve great success in banded covariance matrix and reduce the theoretical computational complexity.
## License
Licensed under the MIT License.
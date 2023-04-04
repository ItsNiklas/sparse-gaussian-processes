#!/usr/bin/env python
# coding: utf-8

import sys

import jax.experimental.sparse
import liesel_sparse
import pandas as pd
from jax.experimental import sparse
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *

sys.path.insert(0, "../")
from GP import *

jax.config.parse_flags_with_absl()

# ### Using Tree data

dendro = pd.read_feather("../data/17766_12_D.feather")
dendro.DOY = dendro.DOY / 48


# weibull_params = pd.read_csv("../data/17766_Wparams.csv", index_col=0, usecols=['index', '0', '1', '2'])

@jax.jit
def weibull_F(x, lambda_, k_):
    return 1 - jnp.exp(-((lambda_ * x) ** k_))


@jax.jit
def kernel(theta, x, y):
    # 37**2 * Matern(length_scale=3, length_scale_bounds=(2,16), nu = 2.5) + WhiteKernel(noise_level=.01, noise_level_bounds="fixed")
    return (
            MaternKernel32(65.3 ** 2, 25, x, y) + ExpSineSquaredKernel(0.4 ** 2, 0.1, 1, x, y)
    ) * WendlandTapering(3, theta, x, y)


@jax.jit
def inv_cov_chol_jax(K, data_y, eps):
    K = K.at[jnp.diag_indices_from(K)].add(eps)

    # Solve Kα=y using the Cholesky decomposition.
    L = jax.lax.linalg.cholesky(K)
    alpha = jax.lax.linalg.triangular_solve(
        L.T,
        jax.lax.linalg.triangular_solve(L, data_y, left_side=True, lower=True),
        left_side=True,
    )

    return L, alpha


def inv_cov_chol_sparse(K, data_y, eps):
    K = K.at[jnp.diag_indices_from(K)].add(eps)

    K = sparse.BCOO.fromdense(K)

    # Solve Kα=y using the Cholesky decomposition.
    L_sp_idx = jnp.argwhere(jax.jit(liesel_sparse.symbolic_factorization)(K) > 0)

    return _solve_sparse(K, data_y, L_sp_idx)

@jax.jit
def _solve_sparse(K, data_y, L_sp_idx):
    L = liesel_sparse.cholesky_sparse(K, L_sp_idx).todense()
    alpha = jax.lax.linalg.triangular_solve(
        L.T,
        jax.lax.linalg.triangular_solve(L, data_y, left_side=True, lower=True),
        left_side=True,
    )

    return L, alpha



def f(MODE: str = "band", X_TEST_SIZE: int = 10000, X_TRAIN_SIZE: int = 1000, N_TREES: int = 1,
      WENDLAND_LIMIT: float = 10.0):
    X_test = jnp.linspace(0, dendro.DOY.max(), X_TEST_SIZE).reshape(-1, 1)

    jaxkey = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)

    if MODE == "full":
        for tree in dendro.dendroNr.unique()[:N_TREES]:
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = np.sort(rng.integers(0, len(df_.DOY) - 1, X_TRAIN_SIZE))

            X_train = df_.DOY.iloc[idx_train].array.reshape(-1, 1)
            y_train = df_.deltagrowth.iloc[idx_train]

            kernel_ = (
                    65.3 ** 2 * Matern(length_scale=25, length_scale_bounds=(2, 16 * 48), nu=2.5)
                    + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed")
                    + ConstantKernel(0.4 ** 2) * ExpSineSquared(0.1, 1, periodicity_bounds="fixed")
            )
            gp_model = gaussian_process.GaussianProcessRegressor(
                kernel=kernel_,
                # n_restarts_optimizer=2,
                # normalize_y=True,
                optimizer=None,
            )

            gp_model.fit(X_train, y_train)
            mean_pred = gp_model.predict(X_test, False)
            p1, p2, p3 = 0, 0, 0  # weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )

            # print(gp_model.log_marginal_likelihood_value_)

    # # Band
    if MODE == "band":
        for tree in dendro.dendroNr.unique()[:N_TREES]:
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = jnp.sort(
                jax.random.randint(jaxkey, (X_TRAIN_SIZE,), 0, len(df_.DOY) - 1)
            )
            X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
            y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

            covariance_function = Partial(kernel, WENDLAND_LIMIT)
            eps = 0.1

            K_ = cov_matrix(X_train, X_train, covariance_function)

            L_, alpha_ = inv_cov_chol(K_, y_train, eps, int(jax.jit(band.bandwidth)(K_)))
            K_trans = cov_matrix(X_train, X_test, covariance_function)
            mean_pred = jnp.dot(K_trans, alpha_)

            p1, p2, p3 = 0, 0, 0  # weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )

            weibull_pred.block_until_ready()
        return

    if MODE == "jax":
        for tree in dendro.dendroNr.unique()[:N_TREES]:
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = jnp.sort(
                jax.random.randint(jaxkey, (X_TRAIN_SIZE,), 0, len(df_.DOY) - 1)
            )
            X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
            y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

            covariance_function = Partial(kernel, jnp.inf)
            eps = 0.1

            K_ = cov_matrix(X_train, X_train, covariance_function)

            L_, alpha_ = inv_cov_chol_jax(K_, y_train, eps)
            K_trans = cov_matrix(X_train, X_test, covariance_function)
            mean_pred = jnp.dot(K_trans, alpha_)

            p1, p2, p3 = 0, 0, 0  # weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )

            weibull_pred.block_until_ready()

        return

    if MODE == "sparse":
        for tree in dendro.dendroNr.unique()[:N_TREES]:
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = jnp.sort(
                jax.random.randint(jaxkey, (X_TRAIN_SIZE,), 0, len(df_.DOY) - 1)
            )
            X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
            y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

            covariance_function = Partial(kernel, WENDLAND_LIMIT)
            eps = 0.1

            K_ = cov_matrix(X_train, X_train, covariance_function)

            L_, alpha_ = inv_cov_chol_sparse(K_, y_train, eps)
            K_trans = cov_matrix(X_train, X_test, covariance_function)
            mean_pred = jnp.dot(K_trans, alpha_)

            p1, p2, p3 = 0, 0, 0  # weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )

            weibull_pred.block_until_ready()

        return


import benchmark

# f(MODE="sparse")
# f(MODE="jax")
# f(MODE="band")
# f(MODE="band", X_TRAIN_SIZE=8000, WENDLAND_LIMIT=10)
print('----------')

# with jax.profiler.trace("/tmp/tensorboard"):
    # Run the operations to be profiled
    # f(MODE="band", X_TRAIN_SIZE=8000, WENDLAND_LIMIT=10)
    # f(MODE="jax")
    # f(MODE="band")
    # f(MODE="full")

param_dicts = [
                  {"MODE": "band", "X_TRAIN_SIZE": y}
                  for y in
                  [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
              ] + \
              [
                  {"MODE": "sparse", "X_TRAIN_SIZE": y}
                  for y in
                  [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
              ] + \
              [{"MODE": "jax", "X_TRAIN_SIZE": y} for y in
               [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]] + \
              [{"MODE": "full", "X_TRAIN_SIZE": y} for y in
               [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]]


print(len(param_dicts), "Benchmarks")
print(*param_dicts, sep='\n')

benchmark.benchmark_suite(
    lambda **kwargs: functools.partial(f, **kwargs),
    param_dicts,
    name=sys.argv[0],
    target_total_secs=1100,
)

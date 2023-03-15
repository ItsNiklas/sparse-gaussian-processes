#!/usr/bin/env python
# coding: utf-8

import sys

import liesel_sparse
import pandas as pd
from jax.experimental import sparse
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *

sys.path.insert(0, "../")
from GP import *

jax.config.parse_flags_with_absl()


# ### Using Tree data

@jax.jit
def weibull_F(x, lambda_, k_):
    return 1 - jnp.exp(-((lambda_ * x) ** k_))


dendro = pd.read_feather("../data/17766_12.feather")
weibull_params = pd.read_csv("../data/17766_Wparams.csv", index_col=0, usecols=['index', '0', '1', '2'])


@jax.jit
def kernel(theta, x, y):
    # 37**2 * Matern(length_scale=3, length_scale_bounds=(2,16), nu = 2.5) + WhiteKernel(noise_level=.01, noise_level_bounds="fixed")
    return MaternKernel32(37 ** 2, 3, x, y) * WendlandTapering(3, theta, x, y)


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
    L = jax.jit(liesel_sparse.cholesky_sparse)(K, L_sp_idx).todense()
    alpha = jax.jit(liesel_sparse.solve_sparse)(K, data_y)

    return L, alpha


jaxkey = jax.random.PRNGKey(0)

def f(MODE: str = "band", X_TEST_SIZE: int = 500, X_TRAIN_SIZE: int = 183,# N_TREES: int = 70,
      WENDLAND_LIMIT: float = 8.0):
    X_test = jnp.linspace(0, dendro.DOY.max(), X_TEST_SIZE).reshape(-1, 1)

    if MODE == "full":
        rng = np.random.default_rng()

        for tree in dendro.dendroNr.unique():
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = np.sort(rng.integers(0, len(df_.DOY) - 1, X_TRAIN_SIZE))

            X_train = df_.DOY.iloc[idx_train].array.reshape(-1, 1)
            y_train = df_.deltagrowth.iloc[idx_train]

            kernel_ = 37 ** 2 * Matern(
                length_scale=3, length_scale_bounds=(2, 16), nu=2.5
            ) + WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
            gp_model = gaussian_process.GaussianProcessRegressor(
                kernel=kernel_,
                # n_restarts_optimizer=2,
                # normalize_y=True,
                optimizer=None,
            )

            gp_model.fit(X_train, y_train)
            mean_pred = gp_model.predict(X_test, False)
            p1, p2, p3 = weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )

            # print(gp_model.log_marginal_likelihood_value_)
        return

    # # Band
    if MODE == "band":
        for tree in dendro.dendroNr.unique():
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = jnp.sort(
                jax.random.randint(jaxkey, (X_TRAIN_SIZE,), 0, len(df_.DOY) - 1)
            )
            X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
            y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

            covariance_function = Partial(kernel, WENDLAND_LIMIT)
            eps = 0.01

            K_ = cov_matrix(X_train, X_train, covariance_function)

            L_, alpha_ = inv_cov_chol(K_, y_train, eps, int(jax.jit(band.bandwidth)(K_)))
            K_trans = cov_matrix(X_train, X_test, covariance_function)
            mean_pred = jnp.dot(K_trans, alpha_)

            p1, p2, p3 = weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )
            weibull_pred.block_until_ready()

        return

    if MODE == "jax":
        for tree in dendro.dendroNr.unique():
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = jnp.sort(
                jax.random.randint(jaxkey, (X_TRAIN_SIZE,), 0, len(df_.DOY) - 1)
            )
            X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
            y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

            covariance_function = Partial(kernel, WENDLAND_LIMIT)
            eps = 0.01

            K_ = cov_matrix(X_train, X_train, covariance_function)

            L_, alpha_ = inv_cov_chol_jax(K_, y_train, eps)
            K_trans = cov_matrix(X_train, X_test, covariance_function)
            mean_pred = jnp.dot(K_trans, alpha_)

            p1, p2, p3 = weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )
            weibull_pred.block_until_ready()

        return

    if MODE == "sparse":
        for tree in dendro.dendroNr.unique():
            df_ = dendro.loc[dendro.dendroNr.eq(tree)]

            idx_train = jnp.sort(
                jax.random.randint(jaxkey, (X_TRAIN_SIZE,), 0, len(df_.DOY) - 1)
            )
            X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
            y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

            covariance_function = Partial(kernel, WENDLAND_LIMIT)
            eps = 0.01

            K_ = cov_matrix(X_train, X_train, covariance_function)

            L_, alpha_ = inv_cov_chol_sparse(K_, y_train, eps)
            K_trans = cov_matrix(X_train, X_test, covariance_function)
            mean_pred = jnp.dot(K_trans, alpha_)

            p1, p2, p3 = weibull_params.loc[tree].values
            weibull_pred = mean_pred + p1 * weibull_F(
                X_test.ravel(), p2, p3
            )
            weibull_pred.block_until_ready()

        return

import benchmark

#param_dicts = [{"MODE": "sparse", "WENDLAND_LIMIT" : 16}, {"MODE" : "jax", "WENDLAND_LIMIT" : 16}, {"MODE": "band", "WENDLAND_LIMIT" : 16}, {"MODE": "full", "WENDLAND_LIMIT" : None}]

param_dicts = [
                  {"WENDLAND_LIMIT": x, "MODE": "band", "X_TRAIN_SIZE" : y}
                  for x in
                  [40,60,80,100,120,140,160,180,200, np.inf,]
                  for y in
                  [50,75,100,125,150,175]
              ] + \
              [
                  {"WENDLAND_LIMIT": x, "MODE": "sparse", "X_TRAIN_SIZE" : y}
                  for x in
                  [40,60,80,100,120,140,160,180,200, np.inf,]
                  for y in
                  [50,75,100,125,150,175]
              ] + \
              [{"WENDLAND_LIMIT": None, "MODE": "jax", "X_TRAIN_SIZE" : y} for y in
                  [50,75,100,125,150,175]] + \
              [{"WENDLAND_LIMIT": None, "MODE": "full", "X_TRAIN_SIZE" : y} for y in
                  [50,75,100,125,150,175]]

print(len(param_dicts), "Benchmarks")

benchmark.benchmark_suite(
    lambda **kwargs: functools.partial(f, **kwargs),
    param_dicts,
    name=sys.argv[0],
    target_total_secs=2,
)

# f(MODE="sparse").block_until_ready()
# f(MODE="jax").block_until_ready()
# f(MODE="band").block_until_ready()
# f(MODE="full").block_until_ready()
# with jax.profiler.trace("/tmp/tensorboard"):
#     # Run the operations to be profiled
#     f(MODE="sparse").block_until_ready()
#     f(MODE="jax").block_until_ready()
#     f(MODE="band").block_until_ready()
#     f(MODE="full").block_until_ready()
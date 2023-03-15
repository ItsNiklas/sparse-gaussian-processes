#!/usr/bin/env python
# coding: utf-8
import liesel_sparse
from jax.experimental import sparse
from sklearn.datasets import fetch_openml

NEW_DATA: bool = True

import datetime
import sys

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

sys.path.insert(0, "../")
from GP import *

jax.config.parse_flags_with_absl()

# Read data
if not NEW_DATA:
    co2 = fetch_openml(data_id=41187, as_frame=True, parser="pandas")
    co2_data = co2.frame
    co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
    co2_data = co2_data[["date", "co2"]].set_index("date")
    co2_data = co2_data.resample("M").mean().dropna(axis="index", how="any")
    X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
    y = co2_data["co2"].to_numpy()
else:
    df_ = pd.read_csv(r"../data/co2_mm_mlo.csv", skiprows=56)
    X = df_["decimal date"].to_numpy().reshape(-1, 1)
    y = df_.average.to_numpy()

today = datetime.datetime.now()
current_month = today.year + 30 + today.month / 12

X = jnp.array(X)
y = jnp.array(y)


# Outside function scope to avoid recompilation for every loop,
# as a new PyTree object is created each time using Partial causing permanent
# cache misses. See #10868, #14743, etc.
@jax.jit
def kernel(theta, x__, y__):
    return (
            MaternKernel52(112 ** 2, 115, x__, y__)
            + MaternKernel52(2.58 ** 2, 199, x__, y__)
            * ExpSineSquaredKernel(1, 1.36, 1, x__, y__)
            + RationalQuadraticKernel(0.575 ** 2, 1.05, 0.672, x__, y__)
            + MaternKernel52(0.208 ** 2, 0.128, x__, y__)
    ) * WendlandTapering(3, theta, x__, y__)


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
    print(K.nse, K.shape, data_y.shape)

    # Solve Kα=y using the Cholesky decomposition.
    L_sp_idx = jnp.argwhere(jax.jit(liesel_sparse.symbolic_factorization)(K) > 0)
    L = jax.jit(liesel_sparse.cholesky_sparse)(K, L_sp_idx).todense()
    alpha = jax.jit(liesel_sparse.solve_sparse)(K, data_y)

    return L, alpha


def f(MODE: str = "band", X_TEST_SIZE: int = 1000, WENDLAND_LIMIT: float = 8.0):
    X_ = X
    y_ = y
    y_mean = y.mean()
    X_test = np.linspace(start=1958, stop=current_month, num=X_TEST_SIZE).reshape(-1, 1)

    if MODE == "full":
        # Set up GPR
        kernel_ = (
                112 ** 2 * Matern(length_scale=115, nu=5 / 2)
                + 2.58 ** 2
                * Matern(length_scale=199, nu=5 / 2)
                * ExpSineSquared(
            length_scale=1.36, periodicity=1, periodicity_bounds="fixed"
        )
                + 0.575 ** 2 * RationalQuadratic(alpha=0.672, length_scale=1.05)
                + 0.208 ** 2 * Matern(length_scale=0.128, nu=5 / 2)
                + WhiteKernel(noise_level=0.0382)
        )
        gpr = GaussianProcessRegressor(kernel=kernel_, optimizer=None)
        gpr.fit(X_, y_ - y_mean)

        # plt.figure(figsize=(12, 5))
        # y_samples = gpr.sample_y(X_test, 5)

        mean_y_pred = gpr.predict(X_test, return_std=False)
        mean_y_pred += y_mean
        # print(gpr.log_marginal_likelihood_value_)
        # plt.plot(X_, y_, color="black", linestyle="dashed", label="Measurements")
        # plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
        # plt.fill_between(
        #     X_test.ravel(),
        #     mean_y_pred - std_y_pred,
        #     mean_y_pred + std_y_pred,
        #     color="tab:blue",
        #     alpha=0.2,
        # )

        # for idx, single_prior in enumerate(y_samples.T):
        #     plt.plot(X_test, single_prior + y_mean, alpha=0.4, lw=1, linestyle="dashed")

        # plt.legend()
        # plt.xlabel("Year")
        # plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
        # plt.title("Monthly average of air samples measurements\nfrom the Mauna Loa Observatory")
        return mean_y_pred

    if MODE == "band":
        data_x = X_
        data_y = y_ - y_mean
        covariance_function = Partial(kernel, WENDLAND_LIMIT)
        eps = 0.0382

        K_ = cov_matrix(data_x, data_x, covariance_function)

        L_, alpha_ = inv_cov_chol(K_, data_y, eps, int(jax.jit(band.bandwidth)(K_)))
        K_trans = cov_matrix(data_x, X_test, covariance_function)
        mean_y_pred = jnp.dot(K_trans, alpha_) + y_mean
        return mean_y_pred.block_until_ready()

    if MODE == "jax":
        data_x = X_
        data_y = y_ - y_mean
        covariance_function = Partial(kernel, jnp.inf)
        eps = 0.0382

        K_ = cov_matrix(data_x, data_x, covariance_function)

        L_, alpha_ = inv_cov_chol_jax(K_, data_y, eps)
        K_trans = cov_matrix(data_x, X_test, covariance_function)
        mean_y_pred = jnp.dot(K_trans, alpha_) + y_mean

        # print("NMLL:", -(
        #         -0.5 * jnp.dot(data_y, alpha_)
        #         - (jnp.log(jnp.diag(L_))).sum()
        #         - 0.5 * L_.shape[0] * jnp.log(2 * jnp.pi)
        # ))

        return mean_y_pred.block_until_ready()

    if MODE == "sparse":
        data_x = X_
        data_y = y_ - y_mean
        covariance_function = Partial(kernel, WENDLAND_LIMIT)
        eps = 0.0382

        K_ = cov_matrix(data_x, data_x, covariance_function)

        L_, alpha_ = inv_cov_chol_sparse(K_, data_y, eps)
        K_trans = cov_matrix(data_x, X_test, covariance_function)
        mean_y_pred = jnp.dot(K_trans, alpha_) + y_mean

        # print("NMLL:", -(
        #         -0.5 * jnp.dot(data_y, alpha_)
        #         - (jnp.log(jnp.diag(L_))).sum()
        #         - 0.5 * L_.shape[0] * jnp.log(2 * jnp.pi)
        # ))

        return mean_y_pred.block_until_ready()


import benchmark

# param_dicts = [
#                   {"WENDLAND_LIMIT": x, "MODE": "band"}
#                   for x in
#                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, np.inf, ]
#               ] + \
#               [
#                   {"WENDLAND_LIMIT": x, "MODE": "sparse"}
#                   for x in
#                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, np.inf, ]
#               ] + \
#               [{"WENDLAND_LIMIT": None, "MODE": "jax"}] + \
#               [{"WENDLAND_LIMIT": None, "MODE": "full"}]
# param_dicts = [{"MODE": "sparse", "WENDLAND_LIMIT" : np.inf}, {"MODE" : "jax", "WENDLAND_LIMIT" : np.inf}, {"MODE": "band", "WENDLAND_LIMIT" : np.inf}, {"MODE": "full", "WENDLAND_LIMIT" : np.inf}]
param_dicts = [
                  {"WENDLAND_LIMIT": x, "MODE": "sparse"}
                  for x in
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, np.inf, ]
              ]

benchmark.benchmark_suite(
    lambda **kwargs: functools.partial(f, **kwargs),
    param_dicts,
    name=sys.argv[0],
    target_total_secs=.1,
)

# f(MODE="sparse").block_until_ready()
# print("---------")
# f(MODE="sparse").block_until_ready()
# with jax.profiler.trace("/tmp/tensorboard"):
#     # Run the operations to be profiled
#     f(MODE="sparse").block_until_ready()

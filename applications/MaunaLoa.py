#!/usr/bin/env python
# coding: utf-8

NEW_DATA: bool = True

import datetime
import sys

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

import benchmark

sys.path.insert(0, "../")
from GP import *

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

def f(MODE: str = "sparse", X_TEST_SIZE: int = 1000, WENDLAND_LIMIT: float = 8.0):
    X_ = X.copy()
    y_ = y.copy()
    X_test = np.linspace(start=1958, stop=current_month, num=X_TEST_SIZE).reshape(-1, 1)

    if MODE == "full":
        # Set up GPR
        kernel_ = (
            112**2 * Matern(length_scale=115, nu=5 / 2)
            + 2.58**2
            * Matern(length_scale=199, nu=5 / 2)
            * ExpSineSquared(length_scale=1.36, periodicity=1, periodicity_bounds="fixed")
            + 0.575**2 * RationalQuadratic(alpha=0.672, length_scale=1.05)
            + 0.208**2 * Matern(length_scale=0.128, nu=5 / 2)
            + WhiteKernel(noise_level=0.0382)
        )
        gpr = GaussianProcessRegressor(kernel=kernel_, optimizer=None)
        y_mean = y_.mean()
        gpr.fit(X_, y_ - y_mean)

        # plt.figure(figsize=(12, 5))
        # y_samples = gpr.sample_y(X_test, 5)

        mean_y_pred = gpr.predict(X_test, return_std=False)
        mean_y_pred += y_mean
        #print(gpr.log_marginal_likelihood_value_)
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

    if MODE == "sparse":
        ##### BAND
        @jax.jit
        def kernel_(x__, y__):
            return (
                MaternKernel52(112**2, 115, x__, y__)
                + MaternKernel52(2.58**2, 199, x__, y__)
                * ExpSineSquaredKernel(1, 1.36, 1, x__, y__)
                + RationalQuadraticKernel(0.575**2, 1.05, 0.672, x__, y__)
                + MaternKernel52(0.208**2, 0.128, x__, y__)
            ) * WendlandTapering(3, WENDLAND_LIMIT, x__, y__)

        y_mean = y_.mean()
        gpr = GPR(
            X_,
            y_ - y_mean,
            kernel_,
            jnp.empty(0),
            eps=0.0382,
        )
        # gpr.fit()
        # plt.figure(figsize=(12, 5))

        mean_y_pred = gpr.predict(X_test, return_std=False) +  y_mean
        # plt.plot(X_, y_, color="black", linestyle="dashed", label="Measurements")
        # plt.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
        # plt.fill_between(
        #     X_test.ravel(),
        #     mean_y_pred - std_y_pred,
        #     mean_y_pred + std_y_pred,
        #     color="tab:blue",
        #     alpha=0.2,
        # )
        # plt.legend()
        # plt.xlabel("Year")
        # plt.ylabel("Monthly average of CO$_2$ concentration (ppm)")
        # plt.title("Monthly average of air samples measurements\nfrom the Mauna Loa Observatory")
        return mean_y_pred.block_until_ready()

#param_dicts = [{"WENDLAND_LIMIT": x, "MODE": 'sparse'} for x in np.linspace(1,100,2)] + [{"WENDLAND_LIMIT": None, "MODE" : 'full'}]
param_dicts = [{"X_TEST_SIZE": 10**x, "MODE": 'full'} for x in range(1)]

benchmark.benchmark_suite(
    lambda **kwargs: functools.partial(f, **kwargs),
    param_dicts,
    name=sys.argv[0],
    target_total_secs=2,
)

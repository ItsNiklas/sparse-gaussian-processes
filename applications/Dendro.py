#!/usr/bin/env python
# coding: utf-8
import sys

import jax.numpy as jnp
import jaxopt
import pandas as pd
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import *
from tqdm import tqdm


# ### Using Tree data


def weibull_F(x, lambda_, k_):
    return 1 - jnp.exp(-((lambda_ * x) ** k_))


dendro = pd.read_feather("../data/17766_12.feather")

# # Fit models
# dendro_spcs = dendro[dendro.species.eq("Beech")]

dendro["deltagrowth"] = np.nan
weibull_params = dict()

for tree in tqdm(dendro.dendroNr.unique()):
    df_ = dendro[dendro.dendroNr.eq(tree)]
    y_ = jnp.array(df_.growth)
    x_ = jnp.array(df_.DOY)

    def f_(params):
        p0, p1, p2 = params
        return jnp.mean((y_ - (p0 * weibull_F(x_, p1, p2))) ** 2)  # MSE Loss
        # return jnp.mean(
        #     jax.vmap(jax.tree_util.Partial(jaxopt.loss.huber_loss, delta = 10.))
        #     (y_, p0 * weibull_F(x_, p1, p2)) # Huber Loss
        # )

    solver = jaxopt.ScipyBoundedMinimize(fun=f_)
    res = solver.run(
        jnp.array([max(y_), 1 / (0.632 * max(y_)), 3]),
        jnp.array([(0.1, 0.00001, 1), (100000, 1, 100)]),
    )
    weibull = lambda x__: res.params[0] * weibull_F(x__.ravel(), res.params[1], res.params[2])

    # with np.printoptions(precision=3, suppress=True, threshold=5, floatmode="fixed"):
    #     print(tree, res.params, res.state.fun_val, res.state.status, res.state.iter_num, sep = '\t')

    dendro.loc[dendro["dendroNr"] == tree, "deltagrowth"] = y_ - weibull(x_)
    weibull_params[tree] = res.params

if "full" in sys.argv:
    # plt.figure(figsize=(16, 7))

    rng = np.random.default_rng()

    X_test = np.linspace(0, dendro.DOY.max(), 500).reshape(-1, 1)

    c = lambda s: 0 if s == "Beech" else (3 if s == "Sycamore" else 1)

    for tree in tqdm(dendro.dendroNr.unique()[:25]):
        df_ = dendro.loc[dendro.dendroNr.eq(tree)]

        idx_train = np.sort(rng.integers(0, len(df_.DOY) - 1, 30))

        X_train = df_.DOY.iloc[idx_train].array.reshape(-1, 1)
        y_train = df_.deltagrowth.iloc[idx_train]

        kernel_ = 37**2 * Matern(
            length_scale=3, length_scale_bounds=(2, 16), nu=2.5
        ) + WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
        gp_model = gaussian_process.GaussianProcessRegressor(
            kernel=kernel_,
            # n_restarts_optimizer=2,
            # normalize_y=True,
            optimizer=None,
        )

        gp_model.fit(X_train, y_train)
        mean_pred, std_pred = gp_model.predict(X_test, True)
        # y_samples = gp_model.sample_y(X_test, 1)

        weibull_pred = weibull_params[tree][0] * weibull_F(
            X_test.ravel(), *weibull_params[tree][1:3]
        )

        print(gp_model.log_marginal_likelihood_value_)

        # plt.plot(X_test, mean_pred + weibull_pred, lw=1, zorder=10, alpha = .7, c=sns.color_palette()[c(df_.species.iloc[0])])

        # plt.plot(df_.DOY, df_.growth, lw=1, zorder=10, alpha = .5, c=sns.color_palette("pastel")[c(df_.species.iloc[0])])

        # plt.fill_between(
        #     X_test.ravel(),
        #     mean_pred + weibull_pred - 1.96 * std_pred,
        #     mean_pred + weibull_pred + 1.96 * std_pred,
        #     alpha=.1, zorder=0, color=sns.color_palette("pastel")[c(df_.species.iloc[0])]
        # )

        # for idx, single_prior in enumerate(y_samples.T):
        #     plt.plot(
        #         X_test, single_prior + weibull_pred
        #         , alpha=.4, lw=1, linestyle="dashed", c=sns.color_palette("pastel")[c(df_.species.iloc[0])]
        #     )

    # plt.xlabel("DOY")
    # plt.ylabel("growth")
    # plt.xlim(0, dendro.DOY.max())
    # plt.show()


# # SPARSE
else:

    sys.path.insert(0, "../")
    from GP import *

    # plt.figure(figsize=(16, 7))

    X_test = jnp.linspace(0, dendro.DOY.max(), 500).reshape(-1, 1)

    c = lambda s: 0 if s == "Beech" else (3 if s == "Sycamore" else 1)

    for tree in tqdm(dendro.dendroNr.unique()[:25]):
        df_ = dendro.loc[dendro.dendroNr.eq(tree)]

        # idx_train = jnp.sort(rng.integers(0, len(df_.DOY) - 1, 30))
        idx_train = jnp.sort(jax.random.randint(jax.random.PRNGKey(0), (30,), 0, len(df_.DOY) - 1))
        X_train = jnp.array(df_.DOY.iloc[idx_train].array).reshape(-1, 1)
        y_train = jnp.array(df_.deltagrowth.iloc[idx_train].array)

        # kernel_ = 37**2 * Matern(length_scale=3, length_scale_bounds=(2,16), nu = 2.5) + WhiteKernel(noise_level=.01, noise_level_bounds="fixed")
        # gp_model = gaussian_process.GaussianProcessRegressor(
        #     kernel=kernel_,
        #     #n_restarts_optimizer=2,
        #     #normalize_y=True,
        #     #optimizer=None
        # )
        def kernel_(s, l, x, y):
            # 37**2 * Matern(length_scale=3, length_scale_bounds=(2,16), nu = 2.5) + WhiteKernel(noise_level=.01, noise_level_bounds="fixed")
            return MaternKernel32(s, l, x, y) * WendlandTapering(3, 8, x, y)

        gp_model = GPR(X_train, y_train, kernel_, jnp.array([37**2, 3]), eps=0.01)
        # gp_model.fit(X_train, y_train)
        mean_pred = gp_model.predict(X_test, False)

        weibull_pred = weibull_params[tree][0] * weibull_F(
            X_test.ravel(), *weibull_params[tree][1:3]
        )

        # plt.plot(X_test, mean_pred + weibull_pred, lw=1, zorder=10, alpha = .7, c=sns.color_palette()[c(df_.species.iloc[0])])

        # plt.plot(df_.DOY, df_.growth, lw=1, zorder=10, alpha = .5, c=sns.color_palette("pastel")[c(df_.species.iloc[0])])

        # plt.fill_between(
        #     X_test.ravel(),
        #     mean_pred + weibull_pred - 1.96 * std_pred,
        #     mean_pred + weibull_pred + 1.96 * std_pred,
        #     alpha=.1, zorder=0, color=sns.color_palette("pastel")[c(df_.species.iloc[0])]
        # )

    # plt.xlabel("DOY")
    # plt.ylabel("growth")
    # plt.xlim(0, dendro.DOY.max())
    # plt.show()

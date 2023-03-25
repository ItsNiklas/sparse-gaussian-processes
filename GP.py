import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from liesel_sparse import band

import functools

# Set x64 mode for increased precision
jax.config.update("jax_enable_x64", True)


### Kernels
# Most use the distance between x and y


# Radial Basis Function kernel
def RBFKernel(sigma_f, length, x, y):
    return (
        sigma_f * jnp.exp(-(jnp.linalg.norm(x - y) ** 2) / (2 * length**2))
    ).astype(float)


# Exponential Sine Squared kernel
def ExpSineSquaredKernel(sigma_f, length, periodictiy, x, y):
    return (
        sigma_f
        * jnp.exp(
            -2 * (jnp.sin(jnp.pi * jnp.linalg.norm(x - y) / periodictiy) / length) ** 2
        )
    ).astype(float)


# Rational Quadratic kernel
def RationalQuadraticKernel(sigma_f, length, alpha, x, y):
    return (
        sigma_f
        * (1 + jnp.linalg.norm(x - y) ** 2 / (2 * alpha * length * length)) ** (-alpha)
    ).astype(float)


# Matern 3/2 kernel
def MaternKernel32(sigma_f, length_scale, x, y):
    arg = jnp.sqrt(3) * jnp.linalg.norm(x - y) / length_scale
    return (sigma_f * (1 + arg) * jnp.exp(-arg)).astype(float)


# Matern 3/2 kernel
def MaternKernel52(sigma_f, length_scale, x, y):
    arg = jnp.sqrt(5) * jnp.linalg.norm(x - y) / length_scale
    return (sigma_f * (1 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)).astype(float)


# Exact sparse kernel from Melkumyan and Ramos (2006)
def ExactSparse(sigma_f, length_scale, x, y):
    r = jnp.linalg.norm(x - y) / length_scale
    return jnp.where(
        r >= 1,
        0,
        (
            sigma_f
            * (
                (2 + jnp.cos(2 * jnp.pi * r)) / 3 * (1 - r)
                + jnp.sin(2 * jnp.pi * r) / (2 * jnp.pi)
            )
        ).astype(float),
    )


# Spherical taper
def SphericalTapering(theta, x, y):
    # Valid taper for Matern v <= 0.5
    r = jnp.linalg.norm(x - y) / theta
    # return jnp.maximum(0, 1 - r)
    return jnp.maximum(0, 1 - r) ** 2 * (r / 2 + 1)


# Wendland tapering for different values of k
def WendlandTapering(k_, theta, x, y):
    # Valid taper for Matern v <= 2.5
    r = jnp.linalg.norm(x - y) / theta
    match k_:
        case 0:
            # d = 1, k = 0
            return jnp.maximum(0, 1 - r)
        case 1:
            # d = 1, k = 1
            return jnp.maximum(0, 1 - r) ** 3 * (3 * r + 1)
        case 2:
            # d = 1, k = 2
            return jnp.maximum(0, 1 - r) ** 5 * (8 * r * r + 5 * r + 1)
        case _:
            # d = 1, k = 3
            return jnp.maximum(0, 1 - r) ** 7 * (
                21 * r * r * r + 19 * r * r + 7 * r + 1
            )


### Implementation


@jax.jit
def cov_matrix(x1, x2, cov_function):
    # Returns the symmetric kernel matrix K given two input arrays x1 and x2
    # and a covariance function.
    return jax.vmap(lambda x_: jax.vmap(lambda y_: cov_function(x_, y_))(x1))(x2)


@functools.partial(jax.jit, static_argnames=["p"])
def log_likelihood(kernel_, params, data_x, data_y, eps, p):
    # Compute the negative marginal log likelihood given a kernel function,
    # its hyperparameters, input data x, output data y, a small epsilon, and the bandwidth p.
    K_ = cov_matrix(data_x, data_x, Partial(kernel_, *params))
    Lb_, alpha_ = inv_cov_chol(K_, data_y, eps, p)

    return -(
        -0.5 * jnp.dot(data_y, alpha_)
        - (jnp.log(Lb_[0])).sum()
        - 0.5 * alpha_.shape[0] * jnp.log(2 * jnp.pi)
    )


@functools.partial(jax.jit, static_argnames=["p"])
def inv_cov_chol(K, data_y, eps, p):
    # Compute the inverse of the cholesky factorization of the covariance matrix K
    # given output data y, a small epsilon, and the bandwidth p.
    # Alternatively using permutations (not required for linear data)
    Kp = K
    # Kp, idx = band.permute(K)
    # Inverse permutation to map solution back
    # idx_inv = jnp.empty_like(idx).at[idx].set(jnp.arange(len(idx), dtype=int))
    Kpb = band.to_band(Kp, p)
    Kpb = Kpb.at[0].add(eps)
    Lb = band.cholesky_band(Kpb)
    alpha = band.solve_band(Lb, data_y)

    return Lb, alpha


class GPR:
    def __init__(self, data_x, data_y, kernel_, params, eps=1e-10):
        # Initialize the GPR model given input data x and output data y,
        # a kernel function and its hyperparameters, a small epsilon,
        # and initialize the model parameters.
        self.data_x = data_x
        self.data_y = data_y
        self.kernel_ = kernel_
        self.covariance_function = Partial(kernel_, *params)
        self.params = params
        self.eps = eps

        self.K_ = None
        self.Lb_ = None
        self.alpha_ = None
        self.log_marginal_likelihood_value_ = None
        self.idx = None

    def predict(self, at_values, return_std=False):
        # Given new input values at_values, predict their output values
        # using the trained GPR model. If return_std is True, return
        # the predicted output values and their standard deviation.
        if self.alpha_ is None:
            # If the alpha values are not computed yet, compute them
            # using the input data x and output data y.
            self.K_ = cov_matrix(self.data_x, self.data_x, self.covariance_function)
            bw = int(band.bandwidth(self.K_))
            self.Lb_, self.alpha_ = inv_cov_chol(self.K_, self.data_y, self.eps, p=bw)

        # Compute the mean prediction at the new input values.
        K_trans = cov_matrix(self.data_x, at_values, self.covariance_function)
        y_mean = jnp.dot(K_trans, self.alpha_)

        # If requested, compute and return the standard deviation of the
        # prediction at the new input values.
        if return_std:
            # Compute the matrix of coefficients for the linear system to solve
            # for the variance.
            V = jax.scipy.linalg.solve_triangular(
                band.to_ltri_full(self.Lb_), K_trans.T, lower=True
            )

            # Compute the diagonal elements of the covariance matrix at the new
            # input values.
            y_var = jnp.diag(cov_matrix(at_values, at_values, self.covariance_function))

            # Compute the residual variance.
            y_var -= jnp.einsum("ij,ji->i", V.T, V)

            # Set any negative values of the variance to zero.
            y_var = y_var.at[jnp.argwhere(y_var < 0)].set(0)

            return y_mean, jnp.sqrt(y_var)

        # If not requested, return only the mean prediction.
        return y_mean

    def fit_gd(self):
        import optax

        # Define a function to return an update function for the optimizer
        def get_update_fn(optimizer_):
            def update(params_, opt_state_):
                # Define an update function that takes in the current parameters and optimizer_ state
                # and returns the updated parameters and optimizer_ state
                grads = jax.grad(Partial(log_likelihood, self.kernel_))(
                    params_, self.data_x, self.data_y
                )
                updates, opt_state_ = optimizer_.update(
                    grads, opt_state_, params_=params_
                )
                params_ = optax.apply_updates(params_, updates)
                return params_, opt_state_

            # Return a jitted version of the update function
            return jax.jit(update)

        # Define the optimizer to be used
        optimizer = optax.sgd(learning_rate=3e-3)

        # Initialize the parameters and optimizer state
        params = self.params
        opt_state = optimizer.init(params)

        # Get the update function for the optimizer
        update_f = get_update_fn(optimizer)

        # Update the parameters using the optimizer for a fixed number of iterations
        for j in range(1000):
            params, opt_state = update_f(params, opt_state)

        # Save the updated parameters and the covariance function using these parameters
        self.params = params
        self.covariance_function = Partial(self.kernel_, *params)
        self.log_marginal_likelihood_value_ = log_likelihood(
            self.kernel_, self.params, self.data_x, self.data_y, self.eps, bw
        )
        self.alpha_ = None

    def fit(self):
        # Use BFGS for Gradient Descent to optimize mll.
        # Only one (Unbounded) run is performed.
        import jaxopt

        # Calculate the bandwidth for the kernel using the data
        bw = int(
            band.bandwidth(
                cov_matrix(self.data_x, self.data_x, self.covariance_function)
            )
        )

        # Define a function that returns the value of the log-likelihood and its gradient
        def fwd(*args, **kwargs):
            return Partial(log_likelihood, self.kernel_, p=bw)(
                *args, **kwargs
            ), jax.jacfwd(Partial(log_likelihood, self.kernel_, p=bw))(*args, **kwargs)

        # Define the solver for the optimization problem
        solver = jaxopt.LBFGS(
            fun=fwd,
            value_and_grad=True,
            maxiter=200,
            min_stepsize=1e-3,
            tol=1e-1,
            stop_if_linesearch_fails=True,
            maxls=5,
            jit=True,
        )

        # Run the optimization solver to find the parameters that maximize the log-likelihood
        soln = solver.run(
            self.params,
            data_x=self.data_x,
            data_y=self.data_y,
            eps=self.eps,
        )

        # Save the optimized parameters and compute the log-marginal likelihood using these parameters
        params, state = soln
        self.params = jnp.where(params > 0, params, 1e-5)
        self.covariance_function = Partial(self.kernel_, *self.params)
        self.log_marginal_likelihood_value_ = log_likelihood(
            self.kernel_, self.params, self.data_x, self.data_y, self.eps, bw
        )
        self.alpha_ = None

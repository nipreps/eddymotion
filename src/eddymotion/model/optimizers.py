# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY kIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
import numpy as np
from scipy.optimize import minimize


def negative_log_likelihood(beta, y, X, kernel, reg_param=1e-6):
    """
    Log marginal likelihood function with regularization for any kernel.

    Parameters
    ----------
    beta : array-like of shape (n_params,)
        Log-transformed hyperparameters.
    y : array-like of shape (n_samples,)
        Observed data.
    X : array-like of shape (n_samples, n_features)
        Input data for kernel function.
    kernel : callable
        Kernel function which takes hyperparameters and returns a kernel instance.
    reg_param : float, default=1e-6
        Regularization parameter.

    Returns
    -------
    float
        Negative log marginal likelihood.
    """
    kernel_instance = kernel(*np.exp(beta))
    K = kernel_instance(X)

    # Add regularization term
    K += reg_param * np.eye(len(X))

    # Check if the kernel matrix is positive definite
    eigenvalues = np.linalg.eigvals(K)
    if np.any(eigenvalues <= 0):
        print("Non-positive definite kernel matrix")
        return 1e10  # Penalize non-positive definite kernel

    log_likelihood = -0.5 * (
        np.dot(y.T, np.linalg.solve(K, y)) + np.linalg.slogdet(K)[1] + len(y) * np.log(2 * np.pi)
    )
    regularization = reg_param * (np.sum(beta**2))
    return -log_likelihood + regularization


def total_negative_log_likelihood(beta, y_all, X, kernel, reg_param=1e-6):
    """
    Total negative log marginal likelihood for all voxels for any kernel.

    Parameters
    ----------
    beta : array-like of shape (n_params,)
        Log-transformed hyperparameters.
    y_all : array-like of shape (n_samples, n_voxels)
        Observed data for all voxels.
    X : array-like of shape (n_samples, n_features)
        Input data for kernel function.
    kernel : callable
        Kernel function which takes hyperparameters and returns a kernel instance.
    reg_param : float, default=1e-6
        Regularization parameter.

    Returns
    -------
    float
        Total negative log marginal likelihood.
    """
    total_log_likelihood = 0
    for y in y_all.T:  # transposed to have shape (n_voxels, n_samples) and iterate over voxels
        total_log_likelihood += negative_log_likelihood(beta, y, X, kernel, reg_param)
    return total_log_likelihood


def loo_cross_validation(beta, y_all, X, kernel):
    """
    Leave-One-Out Cross-Validation for hyperparameter optimization for any kernel.

    Parameters
    ----------
    beta : array-like of shape (n_params,)
        Log-transformed hyperparameters.
    y_all : array-like of shape (n_samples, n_voxels)
        Observed data for all voxels.
    X : array-like of shape (n_samples, n_features)
        Input data for kernel function.
    kernel : callable
        Kernel function which takes hyperparameters and returns a kernel instance.

    Returns
    -------
    float
        Mean LOO-CV error.
    """
    kernel_instance = kernel(*np.exp(beta))
    K = kernel_instance(X)
    n = y_all.shape[0]

    errors = []
    for i in range(n):
        K_train = np.delete(np.delete(K, i, axis=0), i, axis=1)
        y_train = np.delete(y_all, i, axis=0)

        K_test = K[i, np.arange(K.shape[0]) != i]
        # Solve for alpha
        alpha = np.linalg.solve(K_train, y_train)

        # Compute prediction for left-out point
        y_pred = K_test @ alpha

        # Compute the error
        error = (y_all[i] - y_pred) ** 2
        errors.append(error)
    print(f"Error: {np.mean(errors)}")
    return np.mean(errors)


def stochastic_optimization_with_early_stopping(
    initial_beta, data, angles, kernel, batch_size, max_iter=10000, patience=100, tolerance=1e-4
):
    """
    Stochastic Optimization with Early Stopping.

    Parameters
    ----------
    initial_beta : array-like of shape (n_params,)
        Initial guess for the log-transformed hyperparameters.
    data : array-like of shape (n_samples, n_voxels)
        Observed data for all voxels.
    angles : array-like of shape (n_samples, n_features)
        Pairwise angles between gradient directions.
    kernel : Kernel instance
        A kernel instance with hyperparameters to be optimized.
    batch_size : int
        Size of the mini-batches for optimization.
    max_iter : int, default=10000
        Maximum number of iterations.
    patience : int, default=100
        Patience for early stopping.
    tolerance : float, default=1e-4
        Tolerance for improvement in loss.

    Returns
    -------
    best_beta : array-like of shape (n_params,)
        Optimized log-transformed hyperparameters.
    """
    beta = initial_beta
    best_loss = float("inf")
    no_improve_count = 0
    num_voxels = data.shape[1]

    for iteration in range(max_iter):
        batch_indices = np.random.choice(num_voxels, size=batch_size, replace=False)
        batch_data = data[:, batch_indices]

        result = minimize(
            total_negative_log_likelihood,
            beta,
            args=(batch_data, angles, kernel, 1e-6),
            method="L-BFGS-B",
        )
        current_loss = result.fun

        if iteration == 0 or current_loss < best_loss - tolerance:
            best_loss = current_loss
            best_beta = result.x
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at iteration {iteration + 1} due to no improvement")
            break

        beta = result.x
        print(f"Iteration {iteration + 1}: Loss = {current_loss}")
        print(f"Current hyperparameters: {np.exp(beta)}")

    return best_beta

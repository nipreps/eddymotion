import numpy as np
import pytest
from sklearn.gaussian_process.kernels import RBF

from eddymotion.model.optimizers import (
    loo_cross_validation,
    negative_log_likelihood,
    stochastic_optimization_with_early_stopping,
    total_negative_log_likelihood,
)


@pytest.fixture
def sample_data():
    # Generate synthetic data
    np.random.seed(0)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])
    return X, y


@pytest.fixture
def sample_multivoxel_data():
    # Generate synthetic multivoxel data
    np.random.seed(0)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_all = np.vstack([np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0]) for _ in range(10)]).T
    return X, y_all


def test_negative_log_likelihood(sample_data):
    X, y = sample_data

    # Log-transformed initial parameters
    initial_beta = np.log([1.0, 1.0])

    # Define a kernel factory function for RBF kernel
    def kernel_factory(length_scale, noise_level):
        return RBF(length_scale=length_scale) + noise_level * np.eye(len(X))

    # Compute the negative log likelihood
    nll = negative_log_likelihood(initial_beta, y, X, kernel_factory)

    # Check if the negative log likelihood is finite
    assert np.isfinite(nll), "Negative log likelihood should be finite."


def test_total_negative_log_likelihood(sample_multivoxel_data):
    X, y_all = sample_multivoxel_data

    # Log-transformed initial parameters
    initial_beta = np.log([1.0, 1.0])

    # Define a kernel factory function for RBF kernel
    def kernel_factory(length_scale, noise_level):
        return RBF(length_scale=length_scale) + noise_level * np.eye(len(X))

    # Compute the total negative log likelihood
    total_nll = total_negative_log_likelihood(initial_beta, y_all, X, kernel_factory)

    # Check if the total negative log likelihood is finite
    assert np.isfinite(total_nll), "Total negative log likelihood should be finite."


def test_loo_cross_validation(sample_multivoxel_data):
    X, y_all = sample_multivoxel_data

    # Log-transformed initial parameters
    initial_beta = np.log([1.0, 1.0])

    # Define a kernel factory function for RBF kernel
    def kernel_factory(length_scale, noise_level):
        return RBF(length_scale=length_scale) + noise_level * np.eye(len(X))

    # Compute the LOO-CV error
    loo_error = loo_cross_validation(initial_beta, y_all, X, kernel_factory)

    # Check if the LOO-CV error is finite
    assert np.isfinite(loo_error), "LOO-CV error should be finite."


def test_stochastic_optimization_with_early_stopping(sample_multivoxel_data):
    X, y_all = sample_multivoxel_data

    # Log-transformed initial parameters
    initial_beta = np.log([1.0, 1.0])

    # Define a kernel factory function for RBF kernel
    def kernel_factory(length_scale, noise_level):
        return RBF(length_scale=length_scale) + noise_level * np.eye(len(X))

    # Perform stochastic optimization with early stopping
    best_beta = stochastic_optimization_with_early_stopping(
        initial_beta,
        y_all,
        X,
        kernel_factory,
        batch_size=5,
        max_iter=50,
        patience=5,
        tolerance=1e-4
    )

    # Check if the best beta is finite
    assert np.all(np.isfinite(best_beta)), "Optimized hyperparameters should be finite."

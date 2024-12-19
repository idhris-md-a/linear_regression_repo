import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
# Generate synthetic data
np.random.seed(42)
n_samples = 50
X = np.linspace(500, 4000, n_samples).reshape(-1, 1) # Square footage
true_slope = 150 # Price per square foot
true_intercept = 50000 # Base price
noise_std = 20000
y = true_intercept + true_slope * X.flatten() + np.random.normal(0,
noise_std, size=n_samples)
# Add intercept term
X_design = np.hstack((np.ones_like(X), X))
# Bayesian Linear Regression
# Define priors
prior_mean = np.array([0, 0]) # Prior for [intercept, slope]
prior_covariance = np.eye(2) * 1e6 # Large prior variance
# Likelihood precision (precision = 1 / variance)
noise_variance = noise_std**2
likelihood_precision = np.eye(X_design.shape[1]) / noise_variance
# Posterior parameters
posterior_precision = np.linalg.inv(np.linalg.inv(prior_covariance) +
X_design.T @ X_design / noise_variance)
posterior_mean = posterior_precision @ (np.linalg.inv(prior_covariance) @
prior_mean + X_design.T @ y / noise_variance)
# Predictive distribution
X_new = np.linspace(500, 4000, 100).reshape(-1, 1)
X_new_design = np.hstack((np.ones_like(X_new), X_new))
predictive_mean = X_new_design @ posterior_mean
predictive_var = np.sum(X_new_design @ posterior_precision *
X_new_design, axis=1) + noise_variance
predictive_std = np.sqrt(predictive_var)
# Plotting results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Observed Data", alpha=0.7)
plt.plot(X_new, predictive_mean, color="red", label="Mean Prediction")
plt.fill_between(
 X_new.flatten(),
 predictive_mean - 2 * predictive_std,
 predictive_mean + 2 * predictive_std,
 color="pink",
 alpha=0.5,
 label="95% Predictive Interval",
)
plt.title("Bayesian Linear Regression: Predicting House Prices")
plt.xlabel("Square Footage")
plt.ylabel("House Price ($)")
plt.legend()
plt.show()

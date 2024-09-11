import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import generate_multivariate_synthetic_data

class VARModel:
    def __init__(self, data, lags):
        """
        Vector Autoregression (VAR) model
        
        Args:
        data: 2D numpy array, where each column is a time series
        lags: Number of lags to include in the model
        """
        self.data = data
        self.lags = lags
        self.n_series = data.shape[1]
        self.coefficients = None
        self.intercepts = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None
    
    def _prepare_lagged_matrix(self):
        n_obs = self.data.shape[0]
        X, y = [], []
        for i in range(self.lags, n_obs):
            row = []
            for j in range(self.lags):
                row.extend(self.data[i-j-1])
            X.append(row)
            y.append(self.data[i])
        return np.array(X), np.array(y)
    
    def fit(self):
        X, y = self._prepare_lagged_matrix()
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Fit the model using OLS for each series
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        self.intercepts = beta[0]
        self.coefficients = beta[1:].reshape(self.lags, self.n_series, self.n_series)
        self.fitted_values = X_with_intercept @ beta
        self.residuals = y - self.fitted_values
        
        # Calculate AIC and BIC
        n, k = y.shape[0], beta.shape[0]
        rss = np.sum(self.residuals**2)
        self.aic = n * np.log(np.linalg.det(rss / n)) + 2 * k
        self.bic = n * np.log(np.linalg.det(rss / n)) + k * np.log(n)
    
    def predict(self, steps=1):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        history = self.data[-self.lags:].copy()
        predictions = []
        
        for _ in range(steps):
            next_val = self.intercepts.copy()
            for i in range(self.lags):
                next_val += self.coefficients[i] @ history[-i-1]
            predictions.append(next_val)
            history = np.vstack([history[1:], next_val])
        
        return np.array(predictions)

def plot_multivariate_data(data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data[:, 0], label='Series 1', color='blue')
    plt.plot(data[:, 1], label='Series 2', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def fit_var_models_and_calculate_criteria(data, max_lags=6):
    models, aic_values, bic_values = [], [], []
    for lag in range(1, max_lags + 1):
        model = VARModel(data, lags=lag)
        model.fit()
        models.append(model)
        aic_values.append(model.aic)
        bic_values.append(model.bic)
    return models, aic_values, bic_values

def plot_var_results(data, fitted_values, future_time, predicted_data, optimal_lag):
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plot_multivariate_data(data, 'Multivariate Synthetic Data')

    plt.subplot(3, 1, 2)
    plt.plot(data[:, 0], label='Original Series 1', color='blue', linestyle='dotted')
    plt.plot(data[:, 1], label='Original Series 2', color='red', linestyle='dotted')
    plt.plot(fitted_values[:, 0], label='Fitted Series 1', color='blue')
    plt.plot(fitted_values[:, 1], label='Fitted Series 2', color='red')
    plt.title(f'Fitted VAR Model with Optimal Lag = {optimal_lag}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(data[:, 0], label='Original Series 1', color='blue', linestyle='dotted')
    plt.plot(data[:, 1], label='Original Series 2', color='red', linestyle='dotted')
    plt.plot(future_time, predicted_data[:, 0], label='Predicted Series 1', color='blue')
    plt.plot(future_time, predicted_data[:, 1], label='Predicted Series 2', color='red')
    plt.title(f'Future Predictions using VAR Model (Lag={optimal_lag})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(future_steps=50):
    np.random.seed(42)
    synthetic_data = generate_multivariate_synthetic_data()
    plot_multivariate_data(synthetic_data, 'Multivariate Synthetic Data')

    models, aic_values, bic_values = fit_var_models_and_calculate_criteria(synthetic_data)
    
    optimal_lag_aic = np.argmin(aic_values) + 1
    optimal_model_aic = models[optimal_lag_aic - 1]

    fitted_values = np.empty_like(synthetic_data)
    fitted_values[:] = np.nan
    fitted_values[optimal_lag_aic:] = optimal_model_aic.fitted_values
  
    predicted_data = optimal_model_aic.predict(steps=future_steps)
    future_time = np.arange(len(synthetic_data), len(synthetic_data) + future_steps)

    plot_var_results(synthetic_data, fitted_values, future_time, predicted_data, optimal_lag_aic)

    print(f"Optimal lag (AIC): {optimal_lag_aic}")
    print(f"AIC values: {aic_values}")
    print(f"BIC values: {bic_values}")

if __name__ == "__main__":
    main()

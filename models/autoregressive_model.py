import numpy as np
import matplotlib.pyplot as plt
from utils import generate_synthetic_data

class AutoregressiveModel:
    def __init__(self, data, lags):
        """
        Linear autoregressive model
        
        """
        self.data = data
        self.lags = lags
        self.coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
    
    def _prepare_lagged_matrix(self):
        X, y = [], []
        for i in range(self.lags, len(self.data)):
            X.append(self.data[i - self.lags:i][::-1])
            y.append(self.data[i])
        return np.array(X), np.array(y)
    
    def fit(self):
        X, y = self._prepare_lagged_matrix()
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.fitted_values = X_with_intercept @ beta
        self.residuals = y - self.fitted_values
        n, k = len(y), self.lags + 1
        rss = np.sum(self.residuals**2)
        self.aic = n * np.log(rss / n) + 2 * k
    
    def predict(self, steps=1):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        history = list(self.data[-self.lags:])
        predictions = []
        
        for _ in range(steps):
            next_val = self.intercept + sum([coef * val for coef, val in zip(self.coefficients, history[::-1])])
            predictions.append(next_val)
            history.append(next_val)
            history.pop(0)
        
        return np.array(predictions)

def plot_synthetic_data(data):
    plt.figure(figsize=(12, 4))
    plt.plot(data, label='Synthetic Calcium Traces', color='blue')
    plt.title('Synthetic Calcium Traces')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def fit_models_and_calculate_aic(data, max_lags=6):
    models, aic_values = [], []
    for lag in range(1, max_lags + 1):
        model = AutoregressiveModel(data, lags=lag)
        model.fit()
        models.append(model)
        aic_values.append(model.aic)
    return models, aic_values

def plot_models(data, models):
    plt.figure(figsize=(12, 8))
    plt.plot(data, label='Original Data', color='blue', linestyle='dotted')
    
    for i, model in enumerate(models):
        fitted_values = np.empty_like(data)
        fitted_values[:] = np.nan
        fitted_values[model.lags:] = model.fitted_values
        plt.plot(fitted_values, label=f'AR Model (Lag={model.lags})')
    
    plt.title('AR Models with Different Lags')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def plot_final_results(data, fitted_values, future_time, predicted_data, optimal_lag):
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.plot(data, label='Synthetic Calcium Traces', color='blue')
    plt.title('Synthetic Calcium Traces')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(data, label='Original Data', color='blue')
    plt.plot(fitted_values, label=f'Fitted AR Model (Lag={optimal_lag})', color='red')
    plt.title(f'Fitted AR Model with Optimal Lag = {optimal_lag}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(data, label='Original Data', color='blue')
    plt.plot(future_time, predicted_data, label='Predicted Future Data', color='green')
    plt.title(f'Future Predictions using AR Model (Lag={optimal_lag})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(future_steps=50):
    synthetic_data = generate_synthetic_calcium_traces()
    plot_synthetic_data(synthetic_data)

    models, aic_values = fit_models_and_calculate_aic(synthetic_data)
    plot_models(synthetic_data, models)

    optimal_lag = np.argmin(aic_values) + 1
    optimal_model = models[optimal_lag - 1]

    fitted_values = np.empty_like(synthetic_data)
    fitted_values[:] = np.nan
    fitted_values[optimal_lag:] = optimal_model.fitted_values
  
    predicted_data = optimal_model.predict(steps=future_steps)
    future_time = np.arange(len(synthetic_data), len(synthetic_data) + future_steps)

    plot_final_results(synthetic_data, fitted_values, future_time, predicted_data, optimal_lag)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n_points=500, noise_level=0.1):
    t = np.linspace(0, 10, n_points)
    signal = (np.sin(2 * np.pi * t) + 
              0.5 * np.sin(4 * np.pi * t + np.pi/4) + 
              0.2 * np.sin(8 * np.pi * t + np.pi/2))
    signal /= np.max(np.abs(signal))
    noise = np.random.normal(0, noise_level, n_points)
    synthetic_data = signal + noise
    return synthetic_data

class MovingAverageModel:
    def __init__(self, data, order):
        self.data = data
        self.order = order
        self.coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
    
    def _prepare_ma_matrix(self):
        X, y = [], []
        for i in range(self.order, len(self.data)):
            X.append(self.data[i - self.order:i][::-1])
            y.append(self.data[i])
        return np.array(X), np.array(y)
    
    def fit(self):
        X, y = self._prepare_ma_matrix()
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.fitted_values = X_with_intercept @ beta
        self.residuals = y - self.fitted_values
        n, k = len(y), self.order + 1
        rss = np.sum(self.residuals**2)
        self.aic = n * np.log(rss / n) + 2 * k
    
    def predict(self, steps=1):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        history = list(self.residuals[-self.order:])
        predictions = []
        
        for _ in range(steps):
            next_val = self.intercept + sum([coef * val for coef, val in zip(self.coefficients, history[::-1])])
            predictions.append(next_val)
            history.append(0)  # Assume future errors are zero
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

def fit_ma_models_and_calculate_aic(data, max_order=6):
    models, aic_values = [], []
    for order in range(1, max_order + 1):
        model = MovingAverageModel(data, order=order)
        model.fit()
        models.append(model)
        aic_values.append(model.aic)
    return models, aic_values

def plot_ma_models(data, models):
    plt.figure(figsize=(12, 8))
    plt.plot(data, label='Original Data', color='blue', linestyle='dotted')
    
    for i, model in enumerate(models):
        fitted_values = np.empty_like(data)
        fitted_values[:] = np.nan
        fitted_values[model.order:] = model.fitted_values
        plt.plot(fitted_values, label=f'MA Model (Order={model.order})')
    
    plt.title('MA Models with Different Orders')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def plot_final_ma_results(data, fitted_values, future_time, predicted_data, optimal_order):
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plt.plot(data, label='Synthetic Calcium Traces', color='blue')
    plt.title('Synthetic Calcium Traces')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(data, label='Original Data', color='blue')
    plt.plot(fitted_values, label=f'Fitted MA Model (Order={optimal_order})', color='red')
    plt.title(f'Fitted MA Model with Optimal Order = {optimal_order}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(data, label='Original Data', color='blue')
    plt.plot(future_time, predicted_data, label='Predicted Future Data', color='green')
    plt.title(f'Future Predictions using MA Model (Order={optimal_order})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(future_steps=50):
    np.random.seed(42)
    synthetic_data = generate_synthetic_data()
    plot_synthetic_data(synthetic_data)

    models, aic_values = fit_ma_models_and_calculate_aic(synthetic_data)
    plot_ma_models(synthetic_data, models)

    optimal_order = np.argmin(aic_values) + 1
    optimal_model = models[optimal_order - 1]

    fitted_values = np.empty_like(synthetic_data)
    fitted_values[:] = np.nan
    fitted_values[optimal_order:] = optimal_model.fitted_values
  
    predicted_data = optimal_model.predict(steps=future_steps)
    future_time = np.arange(len(synthetic_data), len(synthetic_data) + future_steps)

    plot_final_ma_results(synthetic_data, fitted_values, future_time, predicted_data, optimal_order)

if __name__ == "__main__":
    main()

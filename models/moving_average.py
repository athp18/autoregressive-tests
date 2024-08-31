import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_synthetic_data(n_points=500, noise_level=0.1):
    np.random.seed(42)
    t = np.linspace(0, 10, n_points)
    signal = (np.sin(2 * np.pi * t) + 
              0.5 * np.sin(4 * np.pi * t + np.pi/4) + 
              0.2 * np.sin(8 * np.pi * t + np.pi/2))
    signal /= np.max(np.abs(signal))
    noise = np.random.normal(0, noise_level, n_points)
    synthetic_data = signal + noise
    return t, synthetic_data

class MovingAverageModel:
    def __init__(self, data, order):
        self.data = data
        self.order = order
        self.coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.mse = None
        self.r_squared = None
    
    def _prepare_ma_matrix(self):
        X, y = [], []
        for i in range(self.order, len(self.data)):
            X.append(self.data[i - self.order:i][::-1])
            y.append(self.data[i])
        return np.array(X), np.array(y)
    
    def fit(self):
        X, y = self._prepare_ma_matrix()
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Ordinary Least Squares estimation
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        self.intercept = beta[0]
        self.coefficients = beta[1:]
        self.fitted_values = X_with_intercept @ beta
        self.residuals = y - self.fitted_values
        
        n, k = len(y), self.order + 1
        
        # Calculate performance metrics
        self.mse = np.mean(self.residuals**2)
        tss = np.sum((y - np.mean(y))**2)
        self.r_squared = 1 - (np.sum(self.residuals**2) / tss)
        
        # Information criteria
        self.aic = n * np.log(self.mse) + 2 * k
        self.bic = n * np.log(self.mse) + k * np.log(n)
        
        # Standard errors of coefficients
        sigma2 = np.sum(self.residuals**2) / (n - k)
        var_beta = sigma2 * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        self.std_errors = np.sqrt(np.diag(var_beta))
        
        # t-statistics and p-values
        self.t_stats = beta / self.std_errors
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), n - k))
    
    def predict(self, steps=1):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        history = list(self.residuals[-self.order:])
        predictions = []
        
        for _ in range(steps):
            next_val = self.intercept + np.dot(self.coefficients, history[::-1])
            predictions.append(next_val)
            history.append(0)  # Assume future errors are zero
            history.pop(0)
        
        return np.array(predictions)

def plot_acf_pacf(data, lags=40):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot ACF
    acf = np.array([1] + [np.corrcoef(data[:-i], data[i:])[0, 1] for i in range(1, lags+1)])
    ax1.bar(range(lags+1), acf)
    ax1.axhline(y=0, color='k', linestyle='--')
    ax1.axhline(y=1.96/np.sqrt(len(data)), color='k', linestyle='--')
    ax1.axhline(y=-1.96/np.sqrt(len(data)), color='k', linestyle='--')
    ax1.set_title('Autocorrelation Function (ACF)')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')
    
    # Plot PACF
    pacf = np.zeros(lags+1)
    pacf[0] = 1
    for i in range(1, lags+1):
        y = data[i:]
        X = np.column_stack([data[i-j:-j] for j in range(1, i+1)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pacf[i] = beta[-1]
    
    ax2.bar(range(lags+1), pacf)
    ax2.axhline(y=0, color='k', linestyle='--')
    ax2.axhline(y=1.96/np.sqrt(len(data)), color='k', linestyle='--')
    ax2.axhline(y=-1.96/np.sqrt(len(data)), color='k', linestyle='--')
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Correlation')
    
    plt.tight_layout()
    plt.show()

def plot_model_diagnostics(model):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Residuals vs Fitted
    ax1.scatter(model.fitted_values, model.residuals)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(model.residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q')
    
    # Scale-Location
    standardized_residuals = model.residuals / np.sqrt(np.mean(model.residuals**2))
    ax3.scatter(model.fitted_values, np.sqrt(np.abs(standardized_residuals)))
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location')
    
    # Residuals vs Leverage
    leverage = np.diag(model._prepare_ma_matrix()[0] @ np.linalg.inv(model._prepare_ma_matrix()[0].T @ model._prepare_ma_matrix()[0]) @ model._prepare_ma_matrix()[0].T)
    ax4.scatter(leverage, standardized_residuals)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Standardized Residuals')
    ax4.set_title('Residuals vs Leverage')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate synthetic data
    t, synthetic_data = generate_synthetic_data()
    
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Synthetic Calcium Traces')
    plt.title('Synthetic Calcium Traces')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Plot ACF and PACF
    plot_acf_pacf(synthetic_data)
    
    # Fit MA models with different orders
    max_order = 6
    models = []
    aic_values = []
    bic_values = []
    
    for order in range(1, max_order + 1):
        model = MovingAverageModel(synthetic_data, order=order)
        model.fit()
        models.append(model)
        aic_values.append(model.aic)
        bic_values.append(model.bic)
    
    # Plot AIC and BIC
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_order + 1), aic_values, marker='o', label='AIC')
    plt.plot(range(1, max_order + 1), bic_values, marker='s', label='BIC')
    plt.xlabel('Model Order')
    plt.ylabel('Information Criterion')
    plt.title('Model Selection: AIC and BIC')
    plt.legend()
    plt.show()
    
    # Select optimal model based on AIC
    optimal_order = np.argmin(aic_values) + 1
    optimal_model = models[optimal_order - 1]
    
    # Plot fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Original Data')
    plt.plot(t[optimal_order:], optimal_model.fitted_values, label=f'Fitted MA({optimal_order})', color='red')
    plt.title(f'Moving Average Model - Order {optimal_order}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Plot model diagnostics
    plot_model_diagnostics(optimal_model)
    
    # Make future predictions
    future_steps = 50
    predictions = optimal_model.predict(steps=future_steps)
    future_time = np.arange(len(t), len(t) + future_steps)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Original Data')
    plt.plot(future_time, predictions, label='Predicted Data', color='green')
    plt.title(f'Future Predictions using MA({optimal_order}) Model')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Print model summary
    print(f"Optimal MA Model Summary (Order = {optimal_order}):")
    print(f"AIC: {optimal_model.aic:.4f}")
    print(f"BIC: {optimal_model.bic:.4f}")
    print(f"MSE: {optimal_model.mse:.4f}")
    print(f"R-squared: {optimal_model.r_squared:.4f}")
    print("\nCoefficients:")
    print(f"Intercept: {optimal_model.intercept:.4f}")
    for i, coef in enumerate(optimal_model.coefficients):
        print(f"MA{i+1}: {coef:.4f} (p-value: {optimal_model.p_values[i+1]:.4f})")

if __name__ == "__main__":
    main()

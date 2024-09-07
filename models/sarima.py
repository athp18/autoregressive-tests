import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_synthetic_data(n_points=500, noise_level=0.1, seasonal_period=12):
    np.random.seed(42)
    t = np.linspace(0, 10, n_points)
    trend = 0.1 * t
    seasonal = 0.5 * np.sin(2 * np.pi * t / seasonal_period)
    signal = (np.sin(2 * np.pi * t) + 
              0.5 * np.sin(4 * np.pi * t + np.pi/4) + 
              0.2 * np.sin(8 * np.pi * t + np.pi/2))
    signal = (signal + trend + seasonal) / np.max(np.abs(signal + trend + seasonal))
    noise = np.random.normal(0, noise_level, n_points)
    synthetic_data = signal + noise
    return t, synthetic_data

class SARIMAModel:
    def __init__(self, data, order, seasonal_order, seasonal_period):
        self.original_data = data
        self.p, self.d, self.q = order
        self.P, self.D, self.Q = seasonal_order
        self.m = seasonal_period
        self.ar_coefficients = None
        self.ma_coefficients = None
        self.seasonal_ar_coefficients = None
        self.seasonal_ma_coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.mse = None
        self.r_squared = None
    
    def difference(self, data, order):
        for _ in range(order):
            data = np.diff(data)
        return data
    
    def seasonal_difference(self, data, order):
        for _ in range(order):
            data = data[self.m:] - data[:-self.m]
        return data
    
    def undifference(self, data, original_data):
        for _ in range(self.d + self.D):
            data = np.cumsum(data) + original_data[0]
            original_data = original_data[1:]
        return data
    
    def _prepare_sarima_matrix(self):
        X, y = [], []
        differenced_data = self.seasonal_difference(self.difference(self.original_data, self.d), self.D)
        max_order = max(self.p + self.m * self.P, self.q + self.m * self.Q)
        for i in range(max_order, len(differenced_data)):
            ar_terms = differenced_data[i - self.p:i][::-1] if self.p > 0 else []
            seasonal_ar_terms = differenced_data[i - self.m * self.P:i:self.m][::-1] if self.P > 0 else []
            ma_terms = self.residuals[i - self.q:i][::-1] if self.q > 0 else []
            seasonal_ma_terms = self.residuals[i - self.m * self.Q:i:self.m][::-1] if self.Q > 0 else []
            X.append(np.concatenate([ar_terms, seasonal_ar_terms, ma_terms, seasonal_ma_terms]))
            y.append(differenced_data[i])
        return np.array(X), np.array(y)
    
    def fit(self, max_iterations=100, tolerance=1e-6):
        differenced_data = self.seasonal_difference(self.difference(self.original_data, self.d), self.D)
        max_order = max(self.p + self.m * self.P, self.q + self.m * self.Q)
        self.residuals = np.zeros_like(differenced_data)
        self.fitted_values = np.zeros_like(differenced_data)
        
        for _ in range(max_iterations):
            X, y = self._prepare_sarima_matrix()
            X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
            
            beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            
            new_intercept = beta[0]
            new_ar_coefficients = beta[1:self.p+1] if self.p > 0 else []
            new_seasonal_ar_coefficients = beta[self.p+1:self.p+self.P+1] if self.P > 0 else []
            new_ma_coefficients = beta[self.p+self.P+1:self.p+self.P+self.q+1] if self.q > 0 else []
            new_seasonal_ma_coefficients = beta[self.p+self.P+self.q+1:] if self.Q > 0 else []
            
            if self.intercept is not None:
                if np.all(np.abs(new_intercept - self.intercept) < tolerance) and \
                   np.all(np.abs(np.array(new_ar_coefficients) - np.array(self.ar_coefficients)) < tolerance) and \
                   np.all(np.abs(np.array(new_seasonal_ar_coefficients) - np.array(self.seasonal_ar_coefficients)) < tolerance) and \
                   np.all(np.abs(np.array(new_ma_coefficients) - np.array(self.ma_coefficients)) < tolerance) and \
                   np.all(np.abs(np.array(new_seasonal_ma_coefficients) - np.array(self.seasonal_ma_coefficients)) < tolerance):
                    break
            
            self.intercept = new_intercept
            self.ar_coefficients = new_ar_coefficients
            self.seasonal_ar_coefficients = new_seasonal_ar_coefficients
            self.ma_coefficients = new_ma_coefficients
            self.seasonal_ma_coefficients = new_seasonal_ma_coefficients
            
            self.fitted_values[max_order:] = X_with_intercept @ beta
            self.residuals[max_order:] = y - self.fitted_values[max_order:]
        
        self.fitted_values = self.undifference(self.fitted_values, self.original_data[:self.d + self.D])
        
        n = len(differenced_data) - max_order
        k = self.p + self.P + self.q + self.Q + self.d + self.D + 1
        
        self.mse = np.mean(self.residuals[max_order:]**2)
        tss = np.sum((y - np.mean(y))**2)
        self.r_squared = 1 - (np.sum(self.residuals[max_order:]**2) / tss)
        
        self.aic = n * np.log(self.mse) + 2 * k
        self.bic = n * np.log(self.mse) + k * np.log(n)
        
        sigma2 = np.sum(self.residuals[max_order:]**2) / (n - k)
        var_beta = sigma2 * np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
        self.std_errors = np.sqrt(np.diag(var_beta))
        
        self.t_stats = beta / self.std_errors
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), n - k))
    
    def predict(self, steps=1):
        if self.ar_coefficients is None or self.ma_coefficients is None:
            raise ValueError("Model has not been fitted yet.")
        
        differenced_data = self.seasonal_difference(self.difference(self.original_data, self.d), self.D)
        max_order = max(self.p + self.m * self.P, self.q + self.m * self.Q)
        history = list(differenced_data[-max_order:])
        residuals = list(self.residuals[-max_order:])
        predictions = []
        
        for _ in range(steps):
            ar_terms = np.dot(self.ar_coefficients, history[-self.p:][::-1]) if self.p > 0 else 0
            seasonal_ar_terms = np.dot(self.seasonal_ar_coefficients, history[-self.m*self.P::self.m][::-1]) if self.P > 0 else 0
            ma_terms = np.dot(self.ma_coefficients, residuals[-self.q:][::-1]) if self.q > 0 else 0
            seasonal_ma_terms = np.dot(self.seasonal_ma_coefficients, residuals[-self.m*self.Q::self.m][::-1]) if self.Q > 0 else 0
            next_val = self.intercept + ar_terms + seasonal_ar_terms + ma_terms + seasonal_ma_terms
            predictions.append(next_val)
            history.append(next_val)
            residuals.append(0)  # Assume future errors are zero
        
        predictions = np.array(predictions)
        return self.undifference(predictions, self.original_data[-(self.d + self.D):])

def plot_model_diagnostics(model):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    max_order = max(model.p + model.m * model.P, model.q + model.m * model.Q)
    valid_fitted = model.fitted_values[max_order:]
    valid_residuals = model.residuals[max_order:]
    
    # Residuals vs Fitted
    ax1.scatter(valid_fitted, valid_residuals)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(valid_residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q')
    
    # Scale-Location
    standardized_residuals = valid_residuals / np.sqrt(np.mean(valid_residuals**2))
    ax3.scatter(valid_fitted, np.sqrt(np.abs(standardized_residuals)))
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location')
    
    # Residuals vs Leverage
    X, _ = model._prepare_sarima_matrix()
    leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    ax4.scatter(leverage, standardized_residuals)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Standardized Residuals')
    ax4.set_title('Residuals vs Leverage')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate synthetic data
    t, synthetic_data = generate_synthetic_data(seasonal_period=12)
    
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Synthetic Calcium Traces')
    plt.title('Synthetic Calcium Traces with Seasonal Component')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Fit SARIMA models with different orders
    max_p = max_d = max_q = 2
    max_P = max_D = max_Q = 1
    seasonal_period = 12
    models = []
    aic_values = []
    bic_values = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            if p == d == q == P == D == Q == 0:
                                continue
                            model = SARIMAModel(synthetic_data, order=(p, d, q), 
                                                seasonal_order=(P, D, Q), 
                                                seasonal_period=seasonal_period)
                            model.fit()
                            models.append(model)
                            aic_values.append(model.aic)
                            bic_values.append(model.bic)
    
    # Find optimal model based on AIC
    optimal_index = np.argmin(aic_values)
    optimal_model = models[optimal_index]
    
    # Plot fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Original Data')
    max_order = max(optimal_model.p + optimal_model.m * optimal_model.P, 
                    optimal_model.q + optimal_model.m * optimal_model.Q)
    plt.plot(t[max_order:], optimal_model.fitted_values[max_order:], 
             label=f'Fitted SARIMA({optimal_model.p},{optimal_model.d},{optimal_model.q})({optimal_model.P},{optimal_model.D},{optimal_model.Q}){optimal_model.m}', 
             color='red')
    plt.title(f'SARIMA Model - ({optimal_model.p},{optimal_model.d},{optimal_model.q})({optimal_model.P},{optimal_model.D},{optimal_model.Q}){optimal_model.m}')
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
    plt.title(f'Future Predictions using SARIMA({optimal_model.p},{optimal_model.d},{optimal_model.q})({optimal_model.P},{optimal_model.D},{optimal_model.Q}){optimal_model.m} Model')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Print model summary
    print(f"Optimal SARIMA Model Summary ({optimal_model.p},{optimal_model.d},{optimal_model.q})({optimal_model.P},{optimal_model.D},{optimal_model.Q}){optimal_model.m}:")
    print(f"AIC: {optimal_model.aic:.4f}")
    print(f"BIC: {optimal_model.bic:.4f}")
    print(f"MSE: {optimal_model.mse:.4f}")
    print(f"R-squared: {optimal

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import generate_synthetic_data

class ARMAModel:
    def __init__(self, data, ar_order, ma_order):
        self.data = data
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.ar_coefficients = None
        self.ma_coefficients = None
        self.intercept = None
        self.fitted_values = None
        self.residuals = None
        self.aic = None
        self.bic = None
        self.mse = None
        self.r_squared = None
    
    def _prepare_arma_matrix(self):
        X, y = [], []
        max_order = max(self.ar_order, self.ma_order)
        for i in range(max_order, len(self.data)):
            ar_terms = self.data[i - self.ar_order:i][::-1] if self.ar_order > 0 else []
            ma_terms = self.residuals[i - self.ma_order:i][::-1] if self.ma_order > 0 else []
            X.append(np.concatenate([ar_terms, ma_terms]))
            y.append(self.data[i])
        return np.array(X), np.array(y)
    
    def fit(self, max_iterations=100, tolerance=1e-6):
        max_order = max(self.ar_order, self.ma_order)
        self.residuals = np.zeros_like(self.data)
        self.fitted_values = np.zeros_like(self.data)
        
        for _ in range(max_iterations):
            X, y = self._prepare_arma_matrix()
            X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
            
            beta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            
            new_intercept = beta[0]
            new_ar_coefficients = beta[1:self.ar_order+1] if self.ar_order > 0 else []
            new_ma_coefficients = beta[self.ar_order+1:] if self.ma_order > 0 else []
            
            if self.intercept is not None:
                if np.all(np.abs(new_intercept - self.intercept) < tolerance) and \
                   np.all(np.abs(np.array(new_ar_coefficients) - np.array(self.ar_coefficients)) < tolerance) and \
                   np.all(np.abs(np.array(new_ma_coefficients) - np.array(self.ma_coefficients)) < tolerance):
                    break
            
            self.intercept = new_intercept
            self.ar_coefficients = new_ar_coefficients
            self.ma_coefficients = new_ma_coefficients
            
            self.fitted_values[max_order:] = X_with_intercept @ beta
            self.residuals[max_order:] = y - self.fitted_values[max_order:]
        
        n = len(self.data) - max_order
        k = self.ar_order + self.ma_order + 1
        
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
        
        max_order = max(self.ar_order, self.ma_order)
        history = list(self.data[-max_order:])
        residuals = list(self.residuals[-max_order:])
        predictions = []
        
        for _ in range(steps):
            ar_terms = np.dot(self.ar_coefficients, history[-self.ar_order:][::-1]) if self.ar_order > 0 else 0
            ma_terms = np.dot(self.ma_coefficients, residuals[-self.ma_order:][::-1]) if self.ma_order > 0 else 0
            next_val = self.intercept + ar_terms + ma_terms
            predictions.append(next_val)
            history.append(next_val)
            residuals.append(0)  # Assume future errors are zero
        
        return np.array(predictions)

def plot_model_diagnostics(model):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    max_order = max(model.ar_order, model.ma_order)
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
    X, _ = model._prepare_arma_matrix()
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
    t, synthetic_data = generate_synthetic_data()
    
    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Synthetic Calcium Traces')
    plt.title('Synthetic Calcium Traces')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Fit ARMA models with different orders
    max_ar_order = 3
    max_ma_order = 3
    models = []
    aic_values = []
    bic_values = []
    
    for ar_order in range(max_ar_order + 1):
        for ma_order in range(max_ma_order + 1):
            if ar_order == 0 and ma_order == 0:
                continue
            model = ARMAModel(synthetic_data, ar_order=ar_order, ma_order=ma_order)
            model.fit()
            models.append(model)
            aic_values.append(model.aic)
            bic_values.append(model.bic)
    
    # Find optimal model based on AIC
    optimal_index = np.argmin(aic_values)
    optimal_model = models[optimal_index]
    optimal_ar_order = optimal_model.ar_order
    optimal_ma_order = optimal_model.ma_order
    
    # Plot fitted values
    plt.figure(figsize=(12, 6))
    plt.plot(t, synthetic_data, label='Original Data')
    max_order = max(optimal_ar_order, optimal_ma_order)
    plt.plot(t[max_order:], optimal_model.fitted_values[max_order:], 
             label=f'Fitted ARMA({optimal_ar_order},{optimal_ma_order})', color='red')
    plt.title(f'ARMA Model - AR Order {optimal_ar_order}, MA Order {optimal_ma_order}')
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
    plt.title(f'Future Predictions using ARMA({optimal_ar_order},{optimal_ma_order}) Model')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    # Print model summary
    print(f"Optimal ARMA Model Summary (AR Order = {optimal_ar_order}, MA Order = {optimal_ma_order}):")
    print(f"AIC: {optimal_model.aic:.4f}")
    print(f"BIC: {optimal_model.bic:.4f}")
    print(f"MSE: {optimal_model.mse:.4f}")
    print(f"R-squared: {optimal_model.r_squared:.4f}")
    print("\nCoefficients:")
    print(f"Intercept: {optimal_model.intercept:.4f}")
    for i, coef in enumerate(optimal_model.ar_coefficients):
        print(f"AR{i+1}: {coef:.4f} (p-value: {optimal_model.p_values[i+1]:.4f})")
    for i, coef in enumerate(optimal_model.ma_coefficients):
        print(f"MA{i+1}: {coef:.4f} (p-value: {optimal_model.p_values[i+1+len(optimal_model.ar_coefficients)]:.4f})")

if __name__ == "__main__":
    main()

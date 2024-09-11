import numpy as np

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

def generate_multivariate_synthetic_data(n_points=500, noise_level=0.1):
    """
    Function to generate multivariate synthetic time series data.
    """
    t = np.linspace(0, 10, n_points)
    signal1 = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t + np.pi/4)
    signal2 = np.cos(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t + np.pi/4)
    signal1 /= np.max(np.abs(signal1))
    signal2 /= np.max(np.abs(signal2))
    noise1 = np.random.normal(0, noise_level, n_points)
    noise2 = np.random.normal(0, noise_level, n_points)
    return np.column_stack([signal1 + noise1, signal2 + noise2])

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

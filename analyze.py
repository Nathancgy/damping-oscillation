import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
import math

data_folder = 'data'
csv_files = os.listdir(data_folder)
coefficients = []
areas = []

def damped_oscillator(t, A, b, w, phi, D):
    return A * np.exp(-b * t) * np.cos(w * t + phi) + D

def find_peaks_custom(y):
    peaks, properties = find_peaks(y, height=0) 
    return peaks, y[peaks]

def find_coefficient(file_path):
    df = pd.read_csv(file_path)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    peaks, peak_values = find_peaks_custom(y)

    if len(peaks) == 0:
        print(f"No peaks found in {file_path}")
        return None, None, None

    def exponential_decay(t, A, k):
        return A * np.exp(-k * t)

    initial_guess = [peak_values.iloc[0], 0.01]  
    popt, pcov = curve_fit(exponential_decay, x[peaks], peak_values, p0=initial_guess)
    A, k = popt

    return x, y, peaks, peak_values, A, k

for csv_file in csv_files:
    csv_file_path = os.path.join(data_folder, csv_file)
    x, y, peaks, peak_values, A, k = find_coefficient(csv_file_path)
    
    if x is None:
        continue
    
    mass = float(re.search(r'([\d.]+)g', csv_file).group(1))
    areas.append(float(re.search(r'([\d.]+)cm', csv_file).group(1))**2*math.pi)
    coefficient = mass * k
    print(f"Damping coefficient for {csv_file_path}: {coefficient}")
    coefficients.append(coefficient)

    A_guess = 0.2
    b_guess = 0.01
    w_guess = 2 * np.pi / (x[peaks[1]] - x[peaks[0]]) 
    phi_guess = 0
    D_guess = y.mean()
    initial_guess = [A_guess, b_guess, w_guess, phi_guess, D_guess]
    
    popt, pcov = curve_fit(damped_oscillator, x, y, p0=initial_guess)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data', s=10)
    plt.plot(x, damped_oscillator(x, *popt), color='red', label='Damped Oscillator Fit')
    plt.scatter(x[peaks], peak_values, color='green', label='Peaks', s=50, edgecolor='black')
    plt.plot(x, A * np.exp(-k * x), color='orange', label='Exponential Decay Fit')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Damped Oscillation (Damping Coefficient: {coefficient:.4f})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plt.scatter(areas, coefficients, color='blue', label='Coefficients vs Areas')
plt.xlabel('Areas (cm^2)')
plt.ylabel('Coefficients')
plt.title('Coefficients vs Areas')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
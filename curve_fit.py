import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
import seaborn
data_folder = 'data'
csv_files = os.listdir(data_folder)

def damped_oscillator(t, A, b, w, phi, D):
    return A * np.exp(-b * t) * np.cos(w * t + phi) + D

def find_peaks(y):
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            peaks.append(i)
    return peaks, y[peaks]

def find_coefficient(file_path):
    df = pd.read_csv(file_path)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    peaks, _ = find_peaks(y)

    def exponential_decay(t, k):
        return np.exp(-k * t)

    popt, pcov = curve_fit(exponential_decay, x[peaks], y[peaks])

    k = popt[0]
    return k

for csv_file in csv_files:
    csv_file_path = os.path.join(data_folder, csv_file)
    df = pd.read_csv(csv_file_path)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    
    A_guess = 0.2
    b_guess = 0.01
    w_guess = 2 * np.pi / (x[1] - x[0]) 
    phi_guess = 0
    initial_guess = [A_guess, b_guess, w_guess, phi_guess, 0.2]
    
    popt, pcov = curve_fit(damped_oscillator, x, y, p0=initial_guess)
    
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x, damped_oscillator(x, *popt), color='red', label='Fit')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Damped Oscillation')
    plt.legend()
    plt.grid(True)
    plt.show()

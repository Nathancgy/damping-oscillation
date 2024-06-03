# Damped Oscillation Analysis

This project analyzes damped oscillations using Python. It involves identifying the peaks in oscillatory data, fitting an exponential decay curve to the peaks, and fitting a damped oscillator model to the entire dataset. The project also calculates the damping coefficient based on the extracted data.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

The goal of this project is to analyze damped oscillatory motion by extracting key parameters such as the damping coefficient. The analysis involves:
1. Identifying peaks in the oscillatory data.
2. Fitting an exponential decay curve to the peaks.
3. Fitting a damped oscillator model to the entire dataset.
4. Calculating the damping coefficient.

## Features

- Peak detection using `scipy.signal.find_peaks`.
- Curve fitting using `scipy.optimize.curve_fit`.
- Visualizations of the data and fitted curves using `matplotlib`.

## Installation

1. Clone the repository:
   
   ```sh
   git clone https://github.com/yourusername/damped-oscillation-analysis.git
   cd damped-oscillation-analysis

2. Install the required packages:
   
   ```sh
   pip install -r requirements.txt

## Usage

1. Place your data files in the data directory. The data files should be in CSV format with the following structure:
   
   ```sh
   time,amplitude
   t1,a1
   t2,a2
   ...

2. Run the analyze script:

   ```sh
   Run the analysis script:

3. The script will process each CSV file in the data directory, identify the peaks, fit the curves, calculate the damping coefficient, and display the plots.

## Contributing

Contributions are welcome! Even though this repository only serves as my post-AP project for AP Physics C.

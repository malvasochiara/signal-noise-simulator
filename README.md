# signal-noise-simulator

This repository contains a Python-based implementation for generating periodic signals.

## Installation

To use this repository, first clone it with:

```bash
git clone https://github.com/malvasochiara/signal-noise-simulator.git
cd signal-noise-simulator
```

## Dependencies

This project requires **Python ≥ 3.8** and the following Python packages:

- `numpy==1.23.5`
- `matplotlib==3.10.0`
- `pytest==8.3.4`
- `pytest-cov==6.0.0`

You can install them using:

```bash
python -m pip install -r requirements.txt
```

## Scripts overview

### signal_toolkit.py

The `signal_toolkit.py` script contains a set of functions for generating signals and adding noise. Below is an overview of the available functions:

#### `generate_random_frequencies`
Generates an array of random frequencies within a valid range to avoid aliasing. The frequencies are integers randomly selected between 1 Hz and the Nyquist frequency (half the sampling rate).

**Parameters:**
- `num_components` (int): Number of random frequencies to generate.
- `sampling_rate` (int, optional): Sampling rate in Hz. Must be at least 4 Hz (default is 250 Hz).

**Returns:**
- `frequencies` (`numpy.ndarray`): Array of random integer frequencies in Hz.

---

#### `generate_periodic_signal`
Generates a periodic signal composed of the sum of sinusoidal or square waves. The function allows users to specify the waveform type.

**Parameters:**
- `frequencies` (`numpy.ndarray`): Array of frequencies (in Hz) for the waves.
- `duration` (float, optional): Duration of the signal in seconds (default is 1 second).
- `sampling_rate` (int, optional): Sampling rate in Hz (default is 250 Hz).
- `waveform_type` (str, optional): Type of waveform to generate. Can be either `'sin'` (default) for sinusoidal waves or `'square'` for square waves.

**Returns:**
- `time` (`numpy.ndarray`): Array of time points for the signal (in seconds).
- `signal` (`numpy.ndarray`): Array representing the sum of the generated waves.

**Raises:**
- `TypeError`: If the input parameters are not of the correct type.
- `ValueError`: If any frequency is not positive or exceeds the Nyquist frequency, or if `waveform_type` is not `'sin'` or `'square'`.

---

#### `compute_signal_power`
Computes the root mean square (RMS) power of a given signal.

**Parameters:**
- `signal` (`numpy.ndarray`): Input signal for which the RMS power will be calculated.

**Returns:**
- `power` (float): The root mean square (RMS) power of the signal.

---

#### `compute_white_noise_std`
Computes the standard deviation of white Gaussian noise required to achieve a specified signal-to-noise ratio (SNR).

**Parameters:**
- `signal` (`numpy.ndarray`): Input signal for which the noise standard deviation will be computed.
- `snr_db` (float): Desired signal-to-noise ratio (SNR) in decibels (dB).

**Returns:**
- `noise_std` (float): The standard deviation of the white Gaussian noise needed to achieve the given SNR.

---

#### `generate_white_noise`
Generates white Gaussian noise with a specified SNR relative to a given signal.

**Parameters:**
- `signal` (`numpy.ndarray` or `int`): Input signal to determine the noise level. If a scalar (int or float) is provided, it is treated as an array of length 1. If an array is provided, its length determines the length of the generated noise.
- `snr_db` (float): Desired signal-to-noise ratio (SNR) in decibels (dB).

**Returns:**
- `noise` (`numpy.ndarray`): White Gaussian noise with the computed standard deviation needed to achieve the given SNR relative to the input signal. The output is an array of the same length as the input signal, whether scalar or array.

---

#### `add_white_noise`
Adds white Gaussian noise to a given signal with a specified SNR.

**Parameters:**
- `signal` (`numpy.ndarray` or `int`): Input signal to which noise will be added. If a scalar (int or float) is provided, it is treated as an array of length 1.
- `snr_db` (float): Desired signal-to-noise ratio (SNR) in decibels (dB).

**Returns:**
- `noisy_signal` (`numpy.ndarray`): The input signal with added white Gaussian noise. The output is an array of the same length as the input signal, whether scalar or array.

The generated signals can be used for further analysis, testing, or as inputs to other processing functions.

---

#### `compute_fft`

Computes the full Fast Fourier Transform (FFT) of a real-valued signal.


**Parameters:**

- `signal` (`numpy.ndarray`): Input signal, assumed to be a 1D array of real values.

- `sampling_rate` (`int` or `float`, optional): Sampling rate of the signal in Hz (default is 250 Hz).

**Returns:**

- `fft_coefficients` (`numpy.ndarray`): The full FFT of the input signal. The output is complex-valued, representing both magnitude and phase.

- `frequency_bins` (`numpy.ndarray`): Array of frequency values corresponding to the FFT output, ranging from negative to positive frequencies (centered at zero).
---
#### `compute_ifft`
Computes the inverse Fast Fourier Transform (IFFT) of a given spectrum and returns the resulting time-domain signal.

**Parameters:**

- `spectrum (numpy.ndarray)`: The input spectrum, assumed to be complex-valued, representing the frequency-domain coefficients of the signal.

**Returns:**

- `signal (numpy.ndarray)`: The complex time-domain signal obtained by applying the IFFT to the input spectrum.
---
#### `apply_spectral_slope`
Applies a linear spectral slope to a signal in the frequency domain.

**Parameters:**

- `signal (numpy.ndarray)`: Input time-domain signal, assumed to be a 1D array of real values.
- `slope (float)`: Slope value that defines the linear modification applied to the spectrum. Must be non-negative.
- `sampling_rate (int, optional)`: Sampling rate of the signal in Hz (default is 250 Hz).

**Returns:**

- `modified_spectrum (numpy.ndarray)`: The modified frequency-domain representation of the signal after applying the linear spectral slope. The output is complex-valued.
---

#### `add_colored_noise`

Adds colored noise to a signal based on a given spectral slope.

This function generates white noise at a specified signal-to-noise ratio (SNR), applies a spectral slope to shape its frequency content, and then adds the resulting colored noise to the input signal.

**Note**: The output signal is complex-valued due to the spectral transformation. Discarding the imaginary part can alter the spectral characteristics of the noise.

**Parameters:**

- `signal (numpy.ndarray)`: Input time-domain signal, assumed to be a 1D array of real values.

- `snr_db (float)`: Desired signal-to-noise ratio (SNR) in decibels, defining the power level of the noise relative to the signal.

- `slope (float)`: Slope value that defines the spectral modification applied to the noise.

- `sampling_rate (int)`: Sampling rate of the signal in Hz.

**Returns:**

- `noisy_signal` (numpy.ndarray): The input signal with added colored noise. The output is a 1D array of real values.

---

### utils.py

The `utils.py` file contains utility functions for saving generated data and handling file paths.

**Main Functions**
- `parse_arguments()`: Parses and processes command-line arguments for signal generation and plotting. It handles parameters such as the number of components, duration, sampling rate, waveform type, noise addition, and saving options.

- `plot_clean_signal(time, signal)`: Plots a clean, noise-free periodic signal over time.

- `plot_noisy_signal(time, signal, noisy_signal, snr)`: Plots both the clean and noisy versions of a signal, allowing for comparison when noise is added based on the given signal-to-noise ratio (SNR).

- `save_signal_to_csv(time, signal, noisy_signal=None, save_path='.', waveform_type='sin', sampling_rate=250, snr=None)`: Saves the generated signal (and optionally the noisy version) to a CSV file, automatically creating the directory if needed. The filename encodes signal properties such as waveform type, sampling rate, and SNR level.

- `generate_and_plot_signal()`: Manages the full process of signal generation, noise addition, plotting, and saving. It retrieves arguments, generates the signal based on user preferences, and performs the required operations accordingly.

These functions are used in signal_builder.py to manage data saving in a flexible and automated way.

---

## signal_builder.py

The `signal_builder.py`  generates periodic signals by combining sinusoidal or square waves with either random frequencies or user-defined frequencies. The signal can be configured through command-line arguments, and the script will return the generated signal and corresponding time values. An option to plot the generated signal is also available. Additionally, noise can be added to the signal, and a signal-to-noise ratio (SNR) can be specified for the added noise. The generated signal and time data can be saved to a CSV file. The user can specify a directory where the file should be saved, or it will default to the current directory. If a SNR is provided, both clean and noisy signals will be saved.

### Usage

To use the script from the command line, you can run the following command:

```bash
python signal_builder.py --duration 1.0 --sampling_rate 200 --num_components 10 --snr 10 --plot
```

Alternatively, you can provide a custom list of frequencies:

```bash
python signal_builder.py --duration 1.0 --sampling_rate 200 --frequencies 10,20,30 --snr 10 --plot
```
In this case, the `--num_components` argument will be ignored if `--frequencies` is provided.

To save the generated signal, use the `--save` argument:
```bash
python script.py --num_components 5 --duration 1.0 --sampling_rate 250 --save
```
By default, this saves the file in the current directory (./signal_data.csv). To specify a different directory:
```bash
python script.py --num_components 5 --duration 1.0 --sampling_rate 250 --save /path/to/directory
```
**Arguments:**
- `--duration`: Duration of the signal in seconds (e.g., 1.0 for 1 second).
- `--sampling_rate`: Sampling rate of the signal in Hz (e.g., 200 for 200 Hz).
- `--num_components`: Number of sinusoidal or square wave components to combine (e.g., 10).
- `--frequencies`: Comma-separated list of frequencies in Hz (e.g., '10,20,30'). If provided, the `--num_components` argument will be ignored.
- `--snr`: Signal-to-noise ratio (SNR) in dB (e.g., 10 for a 10 dB SNR). If omitted, the signal will be generated without noise.
- `--plot`: Option to plot the generated signal (add this flag to see the plot).
-  `--save`: Option to save signal and time data as a CSV file. If no path is provided, the file will be saved in the current directory.

### Accessing Help
To get a description of all available parameters and options, you can access the help documentation by running:

```bash
python signal_builder.py --help
```

## Testing

To ensure the correct functionality of the functions, various tests have been implemented. You can run the tests using **pytest** to verify that the function behaves as expected.

## Work in Progress

This repository is under active development. New features and improvements will be added incrementally.
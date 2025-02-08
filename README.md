# signal-noise-simulator

This repository contains a Python-based implementation for generating sinusoidal signals.

## Current Functionality

### `generate_random_frequencies`
Generates an array of random frequencies within a valid range to avoid aliasing. The frequencies are integers randomly selected between 1 Hz and the Nyquist frequency (half the sampling rate).

**Parameters:**
- `num_components` (int): Number of random frequencies to generate.
- `sampling_rate` (int, optional): Sampling rate in Hz. Must be at least 4 Hz (default is 250 Hz).

**Returns:**
- `frequencies` (`numpy.ndarray`): Array of random integer frequencies in Hz.

---

### `generate_periodic_signal`
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

### `compute_signal_power`
Computes the root mean square (RMS) power of a given signal.

**Parameters:**
- `signal` (`numpy.ndarray`): Input signal for which the RMS power will be calculated.

**Returns:**
- `power` (float): The root mean square (RMS) power of the signal.

---

### `compute_white_noise_std`
Computes the standard deviation of white Gaussian noise required to achieve a specified signal-to-noise ratio (SNR).

**Parameters:**
- `signal` (`numpy.ndarray`): Input signal for which the noise standard deviation will be computed.
- `snr_db` (float): Desired signal-to-noise ratio (SNR) in decibels (dB).

**Returns:**
- `noise_std` (float): The standard deviation of the white Gaussian noise needed to achieve the given SNR.

---

### `generate_white_noise`
Generates white Gaussian noise with a specified SNR relative to a given signal.

**Parameters:**
- `signal` (`numpy.ndarray` or `int`): Input signal to determine the noise level. If a scalar (int or float) is provided, it is treated as an array of length 1. If an array is provided, its length determines the length of the generated noise.
- `snr_db` (float): Desired signal-to-noise ratio (SNR) in decibels (dB).

**Returns:**
- `noise` (`numpy.ndarray`): White Gaussian noise with the computed standard deviation needed to achieve the given SNR relative to the input signal. The output is an array of the same length as the input signal, whether scalar or array.

---

The generated signals can be used for further analysis, testing, or as inputs to other processing functions.

## Testing

To ensure the correct functionality of the functions, various tests have been implemented. You can run the tests using **pytest** to verify that the function behaves as expected.

## Work in Progress

This repository is under active development. New features and improvements will be added incrementally.

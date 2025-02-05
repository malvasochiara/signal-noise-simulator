# signal-noise-simulator

This repository contains a Python-based implementation for generating sinusoidal signals. 


## Current Functionality

- **`random_frequencies_generator`**: Generates an array of random frequencies within a valid range to avoid aliasing. The frequencies are integers randomly selected between 1 Hz and the Nyquist frequency (half the sampling rate).
  
  **Parameters:**
  - `num_components` (int): Number of random frequencies to generate.
  - `sampling_rate` (int, optional): Sampling rate in Hz Must be at least 4 Hz (default is 250 Hz).
  
  **Returns:**
  - `frequencies` (numpy.ndarray): Array of random integer frequencies in Hz.

- **`signal_generator`**: Generates a signal by summing sinusoidal waves at specified frequencies. The frequencies passed as input can be either an array chosen by the user or generated randomly using the function **`random_frequencies_generator`**.  
  **Parameters:**
  - `frequencies` (numpy.ndarray): Array of frequencies (in Hz) for the sine waves.
  - `duration` (float, optional): Duration of the signal in seconds (default is 1 second).
  - `sampling_rate` (int, optional): Sampling rate in Hz (default is 250 Hz).
  
  **Returns:**
  - `time` (numpy.ndarray): Array of time points for the signal (in seconds).
  - `signal` (numpy.ndarray): Array representing the sum of the sinusoidal waves.

The generated signals can be used for further analysis, testing, or as inputs to other processing functions.

## Testing

To ensure the correct functionality of the **`signal_generator`** function, various tests have been implemented. You can run the tests using **pytest** to verify that the function behaves as expected.


## Work in Progress
This repository is under active development. New features and improvements will be added incrementally.
 

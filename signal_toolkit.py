# -*- coding: utf-8 -*-
import numpy as np
import random
import math

def signal_generator(max_components, duration=1, sampling_rate=250):
    """
    Generate a signal composed of the sum of sinusoidal waves.

    Parameters
    ----------
    max_components : int
        Maximum number of sine waves to include in the signal.
    duration : float, optional
        Duration of the signal in seconds. Default is 1 second.
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 250 Hz.

    Returns
    -------
    frequencies : numpy.ndarray
        Array of frequencies of the sine waves (in Hz).
    time : numpy.ndarray
        Array of time points (in seconds).
    signal : numpy.ndarray
        The resulting sinusoidal signal.
        
    Raises
    ------
    """
    
    num_components = random.randint(1, max_components)

    # Generate a random 
    frequencies = np.random.randint(1, math.floor(sampling_rate / 2) + 1, size=num_components)

    # Create a time array based on the duration and sampling rate.
    # This provides the x-axis values for the sine waves, ensuring proper sampling of the signal.
    time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    # Initialize the signal array as zeros to serve as a base for summing sine waves.
    signal = np.zeros_like(time)

    sine_waves = [np.sin(2 * np.pi * freq * time) for freq in frequencies]

    # Use numpy.sum to combine all sine waves efficiently in one operation
    signal = np.sum(sine_waves, axis=0)

    # Return the components and the generated signal for further use or analysis.
    return frequencies, time, signal


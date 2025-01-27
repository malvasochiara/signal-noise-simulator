# -*- coding: utf-8 -*-
import numpy as np
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
    # Randomly select the number of sine wave components to ensure variability
    # in the generated signal
    num_components = np.random.randint(1, max_components)

    # Generate random frequencies for the sine waves. These are limited by
    # the Nyquist frequency (half the sampling rate) to avoid aliasing.
    frequencies = np.random.randint(
        1, math.floor(sampling_rate / 2), size=num_components
    )

    # Create the time array based on the signal duration and sampling rate.
    # This ensures proper temporal resolution for the sine waves.
    time = np.linspace(
        0, duration, int(duration * sampling_rate), endpoint=False
    )

    # Sum the sine waves directly to form the final signal.
    signal = sum(np.sin(2 * np.pi * freq * time) for freq in frequencies)

    return frequencies, time, signal

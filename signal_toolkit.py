# -*- coding: utf-8 -*-
import numpy as np
import math


def random_frequencies_generator(num_components, sampling_rate=250):
    """
    Generate random frequencies within the proper range to avoid aliasing

    Parameters
    ----------
    num_components : int
        Number of frequencies to generate.
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 250 Hz.

    Returns
    -------
    frequencies : numpy.ndarray
        Array of randomly chosen integer frequencies (in Hz).
    """
    # Set the maximum value of frequencies to the Nyquist's frequency to avoid aliasing
    frequencies = np.random.randint(
        1, math.floor(sampling_rate / 2), size=num_components
    )

    return frequencies


def signal_generator(frequencies, duration=1, sampling_rate=250):
    """
    Generate a signal composed of the sum of sinusoidal waves.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequencies of the sine waves (in Hz).
    duration : float, optional
        Duration of the signal in seconds. Default is 1 second.
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 250 Hz.

    Returns
    -------

    time : numpy.ndarray
        Array of time points (in seconds).
    signal : numpy.ndarray
        The resulting sinusoidal signal.

    Raises
    ------
    """
    if not all(isinstance(f, (int, np.integer)) for f in frequencies):
        raise TypeError("Frequencies should be integer")

    if np.any(frequencies <= 0):
        raise ValueError("Frequencies should be positive and non-zero")

    # Create the time array based on the signal duration and sampling rate
    # to ensure proper temporal resolution for the sine waves.
    time = np.linspace(
        0, duration, int(duration * sampling_rate), endpoint=False
    )

    # Sum the sine waves directly to form the final signal.
    signal = sum(np.sin(2 * np.pi * freq * time) for freq in frequencies)

    return time, signal

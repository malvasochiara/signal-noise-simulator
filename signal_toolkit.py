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
    if not isinstance(num_components, (int, np.integer)) or isinstance(
        num_components, bool
    ):
        raise TypeError("num_components should be an integer number")

    if not isinstance(sampling_rate, (int, np.integer)) or isinstance(
        sampling_rate, bool
    ):
        raise TypeError("sampling_rate should be an integer number")

    if sampling_rate <= 3:
        raise ValueError("sampling_rate should be greater than or equal to 4")

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

    if not isinstance(
        duration, (int, float, np.integer, np.floating)
    ) or isinstance(duration, bool):
        raise TypeError("Duration should be a number, either integer or float")

    if not isinstance(
        sampling_rate, (int, float, np.integer, np.floating)
    ) or isinstance(sampling_rate, bool):
        raise TypeError(
            "Sampling rate should be a number, either integer or float"
        )

    if np.any(frequencies <= 0):
        raise ValueError("Frequencies should be positive and non-zero")

    if duration < 0:
        raise ValueError("Duration should be greater than or equal to 0")

    if sampling_rate < 0:
        raise ValueError("Sampling rate should be greater than or equal to 0")

    if np.any(frequencies >= math.floor(sampling_rate / 2)):
        raise ValueError(
            "Frequencies should be smaller than Nyquist's frequency, sampling_rate/2"
        )

    # Create the time array based on the signal duration and sampling rate
    # to ensure proper temporal resolution for the sine waves.
    time = np.linspace(
        0, duration, int(duration * sampling_rate), endpoint=False
    )

    # Ensure the function always returns a signal array of the same length as time, even when the frequencies array is empty,
    # to maintain consistency in the output format.
    if frequencies.size == 0:
        signal = np.zeros_like(time)
    else:
        signal = np.sum(
            np.array(
                [np.sin(2 * np.pi * freq * time) for freq in frequencies]
            ),
            axis=0,
        )

    return time, signal

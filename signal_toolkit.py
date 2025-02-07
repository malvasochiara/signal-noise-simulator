# -*- coding: utf-8 -*-
import numpy as np
import math


def generate_random_frequencies(num_components, sampling_rate=250):
    """Generate random frequencies within the proper range to avoid aliasing.

    Parameters
    ----------
    num_components : int
        Number of frequencies to generate.
    sampling_rate : int, optional
        Sampling rate in Hz. Must be at least 4 Hz. Default is 250 Hz.

    Returns
    -------
    frequencies : numpy.ndarray
        Array of randomly chosen integer frequencies (in Hz).

    Raises
    ------
    TypeError
        If `num_components` or `sampling_rate` is not an integer.
    ValueError
        If `sampling_rate` is less than 4.
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


def generate_sinusoidal_signal(frequencies, duration=1, sampling_rate=250):
    """Generate a signal composed of the sum of sinusoidal waves.

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
    TypeError
        If `frequencies` is not an array of integers.
        If `duration` or `sampling_rate` is not a number.
    ValueError
        If any frequency is not positive or exceeds the Nyquist frequency.
        If `duration` or `sampling_rate` is negative.
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


def compute_signal_power(signal):
    """Compute the root mean square (RMS) power of the signal.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal for which the RMS power will be calculated.

    Returns
    -------
    power : float
        The root mean square (RMS) power of the signal.

    Raises
    ------
    ValueError
        If `signal` is an empty array.
    """
    if signal.size == 0:
        raise ValueError("Input signal must be a non-empty array.")
    return np.sqrt(np.mean(signal**2))


def compute_white_noise_std(signal, snr_db):
    """Compute the standard deviation of white Gaussian noise required to
    achieve a specified SNR.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal for which the noise standard deviation will be computed.
    snr_db : float
        Desired signal-to-noise ratio (SNR) in decibels (dB).

    Returns
    -------
    noise_std : float
        The standard deviation of the white Gaussian noise needed to achieve the given SNR.

    Raises
    ------
    TypeError
        If `snr_db` is not a number.
    ValueError
        If `signal` is empty.
    """
    if not isinstance(
        snr_db, (int, float, np.integer, np.floating)
    ) or isinstance(snr_db, bool):
        raise TypeError("SNR should be a number, either integer or float")
    signal_rms = compute_signal_power(signal)
    return signal_rms / (10 ** (snr_db / 10))

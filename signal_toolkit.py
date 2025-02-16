# -*- coding: utf-8 -*-
'''
Author: Chiara Malvaso
Date: February 2025
'''

import numpy as np
import math


def generate_random_frequencies(num_components, sampling_rate=250):
    """Generate random frequencies within the proper range to avoid aliasing.

    This function generates an array of random integer frequencies that are
    within the appropriate range, taking into account the Nyquist frequency
    (half the sampling rate) to avoid aliasing.

    Parameters
    ----------
    num_components : int
        Number of signal components (i.e., number of random frequencies to generate).
    sampling_rate : int, optional
        The sampling rate in Hz. Must be at least 4 Hz. Default is 250 Hz.

    Returns
    -------
    numpy.ndarray
        Array of randomly chosen frequencies (in Hz).

    Raises
    ------
    TypeError
        If `num_components` or `sampling_rate` is not an integer.
    ValueError
        If `sampling_rate` is less than 4 or if `num_components` is negative.
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



def generate_periodic_signal(
    frequencies, duration=1, sampling_rate=250, waveform_type="sin"
):
    """Generate a periodic signal composed of the sum of sinusoidal or square waves.

    This function generates a signal by summing sinusoidal or square waves
    with the given frequencies. It returns the time values and the resulting
    periodic signal.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequencies of the waves (in Hz).
    duration : float, optional
        Duration of the signal in seconds. Default is 1.0 second.
    sampling_rate : int, optional
        Sampling frequency in Hz. Default is 250 Hz.
    waveform_type : {'sin', 'square'}, optional
        Type of waveform ('sin' for sinusoidal wave, 'square' for square wave;
        default is 'sin').

    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: The time points (array of time values).
        - numpy.ndarray: The resulting periodic signal (sum of sinusoids or square waves).

    Raises
    ------
    TypeError
        If `frequencies` is not a valid array of integers or if `duration`,
        `sampling_rate`, or `waveform_type` is of the wrong type.
    ValueError
        If `frequencies` are not valid or if `waveform_type` is not 'sin' or 'square'.
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

    if not isinstance(waveform_type, str):
        raise TypeError("Waveform type should be a string, sin or square.")

    if waveform_type not in {"sin", "square"}:
        raise ValueError("Waveform type should be sin or square.")

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
        # Generate the signal based on the chosen waveform type
        if waveform_type == "sin":
            signal = np.sum(
                [np.sin(2 * np.pi * freq * time) for freq in frequencies],
                axis=0,
            )
        else:  # waveform_type == "square"
            signal = np.sum(
                [
                    np.sign(np.sin(2 * np.pi * freq * time))
                    for freq in frequencies
                ],
                axis=0,
            )

    return time, signal


def compute_signal_power(signal):
    """Compute the root mean square (RMS) power of the signal.

    This function calculates the root mean square (RMS) of the given signal,
    which represents the power of the signal in a statistical sense.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal for which the RMS power will be calculated.

    Returns
    -------
    float
        The RMS power of the signal.

    Raises
    ------
    ValueError
        If `signal` is an empty array.
    TypeError
        If `signal` is not a numeric array.
    """
    if signal.size == 0:
        raise ValueError("Input signal must be a non-empty array.")
    return np.sqrt(np.mean(signal**2))


def compute_white_noise_std(signal, snr_db):
    """Compute the standard deviation of white Gaussian noise required to achieve a
    specified SNR.

    This function calculates the standard deviation of white Gaussian noise
    needed to achieve a given signal-to-noise ratio (SNR) based on the RMS
    power of the input signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal for which the noise standard deviation will be computed.
    snr_db : float
        The desired signal-to-noise ratio (SNR) in decibels (dB).

    Returns
    -------
    float
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


def generate_white_noise(signal, snr_db):
    """Generate white Gaussian noise with a specified SNR relative to a given signal.

    This function generates white Gaussian noise with the correct standard deviation
    to achieve the specified SNR, based on the input signal.

    Parameters
    ----------
    signal : numpy.ndarray or int
        The input signal to determine the noise level. If a scalar (int or float) is
        provided, it is treated as an array of length 1. If an array is provided,
        its length determines the length of the generated noise.
    snr_db : float
        The desired signal-to-noise ratio (SNR) in decibels (dB).

    Returns
    -------
    numpy.ndarray
        White Gaussian noise with the computed standard deviation needed to achieve the given SNR.
    """
    # Check if signal is a scalar and convert to array if so
    if np.isscalar(signal):
        signal = np.array([signal])

    noise_std = compute_white_noise_std(signal, snr_db)
    noise = np.random.normal(0, noise_std, len(signal))
    return noise


def add_white_noise(signal, snr_db):
    """Add white Gaussian noise to a given signal with a specified SNR.

    This function adds white Gaussian noise to a signal in order to achieve the specified SNR.

    Parameters
    ----------
    signal : numpy.ndarray or int
        The input signal to which noise will be added. If a scalar (int or float)
        is provided, it is treated as an array of length 1.
    snr_db : float
        The desired signal-to-noise ratio (SNR) in decibels (dB).

    Returns
    -------
    numpy.ndarray
        The input signal with added white Gaussian noise.
    """

    noisy_signal = signal + generate_white_noise(signal, snr_db)
    return noisy_signal


def compute_fft(signal, sampling_rate=250):
    """Compute the full Fast Fourier Transform (FFT) of a real-valued signal.

    This function computes the FFT of the given signal, returning the complex
    coefficients and the corresponding frequency bins.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal, assumed to be a 1D array of real values.
    sampling_rate : int or float, optional
        The sampling rate of the signal in Hz (default is 250 Hz).

    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: The FFT coefficients of the input signal.
        - numpy.ndarray: The corresponding frequency bins.

    Raises
    ------
    TypeError
        If `signal` is not an array of numbers.
    ValueError
        If `sampling_rate` is less than or equal to 0.
    """
    if not all(
        isinstance(s, (int, float, np.integer, np.floating)) for s in signal
    ):
        raise TypeError(
            "Signal should be an array of numbers, either integer or float"
        )

    if not isinstance(
        sampling_rate, (int, float, np.integer, np.floating)
    ) or isinstance(sampling_rate, bool):
        raise TypeError(
            "Sampling rate should be a number, either integer or float"
        )

    if sampling_rate <= 0:
        raise ValueError("Sampling rate should be greater than or equal to 0")

    fft_coefficients = np.fft.fft(signal)
    frequency_bins = np.fft.fftfreq(len(signal), d=1 / sampling_rate)
    return fft_coefficients, frequency_bins


def compute_ifft(spectrum):
    """Compute the inverse Fast Fourier Transform (IFFT).

    This function computes the IFFT of the given complex-valued spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The input spectrum, assumed to be a complex-valued array.

    Returns
    -------
    numpy.ndarray
        The time-domain signal obtained by applying the IFFT to the input spectrum.

    Raises
    ------
    TypeError
        If `spectrum` is not a NumPy array.
    ValueError
        If `spectrum` does not contain complex values.
    """
    if not isinstance(spectrum, np.ndarray):
        raise TypeError("Input spectrum must be a NumPy array.")

    if not np.issubdtype(spectrum.dtype, np.complexfloating):
        raise ValueError(
            "Input spectrum must be a complex-valued NumPy array."
        )

    return np.fft.ifft(spectrum)


def apply_spectral_slope(signal, slope, sampling_rate=250):
    """Apply a linear spectral slope to a signal in the frequency domain.

    This function applies a linear modification to the signal's frequency spectrum
    by adjusting its magnitude with a specified slope.

    Parameters
    ----------
    signal : numpy.ndarray
        The input time-domain signal, assumed to be a 1D array of real values.
    slope : float
        The slope value that defines the linear modification applied to the spectrum.
    sampling_rate : int or float, optional
        The sampling rate of the signal in Hz (default is 250 Hz).

    Returns
    -------
    numpy.ndarray
        The time-domain signal after the spectral slope is applied.

    Raises
    ------
    TypeError
        If `signal` is not a valid numeric array, or `slope` is not a number.
    ValueError
        If `signal` is empty.
    """
    if not isinstance(
        slope, (int, float, np.integer, np.floating)
    ) or isinstance(slope, bool):
        raise TypeError("Slope should be a number, either integer or float")

    if slope < 0:
        raise ValueError("Slope should be greater than or equal to 0")

    fft_coefficients, frequency_bins = compute_fft(signal, sampling_rate)
    return fft_coefficients + slope * frequency_bins


def add_colored_noise(signal, snr_db, slope, sampling_rate):
    """Add colored noise to a signal based on a given spectral slope.

    This function generates white noise at a specified signal-to-noise ratio (SNR),
    applies a spectral slope to shape its frequency content, and then adds the
    resulting colored noise to the input signal. The output signal is real-valued,
    but the process involves a complex-valued intermediate representation.

    **Note:** The output signal is complex-valued due to the spectral transformation.
    Discarding the imaginary part can alter the spectral characteristics of the noise.

    Parameters
    ----------
    signal : numpy.ndarray
        Input time-domain signal, assumed to be a 1D array of real values.
    snr_db : float
        Desired signal-to-noise ratio (SNR) in decibels, defining the power
        level of the noise relative to the signal.
    slope : float
        Slope value that defines the spectral modification applied to the noise.
    sampling_rate : int, optional
        Sampling rate of the signal in Hz (default is 250 Hz).

    Returns
    -------
    numpy.ndarray
        The input signal with added colored noise. The output is a 1D array
        of real values.

    Raises
    ------
    TypeError
        If `snr_db` is not a number or `signal` is not a valid array.
    ValueError
        If `signal` is empty or `slope` is not a valid number.
    """
    white_noise = generate_white_noise(signal, snr_db)
    colored_noise_spectrum = apply_spectral_slope(white_noise, slope)
    colored_noise = compute_ifft(colored_noise_spectrum)
    return signal + colored_noise

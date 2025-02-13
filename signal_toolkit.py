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
        If `sampling_rate` is less than 4 or if `num_components` is negative
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
    """Generate a periodic signal composed of the sum of sinusoidal or square
    waves.

    Parameters
    ----------
    frequencies : numpy.ndarray
        Array of frequencies of the waves (in Hz).
    duration : float, optional
        Duration of the signal in seconds. Default is 1 second.
    sampling_rate : int, optional
        Sampling rate in Hz. Default is 250 Hz.
    waveform_type : str, optional
        Type of waveform to generate. Can be either 'sin' (default) for
        sinusoidal waves or 'square' for square waves.

    Returns
    -------
    time : numpy.ndarray
        Array of time points (in seconds).
    signal : numpy.ndarray
        The resulting periodic signal.

    Raises
    ------
    TypeError
        If `frequencies` is not an array of integers.
        If `duration` or `sampling_rate` is not a number.
        If `waveform_type` is not a string.
    ValueError
        If any frequency is not positive or exceeds the Nyquist frequency.
        If `duration` or `sampling_rate` is negative.
        If `waveform_type` is not 'sin' or 'square'.
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
    TypeError
        If 'signal' is not a numeric array.
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


def generate_white_noise(signal, snr_db):
    """Generate white Gaussian noise with a specified SNR relative to a given
    signal.

    Parameters
    ----------
    signal : numpy.ndarray or int
        Input signal to determine the noise level. If a scalar (int or float)
        is provided, it is treated as an array of length 1. If an array is
        provided, its length determines the length of the generated noise.
    snr_db : float
        Desired signal-to-noise ratio (SNR) in decibels (dB).

    Returns
    -------
    noise : numpy.ndarray
        White Gaussian noise with the computed standard deviation needed to
        achieve the given SNR relative to the input signal. The output is an array
        of the same length as the input signal, whether scalar or array.
    """
    # Check if signal is a scalar and convert to array if so
    if np.isscalar(signal):
        signal = np.array([signal])

    noise_std = compute_white_noise_std(signal, snr_db)
    noise = np.random.normal(0, noise_std, len(signal))
    return noise


def add_white_noise(signal, snr_db):
    """Add white Gaussian noise to a given signal with a specified SNR.

    Parameters
    ----------
    signal : numpy.ndarray or int
        Input signal to which noise will be added. If a scalar (int or float)
        is provided, it is treated as an array of length 1.
    snr_db : float
        Desired signal-to-noise ratio (SNR) in decibels (dB).

    Returns
    -------
    noisy_signal : numpy.ndarray
        The input signal with added white Gaussian noise. The output is an array
        of the same length as the input signal, whether scalar or array.
    """

    noisy_signal = signal + generate_white_noise(signal, snr_db)
    return noisy_signal


def compute_fft(signal, sampling_rate=250):
    """Compute the full Fast Fourier Transform (FFT) of a real-valued signal.

    This function calculates the two-sided Fourier Transform of a real-valued
    input signal, returning both positive and negative frequency components
    along with their corresponding complex Fourier coefficients.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal, assumed to be a 1D array of real values.
    sampling_rate : int or float, optional
        Sampling rate of the signal in Hz (default is 250 Hz).

    Returns
    -------
    fft_coefficients : numpy.ndarray
        The full FFT of the input signal. The output is complex-valued,
        representing both magnitude and phase.
    frequency_bins : numpy.ndarray
        Array of frequency values corresponding to the FFT output, ranging
        from negative to positive frequencies (centered at zero).
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


def compute_ifft_and_return_real(spectrum):
    """Compute the inverse Fast Fourier Transform (IFFT) and return the real
    part.

    This function calculates the inverse Fourier Transform of a given spectrum
    and returns only the real part of the resulting time-domain signal. The
    IFFT is computed based on the provided spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The input spectrum, assumed to be complex-valued, representing the
        frequency-domain coefficients of the signal.

    Returns
    -------
    real_signal : numpy.ndarray
        The real part of the time-domain signal obtained by applying the IFFT
        to the input spectrum. The output is a 1D array of real values,
        representing the reconstructed signal.

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

    return np.fft.ifft(spectrum).real


def apply_spectral_slope(spectrum, frequencies, slope):
    """Modifica lo spettro del rumore bianco aggiungendo una pendenza
    lineare."""
    return spectrum + slope * frequencies

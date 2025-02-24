# -*- coding: utf-8 -*-
'''
Author: Chiara Malvaso
Date: February 2025
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime
from signal_toolkit import (
    generate_random_frequencies,
    generate_periodic_signal,
    add_white_noise,
    add_colored_noise,
)


def plot_clean_signal(time, signal):
    """Plot a clean periodic signal.

    This function plots a clean (noise-free) periodic signal over time.

    Parameters
    ----------
    time : numpy.ndarray
        1D array representing the time vector in seconds.

    signal : numpy.ndarray
        1D array representing the amplitude of the signal over time.

    Returns
    -------
    None
        The function displays the plot but does not return any value.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.title("Periodic Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_noisy_signal(time, signal, noisy_signal, snr):
    """Plot a clean and noisy signal for comparison.

    This function generates a subplot with two graphs:
    one for the clean signal and one for the noisy signal, based on the given Signal-to-Noise Ratio (SNR).

    Parameters
    ----------
    time : numpy.ndarray
        1D array representing the time vector in seconds.

    signal : numpy.ndarray
        1D array representing the clean (noise-free) signal.

    noisy_signal : numpy.ndarray
        1D array representing the noisy version of the signal.

    snr : float
        Signal-to-noise ratio (SNR) in decibels.

    Returns
    -------
    None
        The function displays the plot but does not return any value.
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title("Clean Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, noisy_signal)
    plt.title(f"Noisy Signal (SNR = {snr} dB)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def save_signal_to_csv(
    time,
    signal,
    noisy_signal=None,
    save_path=".",
    waveform_type="sin",
    sampling_rate=250,
    snr=None,
    noise_type="white",
):
    """Save signal data to a CSV file.

    This function saves the time-domain signal data into a CSV file,
    optionally including a noisy version if provided.

    Parameters
    ----------
    time : numpy.ndarray
        1D array representing the time vector in seconds.

    signal : numpy.ndarray
        1D array representing the clean (noise-free) signal.

    noisy_signal : numpy.ndarray, optional
        1D array representing the noisy signal (default: None).

    save_path : str, optional
        Directory path where the CSV file should be saved (default: current directory).

    waveform_type : str, optional
        Type of waveform used ("sin" for sinusoidal, "square" for square wave; default: "sin").

    sampling_rate : int, optional
        Sampling rate in Hz (default: 250 Hz).

    snr : float, optional
        Signal-to-noise ratio in dB. If None, no noisy signal is saved (default: None).

    noise_type : str, optional
        Type of noise added to the signal ('white' or 'colored'; default: 'white').

    Returns
    -------
    None
        The function writes the data to a file but does not return any value.

    Notes
    -----
    - If the directory specified in `save_path` does not exist, it will be created.
    - The filename follows the format:
      `signal_<waveform_type>_<sampling_rate>Hz[_<noise_type>_<snr>db]_d-m-Y_H-M-S.csv`
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    filename = f"signal_{waveform_type}_{sampling_rate}Hz"
    if snr is not None:
        filename += f"_{noise_type}_{snr}db"
        
    filename += f"_{timestamp}.csv"

    with open(os.path.join(save_path, filename), mode="w", newline="") as file:
        writer = csv.writer(file)
        if noisy_signal is not None:
            writer.writerow(["Time", "Signal", "Noisy Signal"])
            for t, s, ns in zip(time, signal, noisy_signal):
                writer.writerow([t, s, ns])
        else:
            writer.writerow(["Time", "Signal"])
            for t, s in zip(time, signal):
                writer.writerow([t, s])


def generate_and_plot_signal(args):
    """Generate and process a periodic signal.

    This function generates a periodic signal based on user-defined parameters,
    adds noise if requested, and optionally plots or saves the signal.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing the following signal parameters:

        num_components : int
            Number of signal components (ignored if `frequencies` is provided).

        duration : float
            Signal duration in seconds.

        sampling_rate : int
            Sampling frequency in Hz.

        waveform_type : str
            Type of waveform ('sin' or 'square').

        frequencies : str, optional
            Comma-separated list of frequencies in Hz. If not provided, random frequencies are generated.

        freq_seed : int, optional
            Random seed for frequency generation. If provided, ensures reproducibility of the randomly generated frequencies.

        snr : float, optional
            Signal-to-noise ratio in dB. If provided, noise is added.

        noise_type : str, optional
            Type of noise to add. Can be either 'white' for white noise or 'colored' for colored noise.

        slope : float, optional
            Spectral slope for colored noise. This is ignored if `noise_type` is set to 'white'.

        noise_seed : int, optional
            Random seed for noise generation. If provided, ensures reproducibility of the added noise.

        plot : bool
            If True, the generated signal is plotted.

        save : str, optional
            Directory path to save the signal as a CSV file.

    Returns
    -------
    None
        The function processes the signal but does not return any value.

    Notes
    -----
    - If `frequencies` is provided, those values are used directly.
    - Otherwise, `num_components` random frequencies are generated, optionally using `freq_seed` for reproducibility.
    - If `snr` is specified, white noise or colored noise is added to the signal based on the `noise_type` argument, 
      optionally using `noise_seed` for reproducibility.
    - If `plot` is set, the function plots the signal.
    - If `save` is set, the function saves the signal data to a CSV file, including both clean and noisy signals if noise is added.
    """
    if args.frequencies:
        frequencies = np.array([int(f) for f in args.frequencies.split(",")])
    else:
        frequencies = generate_random_frequencies(
            args.num_components, seed=args.freq_seed, sampling_rate=args.sampling_rate
        )

    print("Frequencies used:", np.sort(frequencies))

    time, signal = generate_periodic_signal(
        frequencies, args.duration, args.sampling_rate, args.waveform_type
    )

    noisy_signal = None
    if args.snr is not None:
        if args.noise_type == "white":
            noisy_signal = add_white_noise(signal, args.snr, seed=args.noise_seed)
        elif args.noise_type == "colored":
            noisy_signal = add_colored_noise(
                signal, args.snr, args.slope, args.sampling_rate, seed=args.noise_seed
            )
        plot_noisy_signal(time, signal, noisy_signal, args.snr)
    elif args.plot:
        plot_clean_signal(time, signal)

    if args.save:
        save_signal_to_csv(
            time,
            signal,
            noisy_signal,
            args.save,
            args.waveform_type,
            args.sampling_rate,
            args.snr,
            args.noise_type,
        )

# -*- coding: utf-8 -*-
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from signal_toolkit import (
    generate_random_frequencies,
    generate_periodic_signal,
    add_white_noise,
)


def parse_arguments():
    """Parse and handle command-line arguments for signal generation and
    plotting.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including number of components, duration, sampling rate,
        waveform type, plotting option, and optionally user-defined frequencies.
    """
    parser = argparse.ArgumentParser(
        description="Generate and plot a periodic signal with optional noise."
    )

    parser.add_argument(
        "--num_components",
        type=int,
        default=5,
        help="Number of signal components (default: 5). Ignored if --frequencies is provided.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Signal duration in seconds (default: 1.0 s).",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=250,
        help="Sampling frequency in Hz (default: 250).",
    )
    parser.add_argument(
        "--waveform_type",
        choices=["sin", "square"],
        default="sin",
        help="Waveform type ('sin' for sinusoidal wave, 'square' for square wave; default: 'sin').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, the generated signal will be plotted.",
    )
    parser.add_argument(
        "--frequencies",
        type=str,
        help="Comma-separated list of frequencies in Hz (e.g., '10,20,30'). If provided, --num_components is ignored.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        help="Signal-to-noise ratio in dB. If provided, noise will be added to the signal.",
    )

    parser.add_argument(
        "--save",
        nargs="?",
        const=".",
        type=str,
        help="Path to save the signal and time data as a CSV file. If no path is provided, the file will be saved in the current directory.",
    )
    return parser.parse_args()


def generate_and_plot_signal():
    """Generate a periodic signal, add noise if necessary, and optionally plot
    it."""
    args = parse_arguments()

    if args.frequencies:
        frequencies = np.array([int(f) for f in args.frequencies.split(",")])
    else:
        frequencies = generate_random_frequencies(
            args.num_components, args.sampling_rate
        )

    print("Frequencies used:", np.sort(frequencies))

    time, signal = generate_periodic_signal(
        frequencies, args.duration, args.sampling_rate, args.waveform_type
    )

    if args.snr is not None:
        noisy_signal = add_white_noise(signal, args.snr)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time, signal)
        plt.title("Clean Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time, noisy_signal)
        plt.title(f"Noisy Signal (SNR = {args.snr} dB)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Save CSV if option is provided
        if args.save:
            # Default to current directory if no path is provided
            save_path = args.save
            if not os.path.exists(save_path):
                os.makedirs(
                    save_path
                )  # Create the directory if it doesn't exist
            csv_filename = os.path.join(
                save_path,
                f"signal_{args.waveform_type}_{args.sampling_rate}Hz_{args.snr}db.csv",
            )

            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Signal", "Noisy Signal"])
                for t, s, ns in zip(time, signal, noisy_signal):
                    writer.writerow([t, s, ns])

        return time, signal, noisy_signal
    else:
        if args.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(time, signal)
            plt.title("Periodic Signal")
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

        # Save CSV if option is provided
        if args.save:
            save_path = args.save
            if not os.path.exists(save_path):
                os.makedirs(
                    save_path
                )  # Create the directory if it doesn't exist
            csv_filename = os.path.join(
                save_path,
                f"signal_{args.waveform_type}_{args.sampling_rate}Hz.csv",
            )

            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Signal"])
                for t, s in zip(time, signal):
                    writer.writerow([t, s])

        return time, signal


if __name__ == "__main__":
    generate_and_plot_signal()

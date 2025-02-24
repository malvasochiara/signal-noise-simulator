# -*- coding: utf-8 -*-
'''
Author: Chiara Malvaso
Date: February 2025
'''

import argparse
from utils import generate_and_plot_signal

def main():
    """Main function to parse command-line arguments and generate a periodic signal.

    This script uses the argparse library to collect parameters from the command line,
    such as signal components, duration, sampling rate, waveform type, and optional
    noise settings (SNR, noise type, slope). It then calls the `generate_and_plot_signal`
    function to generate and optionally plot the periodic signal and/or save the signal data
    to a CSV file.

    Command-line arguments include:

    Parameters
    ----------
    num_components : int, optional
        Number of signal components (default is 5). Ignored if `frequencies` is provided.
    duration : float, optional
        Signal duration in seconds (default is 1.0).
    sampling_rate : int, optional
        Sampling frequency in Hz (default is 250).
    waveform_type : {'sin', 'square'}, optional
        Type of waveform ('sin' for sinusoidal wave, 'square' for square wave; default is 'sin').
    frequencies : str, optional
        Comma-separated list of frequencies in Hz (e.g., '10,20,30'). If provided, `num_components` is ignored.
    snr : float, optional
        Signal-to-noise ratio in dB. If provided, noise will be added to the signal.
    noise_type : {'white', 'colored'}, optional
        Type of noise to add when `snr` is specified. Choices are 'white' (Gaussian white noise) or 'colored'
        (frequency-dependent noise that increases linearly). Default is 'white'. Ignored if `snr` is not provided.
    slope : float, optional
        Spectral slope for colored noise. Controls how noise power increases with frequency. Default is 0.5.
        Ignored if `snr` is not provided or if `noise_type` is 'white'. If `noise_type` is 'white' and `slope` is specified,
        an error will be raised.
    freq_seed : int, optional
        Random seed for frequency generation. If provided, ensures the same frequencies are generated
        on each execution. If not set, frequencies will vary randomly.
    noise_seed : int, optional
        Random seed for noise generation. If provided, ensures the same noise is added to the signal
        on each execution. If not set, noise will vary randomly.
    plot : bool, optional
        If set, the generated signal will be plotted.
    save : str, optional
        Path to save the signal and time data as a CSV file. If no path is provided, the file will be saved in
        the current directory.
        
    Notes
    -----
    - If `--snr` is not provided, noise-related arguments such as `--noise_type`, `--slope`, and `--noise_seed` 
      should not be specified. If they are provided, a warning message will be displayed.
    - If `--noise_type` is 'white' and `--slope` is specified, an error will be raised.

    Returns
    -------
    None
        The function does not return any value.
    """
    parser = argparse.ArgumentParser(
        description="Generate and plot a periodic signal with optional noise."
    )
    # Mutually exclusive group for num_components and frequencies
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--num_components",
        type=int,
        default=5,
        help="Number of signal components (default: 5). Cannot be specified with --frequencies.",
    )
    group.add_argument(
        "--frequencies",
        type=str,
        help="Comma-separated list of frequencies in Hz (e.g., '10,20,30'). Cannot be specified with --num_components.",
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
        "--snr",
        type=float,
        help="Signal-to-noise ratio in dB. If provided, noise will be added to the signal.",
    )
    parser.add_argument(
        "--noise_type",
        choices=["white", "colored"],
        default="white",
        help=( 
            "Type of noise to add when --snr is specified. Choices: 'white' (Gaussian "
            "white noise) or 'colored' (frequency-dependent noise that increases linearly). "
            "Default is 'white'. Ignored if --snr is not provided."
        ),
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=0.5,
        help=( 
            "Slope of the colored noise spectrum. Controls how noise power increases "
            "with frequency. Default is 0.5. Ignored if --snr is not provided or if "
            "noise type is 'white'. If --noise_type is 'white' and --slope is specified, an error will be raised."
        ),
    )
    
    parser.add_argument(
    "--freq_seed",
    type=int,
    help="Random seed for frequency generation. If not provided, results will be different on each run.",
    )
    
    parser.add_argument(
        "--noise_seed",
        type=int,
        help="Random seed for noise generation. If not provided, results will be different on each run.",
    )

    parser.add_argument(
        "--save",
        nargs="?",
        const=".",
        type=str,
        help="Path to save the signal and time data as a CSV file. If no path is provided, the file will be saved in the current directory.",
    )
    
    args = parser.parse_args()
    
    if args.snr is None:
        if args.noise_type != "white" or args.slope is not None or args.noise_seed is not None:
            print("Warning: If --snr is not provided, noise-related arguments (such as --noise_type, --slope, --noise_seed) have no effect.")
    
    if args.noise_type == "white" and args.slope is not None:
        parser.error("The --slope argument cannot be used when --noise_type is 'white'.")

    generate_and_plot_signal(args)
    
if __name__ == "__main__":
    main()

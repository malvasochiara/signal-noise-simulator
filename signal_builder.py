# -*- coding: utf-8 -*-
'''
Author: Chiara Malvaso
Date: February 2025
'''

import argparse
import sys
from utils import generate_and_plot_signal, parse_config_file


def main():
    """
    Main function to parse command-line arguments and configuration file,
    then generate a periodic signal.

    This script uses argparse for command-line arguments and configparser for
    reading settings from a configuration file. Command-line arguments take
    precedence over configuration file values.

    Command-line arguments and config.ini settings include:

    Parameters
    ----------
    config : str, optional
        Path to a configuration file (config.ini). If provided, parameters will be loaded from it.
    num_components : int, optional
        Number of signal components. Cannot be used with `frequencies`.
    duration : float, optional
        Signal duration in seconds (default is 1.0).
    sampling_rate : int, optional
        Sampling frequency in Hz (default is 250).
    waveform_type : {'sin', 'square'}, optional
        Type of waveform ('sin' for sinusoidal wave, 'square' for square wave; default is 'sin').
    frequencies : str, optional
        Comma-separated list of frequencies in Hz (e.g., '10,20,30'). Cannot be used with `num_components`.
    snr : float, optional
        Signal-to-noise ratio in dB. If provided, noise will be added to the signal.
    noise_type : {'white', 'colored'}, optional
        Type of noise to add when `snr` is specified. Default is 'white'.
    slope : float, optional
        Spectral slope for colored noise (default is 0.5). Ignored if `snr` is not provided or `noise_type` is 'white'.
    freq_seed : int, optional
        Random seed for frequency generation.
    noise_seed : int, optional
        Random seed for noise generation.
    plot : bool, optional
        If set, the generated signal will be plotted.
    save : str, optional
        Path to save the signal as a CSV file.

    Returns
    -------
    None
        The function does not return any value.

    Raises
    ------
    argparse.ArgumentError
        If `num_components` and `frequencies` are both specified.
    """
    parser = argparse.ArgumentParser(
        description="Generate and plot a periodic signal with optional noise."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (config.ini). Command-line arguments override config file values."
    )
    parser.add_argument(
        "--num_components",
        type=int,
        help="Number of signal components. Cannot be specified with --frequencies.",
    )
    parser.add_argument(
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
        help=( 
            "Slope of the colored noise spectrum. Controls how noise power increases "
            "with frequency. Ignored if --snr is not provided."
            "If --noise_type is 'white' and --slope is specified, an error will be raised."
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
    
    # Parse command-line arguments first
    args = parser.parse_args()
    
    # Load values from the configuration file, if provided
    config_args = {}
    if args.config:
        config_args = parse_config_file(args.config)
    
    # Override configuration file values only if they are set via command-line
    for key, value in vars(args).items():
        if f'--{key}' in sys.argv:  # Override only if the argument was passed via command-line
            config_args[key] = value

    # Combine default values, configuration file values, and command-line arguments
    # If a value is not present, keep the default
    for key, value in config_args.items():
        if value is None and key in parser._option_string_actions:  # If the value is None, use the default value
            setattr(args, key, parser.get_default(key))
        else:
            setattr(args, key, value)
    
    # Check for invalid combinations of arguments
    if args.num_components is not None and args.frequencies is not None:
        parser.error("The --num_components and --frequencies arguments cannot be used together.")
    
    if args.snr is None:
        if args.noise_type != "white" or args.slope is not None or args.noise_seed is not None:
            print("Warning: If --snr is not provided, noise-related arguments (such as --noise_type, --slope, --noise_seed) have no effect.")
    
    if args.noise_type == "white" and args.slope is not None:
        parser.error("The --slope argument cannot be used when --noise_type is 'white'.")
    
    print(vars(args))

    generate_and_plot_signal(args)
    
if __name__ == "__main__":
    main()
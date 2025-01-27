# -*- coding: utf-8 -*-
import numpy as np
import math
from signal_toolkit import signal_generator


def test_signal_length():
    """
    Test that signal_generator returns a signal with length equal to
    duration * sampling_rate.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The length of the signal is correct (duration * sampling_rate).
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    frequencies, time, signal = signal_generator(
        5, duration=2, sampling_rate=250
    )
    expected_length = 2 * 250  # duration * sampling rate
    assert (
        len(signal) == expected_length
    ), f"Signal length should be {expected_length}, but got {len(signal)}"


def test_frequency_range():
    """
    Test that frequencies are within the correct range (positive and
    lower or equal to the Nyquist frequency).

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: All generated frequencies verify Nyquist's theorem.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    frequencies, _, _ = signal_generator(5, duration=1, sampling_rate=250)
    max_frequency = math.floor(250 / 2)  # Nyquist frequency
    invalid_frequencies = [
        freq for freq in frequencies if not (1 <= freq <= max_frequency)
    ]
    assert len(invalid_frequencies) == 0, (
        f"The following frequencies are out of range (1 to {max_frequency}): "
        f"{invalid_frequencies}"
    )


def test_number_of_components():
    """
    Test that the number of components is between 1 and max_components.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The number of components is within the appropriate range.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    max_components = 5
    frequencies, _, _ = signal_generator(
        max_components, duration=1, sampling_rate=250
    )
    assert 1 <= len(frequencies) <= max_components, (
        f"Number of components should be between 1 and {max_components}, but "
        f"got {len(frequencies)}"
    )


def test_time_array():
    """
    Test that the time array is generated correctly.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The time array contains the correct points.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    _, time, _ = signal_generator(5, duration=2, sampling_rate=250)
    expected_time = np.linspace(0, 2, 250 * 2, endpoint=False)
    np.testing.assert_array_equal(
        time, expected_time, "Time array does not match expected values"
    )


def test_signal_amplitude():
    """
    Test that the signal has non-zero amplitude.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The signal has non-zero amplitude.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    _, _, signal = signal_generator(5, duration=1, sampling_rate=250)
    assert np.any(signal != 0), "Signal has zero amplitude"


def test_time_is_ndarray():
    """
    Test that the 'time' variable is of type numpy.ndarray.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The 'time' variable is of type numpy.ndarray.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    _, time, _ = signal_generator(5, duration=1, sampling_rate=250)
    assert isinstance(
        time, np.ndarray
    ), f"Time should be numpy.ndarray but got {type(time)}"


def test_signal_is_ndarray():
    """
    Test that the 'signal' variable is of type numpy.ndarray.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The 'signal' variable is of type numpy.ndarray.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    _, _, signal = signal_generator(5, duration=1, sampling_rate=250)
    assert isinstance(
        signal, np.ndarray
    ), f"Signal should be numpy.ndarray but got {type(signal)}"


def test_frequencies_is_ndarray():
    """
    Test that the 'frequencies' variable is of type numpy.ndarray.

    GIVEN: A valid maximum number of components, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The 'frequencies' variable is of type numpy.ndarray.
    """
    # Fix the seed for reproducibility
    np.random.seed(42)
    frequencies, _, _ = signal_generator(5, duration=1, sampling_rate=250)
    assert isinstance(
        frequencies, np.ndarray
    ), f"Frequencies should be numpy.ndarray but got {type(frequencies)}"

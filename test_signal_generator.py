# -*- coding: utf-8 -*-
import numpy as np
import pytest
from signal_toolkit import signal_generator


def test_signal_length():
    """
    Test that signal_generator returns a signal with length equal to
    duration * sampling_rate.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The length of the signal is correct (duration * sampling_rate).
    """

    _, signal = signal_generator(
        np.array([2, 10]), duration=2, sampling_rate=250
    )
    expected_length = 2 * 250  # duration * sampling rate
    assert (
        len(signal) == expected_length
    ), f"Signal length should be {expected_length}, but got {len(signal)}"


def test_signal_amplitude():
    """
    Test that the signal has non-zero amplitude.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The signal has non-zero amplitude.
    """

    _, signal = signal_generator(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert np.any(signal != 0), "Signal has zero amplitude"


def test_signal_is_ndarray():
    """
    Test that the 'signal' variable is of type numpy.ndarray.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The 'signal' variable is of type numpy.ndarray.
    """
    _, signal = signal_generator(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert isinstance(
        signal, np.ndarray
    ), f"Signal should be numpy.ndarray but got {type(signal)}"


def test_time_is_ndarray():
    """
    Test that the 'time' variable is of type numpy.ndarray.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The 'time' variable is of type numpy.ndarray.
    """
    time, _ = signal_generator(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert isinstance(
        time, np.ndarray
    ), f"Time should be numpy.ndarray but got {type(time)}"


def test_time_array():
    """
    Test that the time array is generated correctly.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: The time array contains the correct points.
    """

    time, _ = signal_generator(
        np.array([2, 10]), duration=2, sampling_rate=250
    )
    expected_time = np.linspace(0, 2, 250 * 2, endpoint=False)
    np.testing.assert_array_equal(
        time, expected_time, "Time array does not match expected values"
    )


def test_invalid_frequencies_value():
    """
    Test that the signal_generator raises an error when a frequency smaller than or equal to zero
    is provided in the frequencies array.
    GIVEN: An array of frequencies containing at least a wrong value, valid signal duration and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError, match="Frequencies should be positive and non-zero"
    ):
        signal_generator(np.array([10, -3]), duration=1, sampling_rate=250)


def test_invalid_frequencies_type():
    """
    Test that the signal_generator raises an error when a not integer frequency
    is provided in the frequencies array.
    GIVEN: An array of frequencies containing at least a non-integer value, valid signal duration and
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(TypeError, match="Frequencies should be integer"):
        signal_generator(
            np.array([10, "undici"]), duration=1, sampling_rate=250
        )

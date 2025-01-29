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


def test_invalid_duration_type():
    """
    Test that the signal_generator raises an error when a not float duration
    is provided.
    GIVEN: A valid array of frequencies, non-float signal duration and a valid
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError, match="Duration should be a number, either integer or float"
    ):
        signal_generator(
            [10, 20], duration=np.array([5, 10]), sampling_rate=250
        )


def test_invalid_duration_value():
    """
    Test that the signal_generator raises an error when a negative duration
    is provided.
    GIVEN: A valid array of frequencies, a negative signal duration and a valid
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError, match="Duration should be greater than or equal to 0"
    ):
        signal_generator(np.array([5, 10]), duration=-0.5, sampling_rate=250)


def test_zero_duration():
    """
    Test that the signal_generator returns an empty array when duration = 0 s.
    GIVEN: A valid array of frequencies, signal duration = 0 and a valid
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: signal is an empty array.
    """
    _, signal = signal_generator(
        np.array([2, 10]), duration=0, sampling_rate=250
    )
    assert (
        signal.size == 0
    ), f"Expected an empty array, but got an array of size {signal.size}."


def test_invalid_sampling_rate_type():
    """
    Test that the signal_generator raises an error when a not float or integer sampling rate
    is provided.
    GIVEN: A valid array of frequencies and duration, a non float or integer
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError,
        match="Sampling rate should be a number, either integer or float",
    ):
        signal_generator(
            np.array([10, 20]), duration=2, sampling_rate="duecentocinquanta"
        )


def test_invalid_sampling_rate_value():
    """
    Test that the signal_generator raises an error when a negative sampling rate
    is provided.
    GIVEN: A valid array of frequencies and duration, a negative
           sampling rate.
    WHEN: The signal_generator function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError,
        match="Sampling rate should be greater than or equal to 0",
    ):
        signal_generator(np.array([10, 20]), duration=2, sampling_rate=-250)


def test_zero_sampling_rate():
    """
    Test that the signal_generator returns an empty array when sampling_rate = 0 Hz.
    GIVEN: A valid array of frequencies and signal duration = 0, sampling_rate = 0.
    WHEN: The signal_generator function is called with these parameters.
    THEN: signal is an empty array.
    """
    _, signal = signal_generator(
        np.array([2, 10]), duration=2, sampling_rate=0
    )
    assert (
        signal.size == 0
    ), f"Expected an empty array, but got an array of size {signal.size}."

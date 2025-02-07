# -*- coding: utf-8 -*-
import numpy as np
import pytest
from signal_toolkit import (
    generate_sinusoidal_signal,
    generate_random_frequencies,
    compute_signal_power,
    compute_white_noise_std,
)


def test_single_frequency():
    """Test that generate_sinusoidal_signal     correctly generates a single
    sinusoidal wave.

    GIVEN: A valid array of frequencies containing just one value, a valid signal duration and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: The generated signal is the expected sinusoidal wave.
    """
    time, signal = generate_sinusoidal_signal(
        np.array([10]), duration=1, sampling_rate=250
    )
    expected_signal = np.sin(2 * np.pi * 10 * time)
    np.testing.assert_allclose(
        signal,
        expected_signal,
        atol=1e-6,
        err_msg="Single frequency signal mismatch",
    )


def test_signal_length():
    """Test that generate_sinusoidal_signal     returns a signal with length
    equal to duration * sampling_rate.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: The length of the signal is correct (duration * sampling_rate).
    """

    _, signal = generate_sinusoidal_signal(
        np.array([2, 10]), duration=2, sampling_rate=250
    )
    expected_length = 2 * 250  # duration * sampling rate
    assert (
        len(signal) == expected_length
    ), f"Signal length should be {expected_length}, but got {len(signal)}"


def test_signal_amplitude():
    """Test that the signal has non-zero amplitude.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: The signal has non-zero amplitude.
    """

    _, signal = generate_sinusoidal_signal(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert np.any(signal != 0), "Signal has zero amplitude"


def test_signal_is_ndarray():
    """Test that the 'signal' variable is of type numpy.ndarray.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: The 'signal' variable is of type numpy.ndarray.
    """
    _, signal = generate_sinusoidal_signal(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert isinstance(
        signal, np.ndarray
    ), f"Signal should be numpy.ndarray but got {type(signal)}"


def test_time_is_ndarray():
    """Test that the 'time' variable is of type numpy.ndarray.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: The 'time' variable is of type numpy.ndarray.
    """
    time, _ = generate_sinusoidal_signal(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert isinstance(
        time, np.ndarray
    ), f"Time should be numpy.ndarray but got {type(time)}"


def test_time_array():
    """Test that the time array is generated correctly.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: The time array contains the correct points.
    """

    time, _ = generate_sinusoidal_signal(
        np.array([2, 10]), duration=2, sampling_rate=250
    )
    expected_time = np.linspace(0, 2, 250 * 2, endpoint=False)
    np.testing.assert_array_equal(
        time, expected_time, "Time array does not match expected values"
    )


def test_negative_frequencies_value():
    """Test that the generate_sinusoidal_signal         raises an error when a
    frequency smaller than or equal to zero is provided in the frequencies
    array.

    GIVEN: An array of frequencies containing at least a wrong value, valid signal duration and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError, match="Frequencies should be positive and non-zero"
    ):
        generate_sinusoidal_signal(
            np.array([10, -3]), duration=1, sampling_rate=250
        )


def test_invalid_frequencies_value():
    """Test that the generate_sinusoidal_signal         raises an error when a
    frequency greater than or equal to Nyquist's frequency' is provided in the
    frequencies array.

    GIVEN: An array of frequencies containing at least a wrong value, valid signal duration and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError,
        match="Frequencies should be smaller than Nyquist's frequency, sampling_rate/2",
    ):
        generate_sinusoidal_signal(
            np.array([10, 200]), duration=1, sampling_rate=250
        )


def test_invalid_frequencies_type():
    """Test that the generate_sinusoidal_signal         raises an error when a
    not integer frequency is provided in the frequencies array.

    GIVEN: An array of frequencies containing at least a non-integer value, valid signal duration and
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(TypeError, match="Frequencies should be integer"):
        generate_sinusoidal_signal(
            np.array([10, "undici"]), duration=1, sampling_rate=250
        )


def test_invalid_duration_type():
    """Test that the generate_sinusoidal_signal         raises an error when a
    not float duration is provided.

    GIVEN: A valid array of frequencies, non-float signal duration and a valid
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError, match="Duration should be a number, either integer or float"
    ):
        generate_sinusoidal_signal(
            [10, 20], duration=np.array([5, 10]), sampling_rate=250
        )


def test_invalid_duration_value():
    """Test that the generate_sinusoidal_signal         raises an error when a
    negative duration is provided.

    GIVEN: A valid array of frequencies, a negative signal duration and a valid
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError, match="Duration should be greater than or equal to 0"
    ):
        generate_sinusoidal_signal(
            np.array([5, 10]), duration=-0.5, sampling_rate=250
        )


def test_zero_duration():
    """Test that the generate_sinusoidal_signal         returns an empty array
    when duration = 0 s.

    GIVEN: A valid array of frequencies, signal duration = 0 and a valid
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: signal is an empty array.
    """
    _, signal = generate_sinusoidal_signal(
        np.array([2, 10]), duration=0, sampling_rate=250
    )
    assert (
        signal.size == 0
    ), f"Expected an empty array, but got an array of size {signal.size}."


def test_invalid_sampling_rate_type():
    """Test that the generate_sinusoidal_signal         raises an error when a
    not float or integer sampling rate is provided.

    GIVEN: A valid array of frequencies and duration, a non float or integer
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError,
        match="Sampling rate should be a number, either integer or float",
    ):
        generate_sinusoidal_signal(
            np.array([10, 20]), duration=2, sampling_rate="duecentocinquanta"
        )


def test_invalid_sampling_rate_value():
    """Test that the generate_sinusoidal_signal         raises an error when a
    negative sampling rate is provided.

    GIVEN: A valid array of frequencies and duration, a negative
           sampling rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError,
        match="Sampling rate should be greater than or equal to 0",
    ):
        generate_sinusoidal_signal(
            np.array([10, 20]), duration=2, sampling_rate=-250
        )


def test_empty_frequencies():
    """Test that the generate_sinusoidal_signal         returns an empty array
    when frequencies is an empty array.

    GIVEN: An empty array of frequencies, a valid signal duration and sampling_rate.
    WHEN: The generate_sinusoidal_signal         function is called with these parameters.
    THEN: signal is an empty array.
    """
    _, signal = generate_sinusoidal_signal(
        np.array([]), duration=2, sampling_rate=350
    )
    assert np.all(
        signal == 0
    ), "Expected an array of zeros, but found nonzero values."


# tests for generate_random_frequencies


def test_frequencies_range_with_default_sampling_rate():
    """Test that the generate_random_frequencies        returns an array of
    frequencies within the appropriate range, from 1 to the Nyquist's
    frequency, when called with the default parameter.

    GIVEN: A valid num_components and the default value for sampling_rate.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: frequencies are all greater than or equal to 1 and smaller than 125.
    """
    np.random.seed(42)

    num_components = 10
    frequencies = generate_random_frequencies(num_components)
    assert np.all(
        (frequencies >= 1) & (frequencies < 125)
    ), "Frequencies should be between 1 and 125 (for sampling_rate=250)"


def test_frequencies_length_with_default_sampling_rate():
    """Test that the generate_random_frequencies        returns an array of
    frequencies with the appropriate length, when called with the default
    parameter.

    GIVEN: A valid num_components and the default value for sampling_rate.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: frequencies is an array of length equal to num_components.
    """
    np.random.seed(42)

    num_components = 10
    frequencies = generate_random_frequencies(num_components)
    assert (
        len(frequencies) == num_components
    ), f"Expected {num_components} frequencies, but got {len(frequencies)}."


def test_frequencies_range_with_valid_sampling_rate():
    """Test that the generate_random_frequencies        returns an array of
    frequencies within the appropriate range, from 1 to the Nyquist's
    frequency, when called with a valid sampling_rate.

    GIVEN: A valid num_components and sampling_rate.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: frequencies are all greater than or equal to 1 and smaller than sampling_rate/2.
    """
    np.random.seed(42)

    num_components = 10
    sampling_rate = 350
    frequencies = generate_random_frequencies(num_components)
    assert np.all(
        (frequencies >= 1) & (frequencies < sampling_rate / 2)
    ), "Frequencies should be between 1 and Nyquist's frequency"


def test_frequencies_length_with_valid_sampling_rate():
    """Test that the generate_random_frequencies        returns an array of
    frequencies with the appropriate length, when called with a valid
    sampling_rate.

    GIVEN: A valid num_components and sampling_rate.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: frequencies is an array of length equal to num_components.
    """
    np.random.seed(42)

    num_components = 10
    sampling_rate = 350
    frequencies = generate_random_frequencies(num_components, sampling_rate)
    assert (
        len(frequencies) == num_components
    ), f"Expected {num_components} frequencies, but got {len(frequencies)}."


def test_invalid_numcomponents_type():
    """Test that the generate_random_frequencies        raises an error when a
    not integer number of components is provided.

    GIVEN: An invalid number of components, default sampling_rate
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    np.random.seed(42)

    with pytest.raises(
        TypeError,
        match="num_components should be an integer number",
    ):
        generate_random_frequencies("cinque")


def test_invalid_numcomponents_value():
    """Test that the generate_random_frequencies        raises an error when a
    negative number of components is provided.

    GIVEN: A negative number of components, default sampling_rate
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: A ValueError is raised.
    """
    np.random.seed(42)

    with pytest.raises(ValueError):
        generate_random_frequencies(-8)


def test_zero_numcomponents():
    """Test that the generate_random_frequencies        returns an empty array
    when 0 number of components is provided.

    GIVEN: A negative number of components, default sampling_rate
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: An empty array is returned.
    """
    np.random.seed(42)

    frequencies = generate_random_frequencies(0)
    assert (
        len(frequencies) == 0
    ), f"Expected an empty frequencies array, but got {len(frequencies)}."


def test_invalid_sampling_rate_type_random_frequencies():
    """Test that the generate_random_frequencies        raises an error when a
    not integer sampling rate is provided.

    GIVEN: A valid number of components, invalid sampling_rate
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: A TypeError is raised with the appropriate message.
    """
    np.random.seed(42)

    with pytest.raises(
        TypeError,
        match="sampling_rate should be an integer number",
    ):
        generate_random_frequencies(5, "duecentocinquanta")


def test_invalid_sampling_rate_value_random_frequencies():
    """Test that the generate_random_frequencies        raises an error when a
    negative sampling rate is provided.

    GIVEN: A valid number of components, a sampling_rate smaller than or equal to 0.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: A ValueError is raised.
    """
    np.random.seed(42)

    with pytest.raises(
        ValueError,
        match="sampling_rate should be greater than or equal to 4",
    ):
        generate_random_frequencies(5, -17)


def test_minimal_valid_sampling_rate():
    """Test that generate_random_frequencies    works correctly when
    sampling_rate is the minimum allowed value.

    GIVEN: A valid number of components, sampling_rate = 4.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: Frequencies is an array with only 1 and 2
    """
    np.random.seed(42)
    frequencies = generate_random_frequencies(50, 4)
    assert np.all(
        np.isin(frequencies, [1, 2])
    ), "Expected frequency to be 1 or 2 when sampling_rate is 4."


def test_minimal_valid_numcomponent():
    """Test that generate_random_frequencies    works when num_components is
    the minimum allowed value.

    GIVEN: num_components = 1, a valid sampling_rate.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: Frequencies is an array of length 1
    """
    np.random.seed(42)
    frequencies = generate_random_frequencies(1, 100)
    assert len(frequencies) == 1, "Expected a single frequency."


def test_large_number_of_components():
    """Test that generate_random_frequencies    can handle large input sizes
    efficiently.

    GIVEN: num_components = 10^8, a valid sampling_rate.
    WHEN: The generate_random_frequencies        function is called with these parameters.
    THEN: Frequencies is an array of length 10^8
    """
    np.random.seed(42)
    frequencies = generate_random_frequencies(10**8, 500)
    assert (
        len(frequencies) == 10**8
    ), f"Expected 10^8 frequencies, but got {len(frequencies)}"


# Test for compute_signal_power


def test_compute_signal_power_type():
    """Test that compute_signal_power returns the correct output type when
    given a valid input.

    GIVEN: A valid 1D numpy array.
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: The return value is of type float
    """
    signal = np.array([1, 2, 3, 4, 5])
    result = compute_signal_power(signal)
    assert isinstance(
        result, float
    ), f"Expected output to be of type float, but got {type(result)}"


def test_compute_signal_power_basic():
    """Test that compute_signal_power returns the correct power when input signal is constant
    GIVEN: A constant signal where all elements are equal (e.g., an array of ones).
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: The RMS power is equal to the value of the constant signal.

    """

    signal = np.ones(100)
    result = compute_signal_power(signal)
    assert result == 1.0, f"Expected power to be 1.0, but got {result}"


def test_compute_signal_power_invalid_signal_type():
    """Test that compute_signal_power raises a TypeError when provided with an
    invalid input type.

    GIVEN: An invalid input (e.g., a string) instead of a numpy array representing a signal.
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: A TypeError is raised.
    """
    with pytest.raises(TypeError):
        compute_signal_power(np.array(["signal", True]))


def test_compute_signal_power_single_element():
    """Test that compute_signal_power returns the correct power for a single-
    element signal.

    GIVEN: A signal with a single element (e.g., an array with one value).
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: The RMS power is equal to the value of the signal.
    """
    signal = np.array([5])
    assert compute_signal_power(signal) == 5.0, "Expected power to be 5.0"


def test_compute_signal_power_large_signal():
    """Test that compute_signal_power handles a large signal correctly.

    GIVEN: A large random signal with a very large number of elements (e.g., 10^8 elements).
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: The return value is of type float.
    """
    np.random.seed(42)
    signal = np.random.rand(10**8)
    assert isinstance(
        compute_signal_power(signal), float
    ), "Expected output to be a float"


def test_compute_signal_power_invalid_signal_element():
    """Test that compute_signal_power raises a TypeError when provided with an
    invalid element in signal array.

    GIVEN: A numpy array containing an invalid type.
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: A TypeError is raised.
    """
    signal = np.array([1, 2, 3, None, 4, 5])
    with pytest.raises(TypeError):
        compute_signal_power(signal)


def test_compute_signal_power_empty_signal():
    """Test that compute_signal_power raises a ValueError when provided with an
    empty array.

    GIVEN: An empty numpy array.
    WHEN: The compute_signal_power function is called with this parameter.
    THEN: A ValueError is raised with the appropriate message.
    """
    signal = np.array([])
    with pytest.raises(
        ValueError, match="Input signal must be a non-empty array."
    ):
        compute_signal_power(signal)


# Test for compute_white_noise_std


def test_compute_white_noise_std_output_type():
    """Test that compute_white_noise_std returns a float.

    GIVEN: A valid signal and SNR.
    WHEN: The compute_signal_power function is called with these parameters.
    THEN: The return value is of type float.
    """
    signal = np.arange(1, 50, 0.5)
    noise_std = compute_white_noise_std(signal, 10)
    assert isinstance(
        noise_std, float
    ), f"Expected float, got {type(noise_std)}"


def test_compute_white_noise_std_zero_snr():
    """Test that with 0 dB SNR, noise std equals signal RMS.

    GIVEN: A valid signal and SNR = 0.
    WHEN: The compute_white_noise_std function is called with these parameters.
    THEN: The return value equals the RMS power of the input signal.
    """
    signal = np.arange(1, 20, 0.5)
    expected_std = compute_signal_power(signal)
    assert np.isclose(
        compute_white_noise_std(signal, 0), expected_std
    ), "Noise std should equal signal RMS for 0 dB SNR"


def test_compute_white_noise_std_high_snr():
    """Test that for high SNR, noise std approaches zero.

    GIVEN: A valid signal and a high SNR (e.g., 1000 dB).
    WHEN: The compute_white_noise_std function is called with these parameters.
    THEN: The return value is close to zero.
    """
    signal = np.arange(1, 20, 0.5)
    assert (
        compute_white_noise_std(signal, 1000) < 1e-6
    ), "Noise std should be close to zero for high SNR"


def test_compute_white_noise_std_negative_snr():
    """Test that for negative SNR, noise std is larger than signal RMS.

    GIVEN: A valid signal and a negative SNR.
    WHEN: The compute_white_noise_std function is called with these parameters.
    THEN: The return value is greater than the signal power.
    """
    signal = np.arange(1, 20, 0.5)
    assert compute_white_noise_std(signal, -5) > compute_signal_power(
        signal
    ), "Noise std should be greater than signal RMS for negative SNR"


def test_compute_white_noise_std_constant_signal():
    """Test that for a constant signal, noise std is computed correctly.

    GIVEN: A constant signal (e.g., an array of ones) and a valid SNR.
    WHEN: The compute_white_noise_std function is called with these parameters.
    THEN: The return value is the expected noise standard deviation.
    """
    signal = np.ones(50)
    expected_std = 1 / (10 ** (10 / 10))
    assert np.isclose(
        compute_white_noise_std(signal, 10), expected_std
    ), "Noise std incorrect for constant signal"


def test_compute_white_noise_std_invalid_snr_type():
    """Test that compute_white_noise_std raises a TypeError when provided with
    an invalid SNR type.

    GIVEN: A valid signal and an invalid type for SNR.
    WHEN: The compute_white_noise_std function is called with these parameters.
    THEN: A TypeError is raised.
    """
    signal = np.arange(1, 20, 0.5)
    with pytest.raises(
        TypeError,
        match="SNR should be a number, either integer or float",
    ):
        compute_white_noise_std(signal, "dieci")

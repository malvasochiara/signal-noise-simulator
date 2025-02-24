# -*- coding: utf-8 -*-
'''
Author: Chiara Malvaso
Date: February 2025
'''

import numpy as np
import pytest
from signal_toolkit import (
    generate_periodic_signal,
    generate_random_frequencies,
    compute_signal_power,
    compute_white_noise_std,
    generate_white_noise,
    add_white_noise,
    compute_fft,
    compute_ifft,
    apply_spectral_slope,
    add_colored_noise,
)

# tests for generate_random_frequencies


def test_frequencies_range_with_default_sampling_rate():
    """Test that the generate_random_frequencies returns an array of frequencies within
    the appropriate range, from 1 to the Nyquist's frequency, when called with the
    default parameter.

    GIVEN: A valid num_components, fixed random seed and default value for sampling_rate.

    WHEN: The generate_random_frequencies function is called with these parameters.

    THEN: frequencies are all greater than or equal to 1 and smaller than 125.
    """
    random_seed = 42

    num_components = 10
    frequencies = generate_random_frequencies(num_components,random_seed)
    assert np.all(
        (frequencies >= 1) & (frequencies < 125)
    ), "Frequencies should be between 1 and 125 (for sampling_rate=250)"


def test_frequencies_length_with_default_sampling_rate():
    """Test that the generate_random_frequencies returns an array of frequencies
    with the appropriate length, when called with the default parameter.

    GIVEN: A valid num_components, fixed random seed and default value for sampling_rate.

    WHEN: The generate_random_frequencies function is called with these parameters.

    THEN: frequencies is an array of length equal to num_components.
    """
    random_seed = 42

    num_components = 10
    frequencies = generate_random_frequencies(num_components, random_seed)
    assert (
        len(frequencies) == num_components
    ), f"Expected {num_components} frequencies, but got {len(frequencies)}."


def test_frequencies_range_with_valid_sampling_rate():
    """Test that the generate_random_frequencies returns an array of frequencies
    within the appropriate range, from 1 to the Nyquist's frequency, when called with a
    valid sampling_rate.

    GIVEN: A valid num_components, fixed random seed and sampling_rate.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: frequencies are all greater than or equal to 1 and smaller than sampling_rate/2.
    """
    random_seed = 42

    num_components = 10
    sampling_rate = 350
    frequencies = generate_random_frequencies(num_components, random_seed)
    assert np.all(
        (frequencies >= 1) & (frequencies < sampling_rate / 2)
    ), "Frequencies should be between 1 and Nyquist's frequency"


def test_frequencies_length_with_valid_sampling_rate():
    """Test that the generate_random_frequencies returns an array of frequencies
    with the appropriate length, when called with a valid sampling_rate.

    GIVEN: A valid num_components, fixed random seed and valid sampling_rate.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: frequencies is an array of length equal to num_components.
    """
    random_seed = 42

    num_components = 10
    sampling_rate = 350
    frequencies = generate_random_frequencies(num_components, sampling_rate, random_seed)
    assert (
        len(frequencies) == num_components
    ), f"Expected {num_components} frequencies, but got {len(frequencies)}."


def test_invalid_numcomponents_type():
    """Test that the generate_random_frequencies raises an error when a not integer
    number of components is provided.

    GIVEN: An invalid number of components, fixed random seed and default sampling_rate
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    random_seed = 42

    with pytest.raises(
        TypeError,
        match="num_components should be an integer number",
    ):
        generate_random_frequencies("cinque", random_seed)


def test_invalid_numcomponents_value():
    """Test that the generate_random_frequencies raises an error when a negative number
    of components is provided.

    GIVEN: A negative number of components, fixed random seed and default sampling_rate
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: A ValueError is raised.
    """
    random_seed = 42

    with pytest.raises(ValueError):
        generate_random_frequencies(-8, random_seed)


def test_zero_numcomponents():
    """Test that the generate_random_frequencies returns an empty array when 0
    number of components is provided.

    GIVEN: A negative number of components, fixed random seed and default sampling_rate.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: An empty array is returned.
    """
    random_seed = 42

    frequencies = generate_random_frequencies(0, random_seed)
    assert (
        len(frequencies) == 0
    ), f"Expected an empty frequencies array, but got {len(frequencies)}."


def test_invalid_sampling_rate_type_random_frequencies():
    """Test that the generate_random_frequencies raises an error when a not
    integer sampling rate is provided.

    GIVEN: A valid number of components, fixed random seed and invalid sampling_rate
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    random_seed = 42

    with pytest.raises(
        TypeError,
        match="sampling_rate should be an integer number",
    ):
        generate_random_frequencies(5,random_seed, "duecentocinquanta")


def test_invalid_sampling_rate_value_random_frequencies():
    """Test that the generate_random_frequencies raises an error when a negative
    sampling rate is provided.

    GIVEN: A valid number of components, fixed random seed, a sampling_rate smaller than or equal to 0.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: A ValueError is raised.
    """
    random_seed = 42

    with pytest.raises(
        ValueError,
        match="sampling_rate should be greater than or equal to 4",
    ):
        generate_random_frequencies(5, random_seed, -17)


def test_minimal_valid_sampling_rate():
    """Test that generate_random_frequencies works correctly when sampling_rate is
    the minimum allowed value.

    GIVEN: A valid number of components, fixed random seed, sampling_rate = 4.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: Frequencies is an array with only 1 and 2
    """
    random_seed = 42
    frequencies = generate_random_frequencies(50, random_seed, 4)
    assert np.all(
        np.isin(frequencies, [1, 2])
    ), "Expected frequency to be 1 or 2 when sampling_rate is 4."


def test_minimal_valid_numcomponent():
    """Test that generate_random_frequencies works when num_components is the minimum
    allowed value.

    GIVEN: num_components = 1,fixed random seed, a valid sampling_rate.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: Frequencies is an array of length 1
    """
    random_seed = 42
    frequencies = generate_random_frequencies(1, random_seed, 100)
    assert len(frequencies) == 1, "Expected a single frequency."


def test_large_number_of_components():
    """Test that generate_random_frequencies can handle large input sizes
    efficiently.

    GIVEN: num_components = 10^8, fixed random seed, a valid sampling_rate.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: Frequencies is an array of length 10^8
    """
    random_seed = 42
    frequencies = generate_random_frequencies(10**8,random_seed, 500)
    assert (
        len(frequencies) == 10**8
    ), f"Expected 10^8 frequencies, but got {len(frequencies)}"


def test_invalid_seed_type ():
    """Test that the generate_random_frequencies raises an error when a non-integer 
    seed is provided.

    GIVEN: A valid number of components, default sampling_rate and floating seed.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: A TypeError is raised.
    """

    with pytest.raises(TypeError):
        generate_random_frequencies(10, 250, 1.3)
        
        
def test_invalid_seed_value ():
    """Test that the generate_random_frequencies raises an error when a negative 
    seed is provided.

    GIVEN: A valid number of components, default sampling_rate and negative seed.
    
    WHEN: The generate_random_frequencies function is called with these parameters.
    
    THEN: A ValueError is raised.
    """

    with pytest.raises(ValueError):
        generate_random_frequencies(10, 250, -15)
        
        
def test_same_seed():
    """Test that the generate_random_frequencies return the same array when called two times
        withe the same seed
        
    GIVEN: A valid number of components, default sampling_rate and valid seed.
    
    WHEN: The generate_random_frequencies function is called two times with these parameters.
    
    THEN: The returned frequencies array are the same.
    """

    frequencies_1 = generate_random_frequencies(10, 250, 15)
    frequencies_2 = generate_random_frequencies(10, 250, 15)
    assert np.array_equal(frequencies_1, frequencies_2), f"Arrays are not equal: {frequencies_1} != {frequencies_2}"
    
# Tests for generate_periodic_signal


def test_generate_periodic_signal_single_frequency():
    """Test that generate_periodic_signal correctly generates a single sinusoidal wave.

    GIVEN: A valid array of frequencies containing just one value, a valid signal duration and
           sampling rate, default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The generated signal is the expected sinusoidal wave.
    """
    time, signal = generate_periodic_signal(
        np.array([10]), duration=1, sampling_rate=250
    )
    expected_signal = np.sin(2 * np.pi * 10 * time)
    np.testing.assert_allclose(
        signal,
        expected_signal,
        atol=1e-6,
        err_msg="Single frequency signal mismatch",
    )


def test_generate_periodic_signal_signal_length():
    """Test that generate_periodic_signal     returns a signal with length equal to
    duration * sampling_rate.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate, default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The length of the signal is correct (duration * sampling_rate).
    """

    _, signal = generate_periodic_signal(
        np.array([2, 10]), duration=2, sampling_rate=250
    )
    expected_length = 2 * 250  # duration * sampling rate
    assert (
        len(signal) == expected_length
    ), f"Signal length should be {expected_length}, but got {len(signal)}"


def test_generate_periodic_signal_signal_amplitude():
    """Test that the signal has non-zero amplitude.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate, and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The signal has non-zero amplitude.
    """

    _, signal = generate_periodic_signal(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert np.any(signal != 0), "Signal has zero amplitude"


def test_generate_periodic_signal_signal_is_ndarray():
    """Test that the 'signal' variable is of type numpy.ndarray.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate, and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The 'signal' variable is of type numpy.ndarray.
    """
    _, signal = generate_periodic_signal(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert isinstance(
        signal, np.ndarray
    ), f"Signal should be numpy.ndarray but got {type(signal)}"


def test_generate_periodic_signal_time_is_ndarray():
    """Test that the 'time' variable is of type numpy.ndarray.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The 'time' variable is of type numpy.ndarray.
    """
    time, _ = generate_periodic_signal(
        np.array([2, 10]), duration=1, sampling_rate=250
    )
    assert isinstance(
        time, np.ndarray
    ), f"Time should be numpy.ndarray but got {type(time)}"


def test_generate_periodic_signal_time_array():
    """Test that the time array is generated correctly.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The time array contains the correct points.
    """

    time, _ = generate_periodic_signal(
        np.array([2, 10]), duration=2, sampling_rate=250
    )
    expected_time = np.linspace(0, 2, 250 * 2, endpoint=False)
    np.testing.assert_array_equal(
        time, expected_time, "Time array does not match expected values"
    )


def test_generate_periodic_signal_negative_frequencies_value():
    """Test that the generate_periodic_signal raises an error when a frequency smaller
    than or equal to zero is provided in the frequencies array.

    GIVEN: An array of frequencies containing at least a wrong value, valid signal duration and
           sampling rate, default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError, match="Frequencies should be positive and non-zero"
    ):
        generate_periodic_signal(
            np.array([10, -3]), duration=1, sampling_rate=250
        )


def test_generate_periodic_signal_invalid_frequencies_value():
    """Test that the generate_periodic_signal raises an error when a frequency greater
    than or equal to Nyquist's frequency' is provided in the frequencies array.

    GIVEN: An array of frequencies containing at least a wrong value, valid signal duration and
           sampling rate, default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError,
        match="Frequencies should be smaller than Nyquist's frequency, sampling_rate/2",
    ):
        generate_periodic_signal(
            np.array([10, 200]), duration=1, sampling_rate=250
        )


def test_generate_periodic_signal_invalid_frequencies_type():
    """Test that the generate_periodic_signal   raises an error when a not integer
    frequency is provided in the frequencies array.

    GIVEN: An array of frequencies containing at least a non-integer value, valid signal duration and
           sampling rate, default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(TypeError, match="Frequencies should be integer"):
        generate_periodic_signal(
            np.array([10, "undici"]), duration=1, sampling_rate=250
        )


def test_generate_periodic_signal_invalid_duration_type():
    """Test that the generate_periodic_signal   raises an error when a not float
    duration is provided.

    GIVEN: A valid array of frequencies, non-float signal duration, a valid
           sampling rate and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError, match="Duration should be a number, either integer or float"
    ):
        generate_periodic_signal(
            [10, 20], duration=np.array([5, 10]), sampling_rate=250
        )


def test_generate_periodic_signal_invalid_duration_value():
    """Test that the generate_periodic_signal   raises an error when a negative
    duration is provided.

    GIVEN: A valid array of frequencies, a negative signal duration, a valid
           sampling rate and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError, match="Duration should be greater than or equal to 0"
    ):
        generate_periodic_signal(
            np.array([5, 10]), duration=-0.5, sampling_rate=250
        )


def test_generate_periodic_signal_zero_duration():
    """Test that the generate_periodic_signal returns an empty array when
    duration = 0 s.

    GIVEN: A valid array of frequencies, signal duration = 0, a valid
           sampling rate and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: signal is an empty array.
    """
    _, signal = generate_periodic_signal(
        np.array([2, 10]), duration=0, sampling_rate=250
    )
    assert (
        signal.size == 0
    ), f"Expected an empty array, but got an array of size {signal.size}."


def test_generate_periodic_signal_invalid_sampling_rate_type():
    """Test that the generate_periodic_signal   raises an error when a not float
    or integer sampling rate is provided.

    GIVEN: A valid array of frequencies and duration, a non float or integer
           sampling rate, default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError,
        match="Sampling rate should be a number, either integer or float",
    ):
        generate_periodic_signal(
            np.array([10, 20]), duration=2, sampling_rate="duecentocinquanta"
        )


def test_generate_periodic_signal_invalid_sampling_rate_value():
    """Test that the generate_periodic_signal   raises an error when a negative
    sampling rate is provided.

    GIVEN: A valid array of frequencies and duration, a negative
           sampling rate and default waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError,
        match="Sampling rate should be greater than or equal to 0",
    ):
        generate_periodic_signal(
            np.array([10, 20]), duration=2, sampling_rate=-250
        )


def test_generate_periodic_signal_empty_frequencies():
    """Test that the generate_periodic_signal returns an empty array when frequencies is
    an empty array.

    GIVEN: An empty array of frequencies, a valid signal duration and sampling_rate, default waveform_type.
    
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: signal is an empty array.
    """
    _, signal = generate_periodic_signal(
        np.array([]), duration=2, sampling_rate=350
    )
    assert np.all(
        signal == 0
    ), "Expected an array of zeros, but found nonzero values."


def test_generate_periodic_signal_invalid_waveform_type():
    """Test that the generate_periodic_signal raises a TypeError when a non- string
    waveform_type is provided.

    GIVEN: A valid array of frequencies, duration, and sampling rate,
           a non-string waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    with pytest.raises(
        TypeError,
        match="Waveform type should be a string, sin or square.",
    ):
        generate_periodic_signal(
            np.array([10, 20]),
            duration=2,
            sampling_rate=500,
            waveform_type=180,
        )


def test_generate_periodic_signal_invalid_waveform_value():
    """Test that the generate_periodic_signal raises a ValueError when an invalid
    waveform_type is provided.

    GIVEN: A valid array of frequencies, duration, and sampling rate,
           an invalid waveform_type.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    with pytest.raises(
        ValueError,
        match="Waveform type should be sin or square.",
    ):
        generate_periodic_signal(
            np.array([10, 20]),
            duration=2,
            sampling_rate=500,
            waveform_type="triangular",
        )


def test_generate_periodic_signal_signal_square_length():
    """Test that generate_periodic_signal     returns a signal with length equal to
    duration * sampling_rate when selecting square waves.

    GIVEN: A valid array of frequencies, signal duration, and
           sampling rate, and waveform_type = square.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The length of the signal is correct (duration * sampling_rate).
    """

    _, signal = generate_periodic_signal(
        np.array([2, 10]),
        duration=2,
        sampling_rate=250,
        waveform_type="square",
    )
    expected_length = 2 * 250  # duration * sampling rate
    assert (
        len(signal) == expected_length
    ), f"Signal length should be {expected_length}, but got {len(signal)}"


def test_generate_periodic_signal_signal_square_amplitude():
    """Test that the signal has non-zero amplitude when selceting square waves.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate, and waveform_type = square.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The signal has non-zero amplitude.
    """

    _, signal = generate_periodic_signal(
        np.array([2, 10]),
        duration=1,
        sampling_rate=250,
        waveform_type="square",
    )
    assert np.any(signal != 0), "Signal has zero amplitude"


def test_generate_periodic_signal_square_signal_is_ndarray():
    """Test that the 'signal' variable is of type numpy.ndarray when selceting square
    waves.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate, and waveform_type = square.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The 'signal' variable is of type numpy.ndarray.
    """
    _, signal = generate_periodic_signal(
        np.array([2, 10]),
        duration=1,
        sampling_rate=250,
        waveform_type="square",
    )
    assert isinstance(
        signal, np.ndarray
    ), f"Signal should be numpy.ndarray but got {type(signal)}"


def test_generate_periodic_signal_square_time_is_ndarray():
    """Test that the 'time' variable is of type numpy.ndarray when selceting square
    waves.

    GIVEN: A valid array of frequencies, signal duration,
           sampling rate and waveform_type = square.
           
    WHEN: The generate_periodic_signal function is called with these parameters.
    
    THEN: The 'time' variable is of type numpy.ndarray.
    """
    time, _ = generate_periodic_signal(
        np.array([2, 10]),
        duration=1,
        sampling_rate=250,
        waveform_type="square",
    )
    assert isinstance(
        time, np.ndarray
    ), f"Time should be numpy.ndarray but got {type(time)}"


# Test for compute_signal_power


def test_compute_signal_power_type():
    """Test that compute_signal_power returns the correct output type when given a valid
    input.

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
    """Test that compute_signal_power raises a TypeError when provided with an invalid
    input type.

    GIVEN: An invalid input (e.g., a string) instead of a numpy array representing a signal.
    
    WHEN: The compute_signal_power function is called with this parameter.
    
    THEN: A TypeError is raised.
    """
    with pytest.raises(TypeError):
        compute_signal_power(np.array(["signal", True]))


def test_compute_signal_power_single_element():
    """Test that compute_signal_power returns the correct power for a single- element
    signal.

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
    """Test that compute_signal_power raises a TypeError when provided with an invalid
    element in signal array.

    GIVEN: A numpy array containing an invalid type.
    
    WHEN: The compute_signal_power function is called with this parameter.
    
    THEN: A TypeError is raised.
    """
    signal = np.array([1, 2, 3, None, 4, 5])
    with pytest.raises(TypeError):
        compute_signal_power(signal)


def test_compute_signal_power_empty_signal():
    """Test that compute_signal_power raises a ValueError when provided with an empty
    array.

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
    """Test that compute_white_noise_std raises a TypeError when provided with an
    invalid SNR type.

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


# Tests for generate_white_noise


def test_generate_white_noise_length():
    """Test that generate_white_noise generates noise of the same length as the input
    signal.

    GIVEN: A valid SNR a valid input signal and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The return value has the same length as the input signal.
    """
    random_seed = 42
    signal = np.linspace(-20, 10, 80)
    snr_db = 10
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert len(noise) == len(
        signal
    ), f"Expected noise length {len(signal)}, but got {len(noise)}"


def test_generate_white_noise_zero_mean():
    """Test that generate_white_noise generates noise with zero mean within a tolerance.

    GIVEN: A valid SNR, a valid input signal and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The return value has a mean of 0 within the tolerance of one standard deviation of the noise.
    """
    random_seed = 42
    signal = np.linspace(-10, 10, 50)
    snr_db = 10
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert np.isclose(
        np.mean(noise), 0, atol=np.std(noise)
    ), f"Expected noise mean to be 0 +/- one std, but got {[np.mean(noise)]}"


def test_generate_white_noise_noise_std():
    """Test the standard deviation of noise generated by generate_white_noise.

    GIVEN: A valid signal, a specified signal-to-noise ratio (SNR) in dB and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The standard deviation of the generated noise matches the expected value
          within a small numerical tolerance.
    """
    random_seed = 42
    signal = np.linspace(-10, 10, 40)
    snr_db = 10
    noise = generate_white_noise(signal, snr_db, random_seed)
    expected_std = compute_white_noise_std(signal, snr_db)

    assert np.isclose(
        np.std(noise), expected_std, atol=0.1
    ), f"Expected {[expected_std]} noise standard deviation, but got {[np.std(noise)]}"


def test_generate_white_noise_scalar_signal():
    """Test that generate_white_noise generates noise for a scalar signal.

    GIVEN: A valid SNR, a scalar signal (single value) and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The return value is an array of length 1.
    """
    random_seed = 42
    signal = 1  # Scalar value
    snr_db = 8
    noise = generate_white_noise(signal, snr_db, random_seed)
    assert (
        len(noise) == 1
    ), f"Expected noise length to be 1, but got {len(noise)}"


def test_generate_white_noise_scalar_signal_float():
    """Test that generate_white_noise works when the signal is a single floating-point
    number.

    GIVEN: A scalar float as signal and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The return value is an array of length 1.
    """
    random_seed = 42
    signal = 3.5
    snr_db = 12
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert (
        len(noise) == 1
    ), f"Expected noise length to be 1, but got {len(noise)}"


def test_generate_white_noise_high_snr():
    """Test that generate_white_noise returns near-zero noise for a very high SNR.

    GIVEN: A very high signal-to-noise ratio (SNR), a valid input signal and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The generated noise should be close to zero for all elements.
    """
    random_seed = 42
    signal = np.linspace(-10, 15, 60)
    snr_db = 1000  # Very high SNR
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert np.all(np.isclose(noise, 0, atol=1e-6)), (
        "Expected all noise values to be close to zero for high SNR, "
        f"but got {noise}"
    )


def test_generate_white_noise_constant_signal():
    """Test that generate_white_noise generates noise even when the input signal is
    constant.

    GIVEN: A constant signal, a valid SNR and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The generated noise is not all zeros.
    """
    random_seed = 42
    signal = np.ones(100)
    snr_db = 15
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert not np.all(
        noise == 0
    ), "Generated noise should not be all zeros for a constant signal"


def test_generate_white_noise_negative_snr():
    """Test that generate_white_noise generates high-amplitude noise for negative SNR.

    GIVEN: A valid signal, a negative SNR and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The noise standard deviation is larger than the signal's standard deviation.
    """
    random_seed = 42
    signal = np.linspace(-5, 5, 50)
    snr_db = -20
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert np.std(noise) > np.std(
        signal
    ), f"Expected noise std {np.std(noise)} to be greater than signal std {np.std(signal)} for negative SNR."


def test_generate_white_noise_zero_signal():
    """Test that generate_white_noise returns all zeros when the input signal is all
    zeros.

    GIVEN: A zero signal, a valid SNR and fixed seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: The generated noise is an array of zeros.
    """
    random_seed = 42
    signal = np.zeros(50)
    snr_db = 10
    noise = generate_white_noise(signal, snr_db, random_seed)

    assert np.all(
        noise == 0
    ), "Expected noise to be all zeros when signal is all zeros"
    

def test_generate_white_noise_invalid_seed_type():
    """Test that the generate_white_noise raises an error when a non-integer 
    seed is provided.

    GIVEN: A valid signal and SNR, floating seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: A TypeError is raised.
    """
    random_seed = 7.5
    signal = np.ones(50)
    snr_db = 10

    with pytest.raises(TypeError):
        generate_white_noise(signal, snr_db, random_seed)
        
def test_generate_white_noise_invalid_seed_value():
    """Test that the generate_white_noise raises an error when a negative
    seed is provided.

    GIVEN: A valid signal and SNR, negative seed.
    
    WHEN: The generate_white_noise function is called with these parameters.
    
    THEN: A ValueError is raised.
    """
    random_seed = -100
    signal = np.ones(50)
    snr_db = 10

    with pytest.raises(ValueError):
        generate_white_noise(signal, snr_db, random_seed)

def test_generate_white_noise_same_seed():
    """Test that generate_white_noise return the same array when called two times
        withe the same seed
        
    GIVEN: Valid signal, SNR and seed.
    
    WHEN: The generate_white_noise function is called two times with these parameters.
    
    THEN: The returned noise arrays are the same.
    """
    random_seed = 42
    signal = np.ones(50)
    snr_db = 10
    
    noise_1 = generate_white_noise(signal, snr_db, random_seed)
    noise_2 = generate_white_noise(signal, snr_db, random_seed)
    assert np.array_equal(noise_1, noise_2), f"Arrays are not equal: {noise_1} != {noise_2}"
    
# Tests for add_white_noise


def test_add_white_noise_noisy_signal_size():
    """Test that the noisy signal has the same length as the input signal.

    GIVEN: A valid input signal and SNR.
    
    WHEN: The add_white_noise function is called with these parameters.
    
    THEN: The output noisy signal is of type numpy.ndarray.
    """
    signal = np.linspace(-5, 25, 70)
    snr_db = 10
    noisy_signal = add_white_noise(signal, snr_db)
    assert len(noisy_signal) == len(signal), (
        f"Expected noisy signal length {len(signal)}, "
        f"but got {len(noisy_signal)}"
    )


def test_add_white_noise_noisy_signal_type():
    """Test that the noisy signal has the same type as the input signal.

    GIVEN: A valid input signal and SNR.
    
    WHEN: The add_white_noise function is called with  these parameters.
    
    THEN: The output noisy signal is of type numpy.ndarray.
    """
    signal = np.linspace(-5, 25, 70)
    snr_db = 10
    noisy_signal = add_white_noise(signal, snr_db)
    assert isinstance(noisy_signal, np.ndarray), "Noisy signal type mismatch"


def test_add_white_noise_scalar():
    """Test that the function handles scalar signals correctly.

    GIVEN: A scalar input signal and a valid SNR.
    
    WHEN: The add_white_noise function is called with  these parameters.
    
    THEN: The noisy signal has length 1.
    """
    signal = 5
    snr_db = 20
    noisy_signal = add_white_noise(signal, snr_db)
    assert (
        len(noisy_signal) == 1
    ), "Noisy signal length should be 1 for scalar input"


def test_add_white_noise_all_zeros():
    """Test that an all-zero input signal returns an all-zero noisy signal.

    GIVEN: An all-zero input signal and a valid SNR.
    
    WHEN: The add_white_noise function is called withthese parameters.
    
    THEN: The output noisy signal consists entirely of zeros.
    """
    random_seed = 42
    signal = np.zeros(10)
    snr_db = 20
    noisy_signal = add_white_noise(signal, snr_db, random_seed)
    assert np.all(
        noisy_signal == 0
    ), "Noisy signal should be all zeros when input signal is all zeros"


def test_add_white_noise_high_snr():
    """Test that a very high SNR returns the input signal, within a certain tolerance.

    GIVEN: A valid input signal and a very high SNR.
    
    WHEN: The add_white_noise function is called.
    
    THEN: The output noisy signal is nearly identical to the input signal.
    """
    random_seed = 42
    signal = signal = np.linspace(-5, 25, 90)
    snr_db = 1000
    noisy_signal = add_white_noise(signal, snr_db, random_seed)
    assert np.all(
        np.isclose(noisy_signal, signal, atol=0.1)
    ), "Noisy signal should be almost identical to input signal"


# Tests for compute_fft


def test_compute_fft_frequency_bins_length():
    """Test that the length of frequency_bins matches the length of the signal.

    GIVEN: Valid input signal and sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: The output frequency_bins array has the same length as the input signal.
    """
    sampling_rate = 250
    signal = np.ones(50)
    _, frequency_bins = compute_fft(signal, sampling_rate)
    assert len(frequency_bins) == len(
        signal
    ), "Expected frequency_bins to be an array of length {len(signal)}, but got {len(frequency_bins)} "


def test_compute_fft_coefficients_length():
    """Test that the length of fft_coefficients matches the length of the signal.

    GIVEN: Valid input signal and sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: The output fft_coefficients array has the same length as the input signal.
    """
    sampling_rate = 350
    signal = np.ones(50)
    fft_coefficients, _ = compute_fft(signal, sampling_rate)
    assert len(fft_coefficients) == len(
        signal
    ), "Expected fft_coefficients to be an array of length {len(signal)}, but got {len(fft_coefficients)} "


def test_compute_fft_invalid_sampling_rate_type():
    """Test that the compute_fft raises an error when a not float or integer sampling
    rate is provided.

    GIVEN: Valid input signal, a non float or integer sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    signal = np.ones(50)
    with pytest.raises(
        TypeError,
        match="Sampling rate should be a number, either integer or float",
    ):
        compute_fft(signal, sampling_rate="duecentocinquanta")


def test_compute_fft_invalid_sampling_rate_value():
    """Test that the compute_fft raises an error when a negative sampling rate is
    provided.

    GIVEN: Valid input signal, a negative sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    signal = np.ones(50)
    with pytest.raises(
        ValueError,
        match="Sampling rate should be greater than or equal to 0",
    ):
        compute_fft(signal, sampling_rate=-250)


def test_compute_fft_invalid_signal_type():
    """Test that the compute_fft raises an error when a not integer or float element is
    provided in the signal array.

    GIVEN: An array containing at least a non-integer or float value, valid sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    signal = np.array([1, 2, 3, 4, "cinque", 6])
    with pytest.raises(
        TypeError,
        match="Signal should be an array of numbers, either integer or float",
    ):
        compute_fft(signal, 350)


def test_compute_fft_empty_signal():
    """Test that the compute_fft raises an error when an empty signal is provided.

    GIVEN: An empty array, valid sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    signal = np.array([])
    with pytest.raises(ValueError):
        compute_fft(signal, 350)


def test_compute_fft_zero_signal():
    """Test that a zero signal returns a zero FFT.

    GIVEN: A signal consisting of all zeros and a valid sampling rate.
    
    WHEN: The compute_fft function is called with these parameters.
    
    THEN: The output fft_coefficients array should contain only zeros.
    """
    signal = np.zeros(50)
    fft_coefficients, _ = compute_fft(signal, 300)
    assert np.allclose(
        fft_coefficients, 0
    ), f"Expected all zero FFT, but got {fft_coefficients}"


# Tests for compute_iift


def test_compute_ifft_complex_output():
    """Test that the output of compute_ifft is real-valued.

    GIVEN: A valid complex spectrum.
    
    WHEN: The compute_ifft function is called with this parameter.
    
    THEN: The output is complex-valued.
    """
    spectrum = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    output = compute_ifft(spectrum)
    assert np.iscomplexobj(output), "Output should be complex-valued."


def test_compute_ifft_length():
    """Test that the length of the output matches the input spectrum length.

    GIVEN: A valid complex spectrum.
    
    WHEN: The compute_ifft function is called with this parameter.
    
    THEN: The output signal has the same length as the input spectrum.
    """
    spectrum = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    output = compute_ifft(spectrum)
    assert len(output) == len(
        spectrum
    ), "Output length does not match input length."


def test_compute_ifft_invalid_spectrum_type():
    """Test that the compute_ifft raises a TypeError when an invalid spectrum type is
    provided.

    GIVEN: An invalid spectrum type (e.g. string instead of a complex numpy array).
    
    WHEN: The compute_ifft function is called with this parameter.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    spectrum = "invalid_input"
    with pytest.raises(
        TypeError, match="Input spectrum must be a NumPy array."
    ):
        compute_ifft(spectrum)


def test_compute_ifft_invalid_spectrum_value():
    """Test that the compute_ifft raises a TypeError when an invalid spectrum type is
    provided.

    GIVEN: An invalid spectrum type (e.g. string instead of a complex numpy array).
    
    WHEN: The compute_ifft function is called with this parameter.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    spectrum = np.array([1, 3, 5, 7, 9])
    with pytest.raises(
        ValueError,
        match="Input spectrum must be a complex-valued NumPy array.",
    ):
        compute_ifft(spectrum)


def test_compute_ifft_empty_spectrum():
    """Test that the compute_ifft raises an error when an empty spectrum is provided.

    GIVEN: An empty array representing the spectrum.
    
    WHEN: The compute_ifft function is called with this parameter.
    
    THEN: A ValueError is raised.
    """
    spectrum = np.array([])
    with pytest.raises(ValueError):
        compute_ifft(spectrum)


def test_compute_ifft_zero_spectrum():
    """Test that a zero spectrum returns a zero signal.

    GIVEN: A spectrum consisting of all zeros (complex values).
    
    WHEN: The compute_ifft function is called with this parameter.
    
    THEN: The output signal should be an array of zeros.
    """
    spectrum = np.array([0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
    signal = compute_ifft(spectrum)
    assert np.allclose(signal, 0), f"Expected all zeros, but got {signal}"


# Tests for apply spectral slope


def test_apply_spectral_slope_output_size():
    """Test that apply_spectral_slope returns an output of the correct size.

    GIVEN: A valid input signal and a positive slope, default sampling rate.
    
    WHEN: The apply_spectral_slope function is called with these parameters.
    
    THEN: The output spectrum has the same shape as the input signal.
    """
    signal = np.ones(100)
    slope = 1.0
    modified_spectrum = apply_spectral_slope(signal, slope)
    assert (
        modified_spectrum.shape == signal.shape
    ), "Output size should match input size"


def test_apply_spectral_slope_zero_slope():
    """Test that apply_spectral_slope returns the original spectrum when slope is zero.

    GIVEN: A valid input signal, slope = 0 and default sampling_rate.
    
    WHEN: The apply_spectral_slope function is called with the parameters.
    
    THEN: The output spectrum is identical to the original FFT of the signal.
    """
    signal = np.ones(100)
    slope = 0.0
    fft_original, _ = compute_fft(signal)
    modified_spectrum = apply_spectral_slope(signal, slope)
    np.testing.assert_array_almost_equal(
        modified_spectrum,
        fft_original,
        err_msg="With slope=0, output should match original spectrum",
    )


def test_apply_spectral_slope_invalid_slope_type():
    """Test that apply_spectral_slope raises a TypeError for invalid slope types.

    GIVEN: A valid input signal, an invalid slope (non-numeric type) and a valid sampling rate.
    
    WHEN: The apply_spectral_slope function is called with these parameters.
    
    THEN: A TypeError is raised with the appropriate message.
    """
    signal = np.array([1, 2, 3, 4, 5, 6])
    with pytest.raises(
        TypeError, match="Slope should be a number, either integer or float"
    ):
        apply_spectral_slope(signal, "one", 350)


def test_apply_spectral_slope_negative_slope():
    """Test that apply_spectral_slope raises a ValueError when slope is negative.

    GIVEN: A valid input signal, a negative slope and a valid sampling rate.
    
    WHEN: The apply_spectral_slope function is called with these parameters.
    
    THEN: A ValueError is raised with the appropriate message.
    """
    signal = np.array([1, 2, 3, 4, 5, 6])
    with pytest.raises(
        ValueError, match="Slope should be greater than or equal to 0"
    ):
        apply_spectral_slope(signal, -1.0, 500)


# Tests for add_colored_noise


def test_add_colored_noise_output_shape():
    """Test that add_colored_noise output has the correct shape.

    GIVEN: A valid input signal, reasonable snr_db, slope, and sampling_rate.
    
    WHEN: The add_colored_noise function is called with these parameters.
    
    THEN: The output has the same shape as the input signal.
    """
    signal = np.ones(100)
    snr_db = 10.0
    slope = 1.0
    sampling_rate = 250

    output = add_colored_noise(signal, snr_db, slope, sampling_rate)

    assert (
        output.shape == signal.shape
    ), "Output should have the same shape as input signal"


def test_add_colored_noise_zero_signal():
    """Test that add_colored_noise correctly generates only colored noise when the input
    is zero.

    GIVEN: An input signal of all zeros, valid snr, slope and sampling_rate.
    
    WHEN: The add_colored_noise function is called with these parameters.
    
    THEN: The output is not all zeros.
    """
    signal = np.zeros(100)
    snr_db = 10.0
    slope = 1.0
    sampling_rate = 250

    output = add_colored_noise(signal, snr_db, slope, sampling_rate)

    assert not np.allclose(
        output, signal
    ), "Output should not be all zeros when input is zero"


def test_add_colored_noise_low_snr():
    """Test that add_colored_noise with a low SNR results in a noise-dominated output.

    GIVEN: A valid input signal, a very low SNR value, valid slope and sampling rate.
    
    WHEN: The add_colored_noise function is called with these parameters.
    
    THEN: The noise power is greater than the signal power.
    """
    signal = np.ones(100)
    snr_db = -10.0
    slope = 1.0
    sampling_rate = 250

    output = add_colored_noise(signal, snr_db, slope, sampling_rate)
    noise_only = output - signal

    assert np.linalg.norm(np.abs(noise_only)) > np.linalg.norm(
        signal
    ), "Noise should dominate at very low SNR"


def test_add_colored_noise_slope_effect():
    """Test that changing the slope affects the noise spectrum.

    GIVEN: A valid input signal, different slope values, valid SNR and sampling_rate.
    
    WHEN: The add_colored_noise function is called two times with different slopes.
    
    THEN: The output noise is different for different slopes.
    """
    signal = np.ones(100)
    snr_db = 10.0
    sampling_rate = 250

    output_slope_1 = add_colored_noise(
        signal, snr_db, slope=1.0, sampling_rate=sampling_rate
    )
    output_slope_2 = add_colored_noise(
        signal, snr_db, slope=2.0, sampling_rate=sampling_rate
    )

    assert not np.allclose(
        output_slope_1, output_slope_2
    ), "Different slopes should generate different noise patterns"

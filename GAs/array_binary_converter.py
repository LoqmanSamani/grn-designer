import numpy as np


def decimal_to_binary(array, precision_bits):

    """
    Convert a NumPy array to a binary string.

    Args:
        - array (numpy.ndarray): Input array of float or int values.
        - precision_bits (list of tuples): List of tuples containing (min_val, max_val, bits) for each element to define the precision.
            - min_val (float): The minimum possible value of the original range for the element.
            - max_val (float): The maximum possible value of the original range for the element.
            - bits (int): The number of bits used to represent the element in the binary string.
    Returns:
        - binary_string (str): Binary string representing the array.
    """

    binary_string = ''.join(
        f'{int((val - min_val) / (max_val - min_val) * ((1 << bits) - 1)):0{bits}b}'
        for val, (min_val, max_val, bits) in zip(array, precision_bits)
    )

    return binary_string


def binary_to_decimal(binary_string, precision_bits):

    """
    Convert a binary string back to a NumPy array.

    Args:
        - binary_string (str): Binary string to be converted.
        - precision_bits (list of tuples): List of tuples containing (min_val, max_val, bits) for each element to define the precision.
            - min_val (float): The minimum possible value of the original range for the element.
            - max_val (float): The maximum possible value of the original range for the element.
            - bits (int): The number of bits used to represent the element in the binary string.

    Returns:
        - arr (numpy.ndarray): Array of float or int values represented by the binary string.
    """

    array = []
    index = 0

    for min_val, max_val, bits in precision_bits:
        segment = binary_string[index:index + bits]
        int_value = int(segment, 2)
        max_int_value = (1 << bits) - 1
        real_value = min_val + (max_val - min_val) * (int_value / max_int_value)
        array.append(real_value)
        index += bits

    return np.array(array)





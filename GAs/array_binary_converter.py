import numpy as np




def decimal_to_binary(array_list, precision_bits_list):

    """
    Convert a list of 2D NumPy arrays to a list of binary strings.

    Args:
        - array_list (list of numpy.ndarray): List of input 2D arrays of float or int values.
        - precision_bits_list (list of tuple): List of tuples containing (min_val, max_val, bits) to define the precision for each array.
            - min_val (float): The minimum possible value of the original range for the element.
            - max_val (float): The maximum possible value of the original range for the element.
            - bits (int): The number of bits used to represent the element in the binary string.

    Returns:
        - binary_strings (list of str): List of binary strings representing the arrays.
    """

    binary_strings = []

    for array, precision_bits in zip(array_list, precision_bits_list):

        min_val, max_val, bits = precision_bits

        binary_string = ''.join(
            f"{int((val - min_val) / (max_val - min_val) * ((1 << bits) - 1)):0{bits}b}"
            for sub_array in array for val in sub_array
        )

        binary_strings.append(binary_string)

    return binary_strings




def binary_to_decimal(binary_string_list, precision_bits_list, shapes):
    """
    Convert a list of binary strings back to a list of 2D NumPy arrays.

    Args:
        - binary_string_list (list of str): List of binary strings to be converted.
        - precision_bits_list (list of tuple): List of tuples containing (min_val, max_val, bits) to define the precision for each array.
            - min_val (float): The minimum possible value of the original range for the element.
            - max_val (float): The maximum possible value of the original range for the element.
            - bits (int): The number of bits used to represent the element in the binary string.
        - shapes (list of tuple): List of shapes for the original arrays.

    Returns:
        - arrays (list of numpy.ndarray): List of arrays of float or int values represented by the binary strings.
    """

    arrays = []

    for binary_string, precision_bits, shape in zip(binary_string_list, precision_bits_list, shapes):

        min_val, max_val, bits = precision_bits
        array = []
        index = 0

        for _ in range(shape[0]):

            row = []

            for _ in range(shape[1]):

                segment = binary_string[index:index + bits]
                int_value = int(segment, 2)
                max_int_value = (1 << bits) - 1
                real_value = min_val + (max_val - min_val) * (int_value / max_int_value)
                row.append(real_value)
                index += bits
            array.append(row)

        arrays.append(np.array(array))

    return arrays




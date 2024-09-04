import tensorflow as tf
import numpy as np
import math


class PoolingLayers:

    def __init__(self, target, pooling_method, pool_size, strides, padding, zero_padding, kernel_size, up_padding, up_strides):

        self.target = target
        self.pooling_method = pooling_method
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.zero_padding = zero_padding
        self.kernel_size = kernel_size
        self.up_padding = up_padding
        self.up_strides = up_strides
        """
        A class to perform pooling and up sampling operations on 2D matrices using TensorFlow.

        This class provides methods for both down sampling (pooling) and up sampling operations on 2D matrices.
        The down sampling can be performed using either max pooling or average pooling. The up sampling is done
        using transposed convolutions.

        Attributes:
            - target (np.ndarray): The target matrix to be used for up sampling.
            - pool_size (tuple of int): The size of the pooling window (height, width).
            - strides (tuple of int): The strides of the pooling operation (height, width).
            - padding (str): Padding mode for the pooling operation. Either 'valid' or 'same'.
            - zero_padding (tuple of int): The amount of zero padding to apply before pooling (height, width).
            - kernel_size (tuple of int): The size of the kernel for the transposed convolution (height, width).
            - up_padding (str): Padding mode for the up sampling operation. Either 'valid' or 'same'.
            - up_strides (tuple of int): The strides of the up sampling operation (height, width).

        Methods:
    
            - pooling(compartment, method): Performs pooling or up sampling based on the specified method.
            - up_sample_population(population, target, kernel_size, padding, strides): Applies transposed convolution to up sample a population of matrices.
            - down_sample_target(target, pooling_method, zero_padding, pool_size, strides, padding): Applies pooling to down sample the target matrix.

        Formulas:
            - Pooling Formula:
              Given:
                 - Input shape: (H_in, W_in)
                 - Pool size: (p_h, p_w)
                 - Strides: (s_h, s_w)
                 - Padding: pad_h, pad_w

              Output shape after pooling:
                 - H_out = floor((H_in + 2 * pad_h - p_h) / s_h) + 1
                 - W_out = floor((W_in + 2 * pad_w - p_w) / s_w) + 1

            - Padding:
                 - 'valid': pad_h = 0, pad_w = 0
                 - 'same': pad_h = floor(((H_in * (s_h - 1) - s_h + p_h) / 2))
              
                   pad_w = floor(((W_in * (s_w - 1) - s_w + p_w) / 2))

            - Up sampling Formula (using Conv2DTranspose):
              Given:
                 - Output shape after pooling: (H_out, W_out)
                 - Kernel size: (k_h, k_w)
                 - Strides: (s_h, s_w)
                 - Padding: pad_h, pad_w

              Output shape after up sampling:
                 - H_up = s_h * (H_out - 1) + k_h - 2 * pad_h
                 - W_up = s_w * (W_out - 1) + k_w - 2 * pad_w
        """


    def pooling(self, compartment, method):
        """
        Performs pooling or up sampling on the given compartment matrix.

        Parameters:
            - compartment (np.ndarray or a list of np.ndarray): The matrix to be processed (either pooled or up sampled).
            - method (str): The method to use: "target pooling" for pooling, or "population up sampling" for up sampling.

        Returns:

            - np.ndarray:  The processed matrix/list after applying the specified method.
        """

        if method == "target pooling":
            compartment = self.down_sample_target(
                target=compartment,
                pooling_method= self.pooling_method,
                zero_padding=self.zero_padding,
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding
            )

        elif method == "population up sampling":
            compartment = self.up_sample_population(
                population=compartment,
                target=self.target,
                kernel_size=self.kernel_size,
                padding=self.up_padding,
                strides=self.up_strides
            )

        return compartment



    def up_sample_population(self, population, target, kernel_size, padding, strides):
        """
        Applies transposed convolution to up sample a population of matrices.

        Parameters:
            - population (list of np.ndarray): The list population to be up sampled.
            - target (np.ndarray): The target matrix used to determine the shape after up sampling.
            - kernel_size (tuple of int): The size of the kernel for the transposed convolution (height, width).
            - padding (str): Padding mode for the up sampling operation.
            - strides (tuple of int): The strides of the up sampling operation (height, width).

        Returns:

            - list of np.ndarray: A list of up sampled matrices.
        """

        y_init, x_init = target.shape
        up_population = []

        for individual in population:

            z, y, x = individual.shape
            num_species = int(individual[-1, -1, 0])
            num_pairs = int(individual[-1, -1, 1])
            pair_start = int(num_species * 2)
            pair_stop = int(pair_start + (num_pairs * 2))
            up_individual = np.zeros(shape=(z, y_init, x_init))

            for i in range(1, num_species * 2, 2):

                up_individual[i, :, :] = tf.nn.relu(tf.keras.layers.Conv2DTranspose(
                    filters=1,  # since we want to up sample a 2D matrix
                    kernel_size=kernel_size,
                    padding=padding,
                    strides=strides
                )(tf.expand_dims(tf.expand_dims(individual[i, :, :], axis=0), axis=-1) # add batch and channel dimensions
                )).numpy()[0, :, :, 0]  # remove the batch and channel dimensions

                up_individual[-1, i-1, 0:3] = individual[-1, i-1, 0:3]

            for i in range(pair_start+1, pair_stop+1, 2):

                up_individual[i, 0, :2] = individual[i, 0, :2]
                up_individual[i, 1, :4] = individual[i, 1, :4]

            up_individual[-1, -1, :5] = individual[-1, -1, :5]
            up_population.append(up_individual)

        return up_population




    def down_sample_target(self, target, pooling_method, zero_padding, pool_size, strides, padding):
        """
        Applies pooling to down sample the target matrix.

        Parameters:

            - target (np.ndarray): The matrix to be down sampled.
            - zero_padding (tuple of int): The amount of zero padding to apply before pooling (height, width).
            - pool_size (tuple of int): The size of the pooling window (height, width).
            - strides (tuple of int): The strides of the pooling operation (height, width).
            - padding (str): Padding mode for the pooling operation.

        Returns:

            - np.ndarray: The down sampled matrix.
        """
        # Expand dimensions to add batch and channel axes
        target_tensor = tf.convert_to_tensor(value=target, dtype=tf.float32)
        target_tensor = tf.expand_dims(tf.expand_dims(target_tensor, axis=0), axis=-1)  # Shape: (1, height, width, 1)

        if zero_padding[0] > 0 or zero_padding[1] > 0:
            target_tensor = tf.keras.layers.ZeroPadding2D(
                padding=zero_padding
            )(target_tensor)

        if pooling_method == "max":
            target_tensor = tf.nn.relu(tf.keras.layers.MaxPool2D(
                pool_size=pool_size,
                strides=strides,
                padding=padding
            )(target_tensor))
        elif pooling_method == "average":
            target_tensor = tf.nn.relu(tf.keras.layers.AveragePooling2D(
                pool_size=pool_size,
                strides=strides,
                padding=padding
            )(target_tensor))

        # Remove the added batch and channel dimensions
        target_array = tf.squeeze(target_tensor, axis=[0, -1]).numpy()

        return target_array



    def calculate_conv2d_transpose_parameters(self, input_shape, output_shape, padding_mode, strides):
        """
        Calculates kernel_size, strides, and padding needed for Conv2DTranspose so that the output shape matches the target shape.

        Parameters:
            - input_shape (tuple of int): Shape of the matrix before up sampling (height, width).
            - output_shape (tuple of int): Desired shape of the matrix after up sampling (height, width).
            - padding_mode (str): Padding mode used in previous down sampling ('valid' or 'same').
            - strides (tuple of int): Strides used in previous down sampling (height_stride, width_stride).

        Returns:
            - tuple: A tuple containing (kernel_size, strides, padding_mode) suitable for Conv2DTranspose.
        """
        H_in, W_in = input_shape
        H_out, W_out = output_shape
        s_h, s_w = strides

        # Calculate padding for Conv2DTranspose based on desired output shape
        if padding_mode == 'valid':
            # No padding during down sampling
            pad_h = pad_w = 0
        elif padding_mode == 'same':
            # For 'same' padding, calculate padding for both sides
            pad_h = math.ceil((s_h * (H_in - 1) + 1 - H_out) / 2)
            pad_w = math.ceil((s_w * (W_in - 1) + 1 - W_out) / 2)
        else:
            raise ValueError("Unsupported padding type. Use 'valid' or 'same'.")

        # Calculate the kernel size
        k_h = (H_out - 1) * s_h - H_in + 2 * pad_h + 1
        k_w = (W_out - 1) * s_w - W_in + 2 * pad_w + 1

        # Ensure kernel size is positive
        k_h = max(1, k_h)
        k_w = max(1, k_w)

        # Adjust padding if necessary
        if padding_mode == 'same':
            pad_h = (s_h * (H_in - 1) + k_h - H_out) // 2
            pad_w = (s_w * (W_in - 1) + k_w - W_out) // 2

        kernel_size = (k_h, k_w)
        strides = (s_h, s_w)

        return kernel_size, strides, padding_mode









import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import seaborn as sns

# Example 2D matrix (downsampled)
downsampled_matrix = np.array([[4, 9, 9, 0, 9, 7], [3,2,1,8, 4, 6], [6,8,9,2, 4, 6]])

# Upsample by a factor of 2 using bilinear interpolation
upsampled_matrix = zoom(downsampled_matrix, zoom=2, order=1)  # order=1 corresponds to bilinear interpolation

print(upsampled_matrix)


# Example 2D matrix
matrix = np.array([[4, 9, 9], [8, 4, 6], [2, 4, 6]])

# Downsample by a factor of 0.5 (reduce size) using bilinear interpolation
downsampled_matrix1 = zoom(upsampled_matrix, zoom=0.5, order=1)  # order=1 for bilinear interpolation

print(downsampled_matrix1)

sns.heatmap(downsampled_matrix)
plt.show()


sns.heatmap(upsampled_matrix)
plt.show()

sns.heatmap(downsampled_matrix1)
plt.show()






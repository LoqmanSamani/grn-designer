import numpy as np
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as ssim





def compute_cost(predictions, target, alpha, beta, max_val):
    """
    Compute the cost between predicted and target matrices based on the chosen method.

    Parameters:
    - predictions (np.ndarray): An array of predicted matrices with shape (m, y, x).
    - target (np.ndarray): A target matrix with shape (y, x).
    - delta_D (list of float): A list of maximum concentration changes for each prediction.
    - alpha (float): Error concentration threshold.
    - beta (float): Equilibrium penalty threshold.
    - kernel_size (int): Size of the box blur kernel.
    - method (str): Method for calculating the cost ('MSE', 'NCC', or 'GRM').

    Returns:
    - np.ndarray: An array of costs with shape (m,).
    """
    num_predictions, height, width = predictions.shape
    losses = np.zeros(num_predictions)

    for i in range(num_predictions):
        ssim_loss_value = ssim_loss_(
            y_hat=predictions[i],
            target=target,
            data_range=max_val)
        mse_loss_value = np.mean((target - predictions[i, :, :]) ** 2)
        losses[i] = alpha * mse_loss_value + beta * ssim_loss_value

    return losses


def ssim_loss_(y_hat, target, data_range):
    """
    Compute the Structural Similarity Index (SSIM) loss between two matrices.

    SSIM is used to measure the perceptual similarity between two images or matrices. A higher SSIM score indicates
    higher similarity. The SSIM loss is calculated as `1 - SSIM score`, so a lower SSIM loss indicates more perceptual
    similarity.

    Parameters:
            - y_hat (tf.Tensor): A 2D tensor representing the predicted matrix or image. Shape: (y, x).
            - target (tf.Tensor): A 2D tensor representing the target matrix or image. Shape: (y, x).
            - max_val (float, optional): The dynamic range of the input values, typically the maximum value of the pixel
              intensity. Default is 1.0.

    Returns:
            - float: The SSIM loss, computed as `1 - SSIM score`, where the SSIM score is between 0 and 1. A lower
              loss indicates more perceptual similarity between `y_hat` and `target`.
    """
    ssim_score, _ = ssim(y_hat, target, full=True, data_range=data_range)

    return 1 - ssim_score



import numpy as np
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity as ssim





def compute_cost(predictions, target, delta_D, alpha, beta, max_val, kernel_size, method):
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
    costs = np.zeros(num_predictions)

    if method == "MSE":
        costs = compute_combined_loss(
            predictions=predictions,
            target=target,
            alpha=alpha,
            beta=beta,
            ssim_max_val=max_val
        )

    elif method == "NCC":
        costs = compute_normalized_cross_correlation(
            predictions=predictions,
            target=target
        )

    elif method == "GRM":
        costs = compute_grm_fitness_error(
            predictions=predictions,
            target=target,
            kernel_size=kernel_size,
            alpha=alpha,
            beta=beta,
            delta_D=delta_D
        )

    return costs






# Combined loss calculation using NumPy
def compute_combined_loss(predictions, target, alpha, beta, ssim_max_val):
    """
    Compute a combined loss between Mean Squared Error (MSE) and Structural Similarity Index (SSIM) for each prediction.

    This loss function computes the combined loss for each predicted matrix in a batch against a single target matrix.
    The combined loss is a weighted sum of MSE and SSIM, where `alpha` controls the weight of the MSE, and `beta` controls
    the weight of the SSIM.

    Parameters:
        - predictions (np.ndarray): A 3D array of predicted matrices with shape (m, y, x), where `m` is the number of
          predicted matrices, and `y`, `x` are the dimensions of each matrix.
        - target (np.ndarray): A 2D target matrix with shape (y, x), representing the ground truth.
        - alpha (float, optional): The weight assigned to the MSE part of the combined loss.
        - beta (float, optional): The weight assigned to the SSIM part of the combined loss.
        - ssim_max_val (float, optional): The maximum value for SSIM calculations, representing the dynamic range of the input.

    Returns:
        - np.ndarray: A 1D array of loss values with shape (m,), where each value is the combined loss (MSE + SSIM) for
          each prediction in the batch.
    """
    num_predictions, height, width = predictions.shape
    losses = np.zeros(num_predictions)

    for i in range(num_predictions):
        # Compute SSIM loss
        ssim_loss_value = ssim_loss_(y_hat=predictions[i], target=target, data_range=ssim_max_val)
        # Compute MSE loss
        mse_loss_value = np.mean((target - predictions[i, :, :]) ** 2)
        # Calculate the combined loss
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
    # Compute SSIM score between y_hat and target
    ssim_score, _ = ssim(y_hat, target, full=True, data_range=data_range)
    # SSIM loss is 1 - SSIM score
    return 1 - ssim_score


def compute_grm_fitness_error(predictions, target, kernel_size, alpha, beta, delta_D):
    """
    Compute the GRM fitness error based on the blurred target and predicted patterns.

    Parameters:
    - predictions (np.ndarray): An array of predicted matrices with shape (m, y, x).
    - target (np.ndarray): A target matrix with shape (y, x).
    - kernel_size (int): Size of the box blur kernel.
    - alpha (float): Error concentration threshold.
    - beta (float): Equilibrium penalty threshold.
    - delta_D (list of float): A list of maximum concentration changes for each prediction.

    Returns:
    - np.ndarray: An array of GRM fitness errors with shape (m,).
    """
    num_predictions, height, width = predictions.shape
    kernel = create_box_blur_kernel(size=kernel_size)
    costs = np.zeros(num_predictions)

    blurred_target = convolve(target, kernel, mode='constant', cval=0.0)

    for i in range(num_predictions):
        blurred_prediction = convolve(predictions[i, :, :], kernel, mode='constant', cval=0.0)
        diff = np.abs(blurred_prediction - blurred_target)
        log_diff = np.log1p(np.maximum(diff - alpha, 0))
        log_diff_error = np.mean(log_diff)
        equilibrium_penalty = np.maximum(delta_D[i] - beta, 0)
        costs[i] = log_diff_error + equilibrium_penalty

    return costs



def compute_normalized_cross_correlation(predictions, target):
    """
    Compute the Normalized Cross-Correlation (NCC) between predicted and target matrices.

    Parameters:
    - predictions (np.ndarray): An array of predicted matrices with shape (m, y, x).
    - target (np.ndarray): A target matrix with shape (y, x).

    Returns:
    - np.ndarray: An array of NCC values with shape (m,).
    """
    num_predictions, height, width = predictions.shape
    costs = np.zeros(num_predictions)

    target_mean = np.mean(target)
    target_std = np.std(target)
    for i in range(num_predictions):
        pred_mean = np.mean(predictions[i, :, :])
        pred_std = np.std(predictions[i, :, :])

        if target_std > 0 and pred_std > 0:
            ncc = np.sum((target - target_mean) * (predictions[i, :, :] - pred_mean)) / (target_std * pred_std)
            ncc /= (height * width)
        else:
            ncc = 0

        costs[i] = ncc

    return costs


def create_box_blur_kernel(size):
    """
    Create a box blur kernel of given size.

    Parameters:
    - size (int): Size of the box blur kernel.

    Returns:
    - np.ndarray: A box blur kernel with shape (size, size).
    """
    return np.ones((size, size)) / (size * size)

import numpy as np
from scipy.ndimage import convolve


def compute_cost(predictions, target, delta_D, alpha, beta, kernel_size, method):
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
        costs = compute_mean_squared_error(
            predictions=predictions,
            target=target
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



def compute_mean_squared_error(predictions, target):
    """
    Compute the Mean Squared Error (MSE) between predicted and target matrices.

    Parameters:
    - predictions (np.ndarray): An array of predicted matrices with shape (m, y, x).
    - targets (np.ndarray): A target matrix with shape (y, x).

    Returns:
    - np.ndarray: An array of MSE values with shape (m,).
    """
    num_predictions, height, width = predictions.shape
    costs = np.zeros(num_predictions)

    for i in range(num_predictions):
        costs[i] = np.mean((target - predictions[i, :, :]) ** 2) #  / (height * width)
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




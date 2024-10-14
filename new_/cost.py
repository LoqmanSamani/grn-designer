import numpy as np
from skimage.metrics import structural_similarity as ssim





def compute_cost(predictions, target, alpha, beta, max_val):
    """
    Compute the total cost between predicted and target matrices using a combination of
    Mean Squared Error (MSE) and Structural Similarity Index (SSIM) loss.

    The cost function incorporates both MSE and SSIM, weighted by the parameters alpha and beta.
    A higher value of alpha emphasizes MSE, while a higher beta emphasizes SSIM loss.

    Parameters:
    - predictions (np.ndarray): An array of predicted matrices with shape (z, m, y, x),
      where 'z' is the number of patterns, 'm' is the number of predicted and (y, x) are the dimensions of each matrix.
    - target (np.ndarray): A target matrix or set of matrices with shape (z, y, x) to compare against.
    - alpha (float): The weight for MSE in the cost function.
    - beta (float): The weight for SSIM loss in the cost function.
    - max_val (float): The maximum possible value in the matrices, used for SSIM calculation.

    Returns:
    - np.ndarray: An array of costs with shape (m,), where each entry corresponds to the
      computed cost for a prediction compared to the target matrix.
    """
    num_patterns, num_predictions, height, width = predictions.shape
    losses = np.zeros(num_predictions)

    for i in range(num_predictions):
        ssim_loss_value = ssim_loss_(
            y_hat=predictions[:, i, :, :],
            target=target,
            num_patterns=num_patterns,
            data_range=max_val
        )
        mse_ = mean_squared_error(
            y_hat=predictions[:, i, :, :],
            target=target,
            num_patterns=num_patterns
        )

        losses[i] = alpha * mse_ + beta * ssim_loss_value

    return losses


def mean_squared_error(y_hat, target, num_patterns):
    """
    Calculate the Mean Squared Error (MSE) between the predicted and target matrices.

    MSE measures the average squared difference between the predicted and target values,
    providing a straightforward measure of prediction accuracy.

    Parameters:
    - y_hat (np.ndarray): The predicted matrix or set of matrices with shape (z, y, x),
      where 'z' is the number of patterns.
    - target (np.ndarray): The target matrix or set of matrices with shape (z, y, x).
    - num_patterns (int): The number of pattern matrices to compare (corresponding to the first dimension of y_hat and target).

    Returns:
    - float: The total MSE value summed over all patterns.
    """
    mse_ = 0.0
    for j in range(num_patterns):
        mse_loss_value = np.mean((target[j, :, :] - y_hat[j, :, :]) ** 2)
        mse_ += mse_loss_value
    return mse_


def ssim_loss_(y_hat, target, num_patterns, data_range):
    """
    Compute the Structural Similarity Index (SSIM) loss between predicted and target matrices.

    SSIM evaluates perceptual similarity between two matrices or images. It ranges from -1 to 1,
    where 1 indicates perfect similarity. The SSIM loss is computed as `1 - SSIM score`, so
    lower values of SSIM loss indicate higher perceptual similarity.

    Parameters:
    - y_hat (np.ndarray): The predicted matrix or set of matrices with shape (z, y, x).
    - target (np.ndarray): The target matrix or set of matrices with shape (z, y, x).
    - num_patterns (int): The number of pattern matrices to compare.
    - data_range (float): The dynamic range of the input values, typically the maximum
      possible value of pixel intensity (used in SSIM calculation).

    Returns:
    - float: The total SSIM loss, computed as `1 - SSIM score`, summed over all patterns.
    """
    ssim_ = 0.0
    for i in range(num_patterns):
        ssim_score, _ = ssim(y_hat[i, :, :], target[i, :, :], full=True, data_range=data_range)
        ssim_ += ssim_score

    return 1 - ssim_


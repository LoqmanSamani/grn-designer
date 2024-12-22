import numpy as np
from skimage.metrics import structural_similarity as ssim





def compute_cost(predictions, target, alpha, beta):
    """
    Computes the cost for a set of predictions against a target pattern.

    This function evaluates the cost of each prediction based on a weighted combination
    of Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM).

    Args:
        predictions (list): List of 2D numpy arrays representing predicted patterns.
        target (np.ndarray): 2D numpy array representing the target pattern.
        alpha (float): Weight for the MSE component in the cost calculation.
        beta (float): Weight for the SSIM component in the cost calculation.

    Returns:
        np.ndarray: Array of computed costs for each prediction.
    """

    num_predictions = len(predictions)
    losses = np.zeros(shape=num_predictions, dtype=np.float32)

    for i in range(num_predictions):

        ssim_loss_value = ssim_loss_(
            predicted=predictions[i],
            target=target
        )

        mse_ = mean_squared_error(
            predicted=predictions[i],
            target=target
        )

        losses[i] = alpha * mse_ + beta * ssim_loss_value

    return losses


def mean_squared_error(predicted, target):
    """
    Computes the Mean Squared Error (MSE) between a predicted and target pattern.

    Args:
        predicted (np.ndarray): 2D numpy array representing the predicted pattern.
        target (np.ndarray): 2D numpy array representing the target pattern.

    Returns:
        float: The computed MSE value.
    """

    mse_ = np.mean((target - predicted) ** 2)

    return mse_

def ssim_loss_(predicted, target):
    """
    Computes the Structural Similarity Index Measure (SSIM)-based loss.

    This function calculates the SSIM between a predicted and target pattern
    and returns the loss as `1 - SSIM`.

    Args:
        predicted (np.ndarray): 2D numpy array representing the predicted pattern.
        target (np.ndarray): 2D numpy array representing the target pattern.

    Returns:
        float: The SSIM loss value.
    """

    ssim_, _ = ssim(predicted, target, full=True, data_range=1.0)

    return 1 - ssim_



def weighted_prediction(prediction, pattern_proportion):
    """
    Generates a weighted average of predictions for pattern evaluation.

    If the input is a 3D array, each slice along the first dimension is weighted
    based on the provided pattern proportion, and the weighted sum is computed.
    For 2D arrays, the input is returned as-is.

    Args:
        prediction (np.ndarray): 2D or 3D numpy array representing predicted patterns.
                                 If 3D, the first dimension represents a batch of predictions.
        pattern_proportion (float): Weight for the first prediction in the batch. Other
                                     predictions are given equal weights.

    Returns:
        np.ndarray: A single weighted prediction pattern.
    """

    if prediction.ndim == 3:
        batch_size, height, width = prediction.shape

        weights = np.ones(batch_size)
        weights[0] = pattern_proportion
        weights = weights / weights.sum()
        norm_prediction = (prediction * weights[:, None, None]).sum(axis=0)
    else:
        norm_prediction = prediction

    return norm_prediction
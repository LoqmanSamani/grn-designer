import numpy as np
from skimage.metrics import structural_similarity as ssim





def compute_cost(predictions, target, alpha, beta):

    num_patterns, num_predictions, height, width = predictions.shape
    losses = np.zeros(num_predictions)

    for i in range(num_predictions):

        ssim_loss_value = ssim_loss_(
            y_hat=predictions[:, i, :, :],
            target=target,
            num_patterns=num_patterns
        )
        mse_ = mean_squared_error(
            y_hat=predictions[:, i, :, :],
            target=target,
            num_patterns=num_patterns
        )

        losses[i] = alpha * mse_ + beta * ssim_loss_value

    return losses


def mean_squared_error(y_hat, target, num_patterns):

    mse_ = 0.0
    for j in range(num_patterns):
        mse_loss_value = np.mean((target[j, :, :] - y_hat[j, :, :]) ** 2)
        mse_ += mse_loss_value
    return mse_


def ssim_loss_(y_hat, target, num_patterns):

    ssim_ = 0.0
    for i in range(num_patterns):
        ssim_score, _ = ssim(y_hat[i, :, :], target[i, :, :], full=True, data_range=1.0)
        ssim_ += ssim_score

    return 1 - ssim_


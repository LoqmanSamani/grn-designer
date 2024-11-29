import numpy as np
from skimage.metrics import structural_similarity as ssim





def compute_cost(predictions, target, alpha, beta):

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
    mse_ = np.mean((target - predicted) ** 2)

    return mse_

def ssim_loss_(predicted, target):
    ssim_, _ = ssim(predicted, target, full=True, data_range=1.0)

    return 1 - ssim_



def weighted_prediction(prediction, pattern_proportion):

    if prediction.ndim == 3:
        batch_size, height, width = prediction.shape

        weights = np.ones(batch_size)
        weights[0] = pattern_proportion
        weights = weights / weights.sum()
        norm_prediction = (prediction * weights[:, None, None]).sum(axis=0)
    else:
        norm_prediction = prediction

    return norm_prediction
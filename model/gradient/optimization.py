import tensorflow as tf


"""
def compute_cost(self, X, W, b, Y, R, lambda_):
    cost_ = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    cost = 0.5 * tf.reduce_sum(cost_ ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))

    return cost


optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        for i in range(epochs):

            start_time = time.time()

            with tf.GradientTape() as tape:

                cost = self.compute_cost(
                    X=X,
                    W=W,
                    b=b,
                    Y=Y_norm,
                    R=R,
                    lambda_=lambda_
                )

            grads = tape.gradient(cost, sources=[X, W, b])  # calculate automatically the gradients
            optimizer.apply_gradients(zip(grads, [X, W, b]))
"""

def gradient_optimization(individual, epochs, learning_rate):
    parameters = {}

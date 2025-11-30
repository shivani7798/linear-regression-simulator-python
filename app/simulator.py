# app/simulator.py
import numpy as np

class LinearRegressionGD:
    """
    Univariate linear regression trained with batch gradient descent.
    Provides a train_generator method that yields updates for realtime plotting.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = float(learning_rate)
        self.m = 0.0
        self.b = 0.0
        self.X = None
        self.y = None

    def set_data(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if X.shape != y.shape:
            raise ValueError("X and y must have same shape")
        self.X = X
        self.y = y
        # reset params when data is set
        self.m = 0.0
        self.b = 0.0

    def predict(self, X=None):
        if X is None:
            X = self.X
        return self.m * X + self.b

    def compute_loss(self):
        preds = self.predict()
        return np.mean((self.y - preds) ** 2)

    def gradients(self):
        N = len(self.X)
        preds = self.predict()
        error = self.y - preds
        dm = (-2.0 / N) * np.sum(self.X * error)
        db = (-2.0 / N) * np.sum(error)
        return dm, db

    def train_generator(self, epochs=200, update_every=1, yield_updates=True):
        """
        Train and yield (epoch, m, b, loss) every `update_every` epochs.
        If yield_updates=False, will run silently and return final m,b,loss.
        """
        if self.X is None or self.y is None:
            raise ValueError("Data not set. Call set_data(X, y) first.")

        losses = []
        for epoch in range(1, int(epochs) + 1):
            dm, db = self.gradients()
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

            loss = self.compute_loss()
            losses.append(loss)

            if yield_updates and (epoch % update_every == 0 or epoch == epochs):
                yield epoch, self.m, self.b, loss

        if not yield_updates:
            return self.m, self.b, losses

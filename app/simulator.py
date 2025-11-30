import numpy as np
Usage:
model = LinearRegressionGD(learning_rate=0.01)
model.set_data(X, y)
m, b, losses = model.train(epochs=200)
"""
def __init__(self, learning_rate=0.01, verbose=False):
self.learning_rate = float(learning_rate)
self.verbose = verbose
self.m = 0.0
self.b = 0.0
self.X = None
self.y = None


def set_data(self, X, y):
X = np.array(X, dtype=float)
y = np.array(y, dtype=float)
assert X.shape == y.shape, "X and y must be same shape"
self.X = X
self.y = y


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


def train(self, epochs=200, callback=None, reset_params=True):
if self.X is None or self.y is None:
raise ValueError("Data not set. Call set_data(X, y) first.")


if reset_params:
self.m = 0.0
self.b = 0.0


losses = []
for epoch in range(1, int(epochs) + 1):
dm, db = self.gradients()
self.m -= self.learning_rate * dm
self.b -= self.learning_rate * db


loss = self.compute_loss()
losses.append(loss)


if callback is not None:
callback(epoch, self.m, self.b, loss)


if self.verbose and epoch % max(1, epochs // 10) == 0:
print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} m: {self.m:.4f} b: {self.b:.4f}")


return self.m, self.b, losses

import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X=np.asarray(X)
    y=np.asarray(y)
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features)
    b=0.0

    # Training loop
    for _ in range(steps):
        # Linear combination
        linear_model = np.dot(X,w)+b

        # Apply sigmoid function
        y_predicted = 1/(1+np.exp(-linear_model))

        # Compute gradients
        dw = (1/n_samples)*np.dot(X.T, (y_predicted-y))
        db = (1/n_samples)*np.sum(y_predicted-y)

        # Update parameters
        w -= lr*dw
        b -= lr*db
    return w.tolist(), float(b)
    pass
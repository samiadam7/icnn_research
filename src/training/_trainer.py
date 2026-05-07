import torch
import torch.nn as nn


def train(model, y_train, t_train, y_test, t_test,
          epochs=500, lr=1e-3, weight_decay=1e-4, batch_size=256):
    """Train a convex model with mini-batch SGD and return loss curves.

    project_weights() is called after every optimizer step to maintain
    the non-negativity constraint required for convexity (no-op for exp/softplus).
    """
    optimizer = model.get_optimizer(lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_losses, test_losses = [], []

    for _ in range(epochs):
        idx = torch.randperm(y_train.shape[0])
        y_train, t_train = y_train[idx], t_train[idx]

        model.train()
        epoch_loss = 0.0
        for start in range(0, y_train.shape[0], batch_size):
            yb = y_train[start:start + batch_size]
            tb = t_train[start:start + batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(yb), tb)
            loss.backward()
            optimizer.step()
            model.project_weights()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(y_test), t_test).item()

        train_losses.append(epoch_loss / (y_train.shape[0] / batch_size))
        test_losses.append(test_loss)

    return train_losses, test_losses

import torch


def make_dataset(f, n=2000, y_range=(-3, 3), seed=42):
    """Generate a 1D dataset and return z-score normalized targets."""
    torch.manual_seed(seed)
    y = torch.FloatTensor(n, 1).uniform_(*y_range)
    target = f(y)
    mean, std = target.mean(), target.std()
    return y, (target - mean) / std, mean, std


def make_dataset_2d(f, n=2000, y_range=(-3, 3), seed=42):
    """Generate a 2D dataset and return z-score normalized targets."""
    torch.manual_seed(seed)
    y = torch.FloatTensor(n, 2).uniform_(*y_range)
    target = f(y)
    mean, std = target.mean(), target.std()
    return y, (target - mean) / std, mean, std

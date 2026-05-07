import torch
import torch.nn as nn
import torch.nn.functional as F

# All three strategies guarantee W_z >= 0, which is required for convexity.
# clip: projection — simplest, but non-smooth at 0.
# exp: reparametrize W_z = exp(W̃_z); always positive but needs weight decay on W̃_z
#      to prevent exp from diverging during training.
# softplus: reparametrize W_z = softplus(W̃_z); smoother than clip, more stable than exp.
VALID_CONSTRAINTS = ("clip", "exp", "softplus")


class FICNN(nn.Module):
    def __init__(self, y_dim: int, hidden: list[int], constraint: str = "clip"):
        super().__init__()
        assert constraint in VALID_CONSTRAINTS, (
            f"constraint must be one of {VALID_CONSTRAINTS}"
        )
        self.constraint = constraint

        dims = [y_dim] + hidden + [1]
        self.num_layers = len(dims) - 1

        # Passthrough from input y to every layer — unconstrained.
        self.W_y = nn.ParameterList(
            [
                nn.Parameter(torch.randn(dims[i + 1], y_dim) * 0.01)
                for i in range(self.num_layers)
            ]
        )

        # Recurrent weights W_z must be non-negative to preserve convexity.
        if constraint == "clip":
            self.W_z = nn.ParameterList(
                [
                    nn.Parameter(torch.abs(torch.randn(dims[i + 1], dims[i])) * 0.01)
                    for i in range(1, self.num_layers)
                ]
            )
        else:
            # exp(0) = 1, softplus(0) ≈ 0.69 — reasonable starting point.
            self.W_z = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(dims[i + 1], dims[i]))
                    for i in range(1, self.num_layers)
                ]
            )

        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(dims[i + 1])) for i in range(self.num_layers)]
        )

        self.activation = nn.ReLU()

    def get_W_z(self, i: int) -> torch.Tensor:
        W = self.W_z[i]
        if self.constraint == "clip":
            return W
        elif self.constraint == "exp":
            return torch.exp(W)
        elif self.constraint == "softplus":
            return F.softplus(W)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(y.shape[0], 1, device=y.device)  # z_0 = 0 (no bias at input)

        for i in range(self.num_layers):
            pre_act = y @ self.W_y[i].T + self.b[i]
            if i > 0:
                pre_act = pre_act + z @ self.get_W_z(i - 1).T
            z = self.activation(pre_act) if i < self.num_layers - 1 else pre_act

        return z  # (batch, 1)

    def project_weights(self):
        # Clip constraint requires explicit projection after each optimizer step.
        # exp/softplus are handled implicitly by the reparametrization.
        if self.constraint == "clip":
            with torch.no_grad():
                for W in self.W_z:
                    W.clamp_(min=0.0)

    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 0.0):
        # For exp, regularize only W̃_z to keep exp(W̃_z) from exploding.
        # Applying weight decay to W_y or b would distort the function fit.
        if self.constraint == "exp" and weight_decay > 0:
            return torch.optim.AdamW(
                [
                    {"params": self.W_z, "weight_decay": weight_decay},
                    {"params": self.W_y, "weight_decay": 0.0},
                    {"params": self.b, "weight_decay": 0.0},
                ],
                lr=lr,
            )
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.0)


class FICNNSoftplusShifted(FICNN):
    """FICNN with softplus constraint, W̃_z initialized at -2.

    softplus(-2) ≈ 0.13 gives more capacity to represent near-zero weights
    early in training, consistently outperforming the default init=0 baseline.
    """

    def __init__(self, y_dim: int, hidden: list[int]):
        super().__init__(y_dim=y_dim, hidden=hidden, constraint="softplus")
        with torch.no_grad():
            for W in self.W_z:
                W.fill_(-2.0)

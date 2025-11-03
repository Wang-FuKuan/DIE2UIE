import torch
class L1_loss(torch.nn.Module):
    """L1 loss."""
    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self, X, Y):
        loss = torch.mean(torch.abs(X-Y))
        return loss
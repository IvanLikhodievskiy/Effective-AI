import torch
import torch.nn as nn

def compute_smooth_scale(X_calib: torch.Tensor, W: torch.Tensor, alpha: float = 0.5):
    act_max = torch.max(torch.abs(X_calib.reshape(-1, X_calib.shape[-1])), dim=0).values
    
    weight_max = torch.max(torch.abs(W), dim=0).values
    
    s = (act_max ** alpha) / (weight_max ** (1 - alpha)).clamp(min=1e-5)
    return s

def smooth_linear_layer_(linear: nn.Linear, s: torch.Tensor):
    with torch.no_grad():
        linear.weight.mul_(s)

class SmoothQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, s: torch.Tensor):
        super().__init__()
        self.linear = linear
        self.register_buffer('s_inv', 1.0 / s)
        
    def forward(self, x: torch.Tensor):
        x_smoothed = x * self.s_inv
        return self.linear(x_smoothed)


torch.manual_seed(42)

in_features = 8
out_features = 4
linear = nn.Linear(in_features, out_features, bias=False)

X = torch.randn(2, 5, in_features)
X[:, :, 2] *= 100.0

with torch.no_grad():
    y_original = linear(X)

s = compute_smooth_scale(X, linear.weight.data, alpha=0.5)

print("Original max in activations:", X.abs().max().item())
print("Smoothened max in activations:  ", (X / s).abs().max().item())
print()

smooth_linear_layer_(linear, s)

sq_linear = SmoothQuantLinear(linear, s)

with torch.no_grad():
    y_smooth = sq_linear(X)

diff = torch.max(torch.abs(y_original - y_smooth)).item()
print(f"Maximal result difference: {diff:.6f}")
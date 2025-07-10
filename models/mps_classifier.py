import torch
import torch.nn as nn

class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=10, d_feature=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.d_feature = d_feature

        self.cores = nn.ParameterList()
        for i in range(input_dim):
            left_dim = 1 if i == 0 else bond_dim
            right_dim = 1 if i == input_dim - 1 else bond_dim
            core = nn.Parameter(torch.randn(left_dim, d_feature, right_dim) * 0.1)
            self.cores.append(core)

        # 💡 Replace the linear classifier with per-class output tensors
        self.output_tensors = nn.Parameter(
            torch.randn(output_dim, bond_dim, 1) * 0.1
        )

    def feature_map(self, x):
        x = x.view(x.size(0), -1)
        return torch.stack([
            torch.cos(torch.pi * x),
            torch.sin(torch.pi * x)
        ], dim=-1)

    def forward(self, x):
        B = x.size(0)
        x = self.feature_map(x)

        result = None
        for i in range(self.input_dim):
            core = self.cores[i]
            x_i = x[:, i]
            v = torch.einsum('bd,ldr->blr', x_i, core)
            result = v if result is None else torch.bmm(result, v)

        logits = torch.einsum('bdi,cdi->bc', result, self.output_tensors)
        return logits  # Shape: (B, C)

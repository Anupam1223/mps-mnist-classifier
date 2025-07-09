import torch
import torch.nn as nn

class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim

        self.cores = nn.ParameterList()
        for i in range(input_dim):
            left_dim = 1 if i == 0 else bond_dim
            right_dim = 1 if i == input_dim - 1 else bond_dim
            core = nn.Parameter(torch.randn(left_dim, 2, right_dim) * 0.1)
            self.cores.append(core)

        self.fc = nn.Linear(1, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)

        x_stack = torch.stack([1 - x, x], dim=-1)  # No binarization



        result = None
        for i in range(self.input_dim):
            core = self.cores[i]
            v = torch.einsum('bi, lir -> blr', x_stack[:, i], core)
            result = v if result is None else torch.bmm(result, v).contiguous()

        out = self.fc(result.squeeze(1))
        return out

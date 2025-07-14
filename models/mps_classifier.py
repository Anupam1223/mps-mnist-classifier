import torch
import torch.nn as nn
import torch.nn.functional as F

class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=10, d_feature=2, dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.d_feature = d_feature
        self.dtype = dtype

        # MPS cores
        self.cores = nn.ParameterList()

        gain = nn.init.calculate_gain('linear')

        for i in range(input_dim):
            if i == 0:
                fan_in = d_feature
                fan_out = d_feature * bond_dim
                std = gain * (2.0 / (fan_in + fan_out))**0.5
                core = nn.Parameter(torch.empty(1, d_feature, bond_dim, dtype=self.dtype).uniform_(-std, std))
            elif i == input_dim - 1:
                fan_in = bond_dim * d_feature
                fan_out = d_feature * bond_dim
                std = gain * (2.0 / (fan_in + fan_out))**0.5
                core = nn.Parameter(torch.empty(bond_dim, d_feature, bond_dim, dtype=self.dtype).uniform_(-std, std))
            else:
                fan_in = bond_dim * d_feature
                fan_out = d_feature * bond_dim
                std = gain * (2.0 / (fan_in + fan_out))**0.5
                core = nn.Parameter(torch.empty(bond_dim, d_feature, bond_dim, dtype=self.dtype).uniform_(-std, std))
            self.cores.append(core)

        self.classifier = nn.Parameter(torch.randn(output_dim, bond_dim, dtype=self.dtype) * 0.1) # Revert to original classifier init for simplicity
        # Use Xavier for classifier too if preferred for consistency
        # fan_in_classifier = bond_dim
        # fan_out_classifier = output_dim
        # std_classifier = gain * (2.0 / (fan_in_classifier + fan_out_classifier))**0.5
        # self.classifier = nn.Parameter(torch.empty(output_dim, bond_dim, dtype=self.dtype).uniform_(-std_classifier, std_classifier))


    def feature_map(self, x):
        x = x.view(x.size(0), -1)
        return torch.stack([
            1 - x,
            x
        ], dim=-1).to(self.dtype)

    def forward(self, x):
        B = x.shape[0]
        features = self.feature_map(x)

        effective_tensors = []
        for i in range(self.input_dim):
            contracted_tensor = torch.einsum('ldh,bd->blh', self.cores[i], features[:, i, :])
            effective_tensors.append(contracted_tensor)

        current_tensors = effective_tensors
        log_total_norm = torch.zeros(B, dtype=self.dtype, device=x.device)

        while len(current_tensors) > 1:
            next_tensors = []
            for i in range(0, len(current_tensors), 2):
                if i + 1 < len(current_tensors):
                    tensor1 = current_tensors[i]
                    tensor2 = current_tensors[i+1]

                    contracted_pair = torch.matmul(tensor1, tensor2)

                    # --- START OF THE CHANGE ---

                    # This is the new, fast normalization method
                    batch_norms = torch.sqrt(torch.sum(contracted_pair**2, dim=(-2, -1)))
                    
                    # --- END OF THE CHANGE ---
                    
                    batch_norms_clamped = batch_norms.clamp(min=1e-12)
                    
                    log_total_norm = log_total_norm + torch.log(batch_norms_clamped)
                    
                    contracted_pair = contracted_pair / batch_norms_clamped.view(B, 1, 1)

                    next_tensors.append(contracted_pair)
                else:
                    next_tensors.append(current_tensors[i])
            current_tensors = next_tensors

        mps_output = current_tensors[0].squeeze(1)
        
        logits = torch.matmul(mps_output, self.classifier.t())
        logits = logits + log_total_norm.view(B, 1)

        return logits
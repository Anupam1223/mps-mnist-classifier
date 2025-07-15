import torch
import torch.nn as nn
import torch.nn.functional as F

class MPSClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim=10, d_feature=2, dtype=torch.float32):
        """
        Matrix Product State (MPS) Classifier.
        
        Args:
            input_dim (int): The number of features in the input sequence (e.g., 64 for an 8x8 image).
            output_dim (int): The number of output classes (e.g., 10 for MNIST).
            bond_dim (int): The maximum bond dimension of the MPS. This controls the model's capacity.
            d_feature (int): The dimension of the feature-mapped input pixels. Default is 2.
            dtype (torch.dtype): The data type for the model's parameters.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.d_feature = d_feature
        self.dtype = dtype

        # MPS cores initialization
        self.cores = nn.ParameterList()
        gain = nn.init.calculate_gain('leaky_relu') # A common choice for gain

        for i in range(input_dim):
            if i == 0:
                # First core: (left_bond=1, physical_dim, right_bond)
                core_shape = (1, d_feature, bond_dim)
            elif i == input_dim - 1:
                # Last core: (left_bond, physical_dim, right_bond=bond_dim for classification)
                core_shape = (bond_dim, d_feature, bond_dim)
            else:
                # Middle cores: (left_bond, physical_dim, right_bond)
                core_shape = (bond_dim, d_feature, bond_dim)
            
            core = nn.Parameter(torch.empty(core_shape, dtype=self.dtype))
            nn.init.xavier_uniform_(core, gain=gain)
            self.cores.append(core)

        # Classifier layer to map the final MPS output vector to class logits
        self.classifier = nn.Parameter(torch.empty(output_dim, bond_dim, dtype=self.dtype))
        nn.init.xavier_uniform_(self.classifier, gain=gain)


    def feature_map(self, x):
        """ Maps input pixel values [0, 1] to a 2D feature space. """
        x = x.view(x.size(0), -1) # Flatten image
        # Simple linear feature map
        return torch.stack([1 - x, x], dim=-1).to(self.dtype)

    def forward(self, x):
        """
        Performs the forward pass by contracting the MPS with the input features.
        This implementation uses a batched tree contraction for efficiency.
        """
        B = x.shape[0]
        features = self.feature_map(x)

        # Step 1: Contract each core with its corresponding feature vector.
        effective_tensors = []
        for i in range(self.input_dim):
            contracted_tensor = torch.einsum('ldh,bd->blh', self.cores[i], features[:, i, :])
            effective_tensors.append(contracted_tensor)

        current_tensors = effective_tensors
        log_total_norm = torch.zeros(B, dtype=self.dtype, device=x.device)

        # Step 2: Contract the virtual bonds in a tree-like structure.
        while len(current_tensors) > 1:
            
            # --- START OF BUG FIX ---
            # The first tensor in the MPS chain has a unique shape (B, 1, D), while others are (B, D, D).
            # This causes torch.stack to fail. We fix this by identifying the special tensor,
            # padding it to match the others before the batched operation, and then slicing the
            # result back to its correct shape.
            is_first_tensor_special = current_tensors[0].shape[1] == 1
            
            tensors_for_contraction = list(current_tensors) # Make a copy to modify

            if is_first_tensor_special:
                first_tensor = tensors_for_contraction.pop(0)
                # Pad the middle dimension from 1 to bond_dim
                padded_first = F.pad(first_tensor, (0, 0, 0, self.bond_dim - 1))
                tensors_for_contraction.insert(0, padded_first)
            # --- END OF BUG FIX ---

            is_odd = len(tensors_for_contraction) % 2 != 0
            if is_odd:
                last_tensor = tensors_for_contraction.pop()

            # If there are no pairs to contract, just handle the leftover odd tensor
            if not tensors_for_contraction:
                current_tensors = [last_tensor]
                continue
            
            # Batching the Contractions
            num_pairs = len(tensors_for_contraction) // 2
            stack1 = torch.stack(tensors_for_contraction[0::2])
            stack2 = torch.stack(tensors_for_contraction[1::2])

            stack1 = stack1.permute(1, 0, 2, 3)
            stack2 = stack2.permute(1, 0, 2, 3)
            
            contracted_pairs = torch.matmul(stack1, stack2)

            # Normalization of the Batched Tensors
            batch_norms = torch.sqrt(torch.sum(contracted_pairs**2, dim=(-2, -1))).clamp(min=1e-12)
            log_total_norm += torch.sum(torch.log(batch_norms), dim=1)
            normalized_pairs = contracted_pairs / batch_norms.view(B, num_pairs, 1, 1)

            # Unstack the results back into a list
            next_tensors = [t for t in normalized_pairs.permute(1, 0, 2, 3)]
            
            # --- START OF BUG FIX ---
            # If we padded the first tensor, we must "un-pad" the result by slicing.
            if is_first_tensor_special:
                next_tensors[0] = next_tensors[0][:, 0:1, :]
            # --- END OF BUG FIX ---

            if is_odd:
                next_tensors.append(last_tensor)
            
            current_tensors = next_tensors

        mps_output = current_tensors[0].squeeze(1)
        
        logits = torch.matmul(mps_output, self.classifier.t())
        logits = logits + log_total_norm.view(B, 1)

        return logits

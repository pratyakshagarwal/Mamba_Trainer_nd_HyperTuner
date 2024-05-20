# !pip install kaleido
# !pip install mamba-ssm
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from .config_nd_params import MambaConfig

#mamba_model.py
class MambaModel(nn.Module):
    def __init__(self, config, device, dtype=torch.float32):
        """
        Initialize the MambaModel.

        Parameters:
        - config (MambaConfig): Configuration object for the model.
        - device (torch.device): Device on which the model will be trained and run.
        - dtype (torch.dtype, optional): Data type for the model's parameters. Defaults to torch.float32.
        """
        super(MambaModel, self).__init__()
        self.config = config
        # Initialize the MambaLMHeadModel using the provided configuration
        self.model = MambaLMHeadModel(config, device=device, dtype=dtype)
        
        print(f"Trainable Parameters {sum([p.numel() for p in self.model.parameters()])}")

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """
        Perform a forward pass through the model.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - y (torch.Tensor, optional): Target tensor for calculating loss. Defaults to None.

        Returns:
        - logits (torch.Tensor): Output logits.
        - loss (torch.Tensor or None): Computed loss if y is provided, otherwise None.
        """
        # Forward pass through the model
        logits = self.model(x)[0]
        loss = None
        if y is not None:
            logits = logits.float()
            # Calculate cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=2):
        """
        Generate new tokens using the model.

        Parameters:
        - idx (torch.Tensor): Input tensor representing the starting sequence.
        - max_new_tokens (int): Maximum number of tokens to generate.
        - temperature (float, optional): Softmax temperature for sampling. Defaults to 1.0.
        - top_k (int, optional): Number of top tokens to consider for sampling. Defaults to 2.

        Returns:
        - idx (torch.Tensor): Tensor containing the generated sequence.
        """
        for _ in range(max_new_tokens):
            # Truncate the input if it exceeds the block size
            idx_cond = idx if idx.size(-1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Sample the next token using multinomial distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def save_model(self, output_dir):
        """
        Save the model and its configuration to the specified directory.

        Parameters:
        - output_dir (str): Path to the directory where the model will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        num_params = sum(p.numel() for p in self.model.parameters())
        # Construct model filename based on the number of parameters
        model_filename = f"mamba_model_{num_params // 1_000_000}m.pt"
        model_filepath = os.path.join(output_dir, model_filename)
        # Save model parameters
        torch.save(self.model.state_dict(), model_filepath)
        # Save model configuration as JSON
        config_dict = self.config.__dict__
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Model saved to {model_filepath}")

    def load_model(self, output_dir, model_filename, device):
        """
        Load the model and its configuration from the specified directory.

        Parameters:
        - output_dir (str): Path to the directory where the model is saved.
        - model_filename (str): Filename of the model.
        - device (torch.device): Device on which the model will be loaded.

        Returns:
        - None
        """
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = MambaConfig(**config_dict)
        # Reinitialize the model with the loaded configuration
        self.model = MambaLMHeadModel(config, device=device, dtype=torch.float32)
        model_filepath = os.path.join(output_dir, model_filename)
        # Load model parameters
        self.model.load_state_dict(torch.load(model_filepath, map_location=device))
        self.model.to(device)
        print(f"Model loaded from {model_filepath}")

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config_nd_params import MambaConfig, TrainerParams
from model import MambaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_or_build_tokenizer(ds: str, tokenizer_path: str = "mamba_ssm_tokenizer.json") -> Tokenizer:
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]"])
        tokenizer.train_from_iterator([ds], trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        logger.info("Tokenizer built and saved.")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info("Tokenizer loaded from file.")
    return tokenizer

class MambaTrainer:
    def __init__(self, config: MambaConfig, training_params: TrainerParams, device: torch.device, dtype: torch.dtype = torch.float32, model: nn.Module = None):
        """
        Initialize the MambaTrainer.
        """
        self.model = model if model else MambaModel(config, device, dtype)
        self.config = config
        self.tparams = training_params
        self.device = device
        self.dtype = dtype
    
    def train(self, train_dataset, val_dataset, num_epochs: int, verbose: int = 1, save_model: bool = True, optimizer=None) -> None:
        """
        Train the MambaModel.
        """
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.tparams.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.tparams.batch_size, shuffle=True)
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.tparams.lr)
        
        if verbose == 1:
            logger.info(f"Using {self.device}")

        for epoch in range(num_epochs):
            self.model.train()
            for iter in range(200):
                xb, yb = self.get_batch('train')
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if verbose == 1:
                losses = self.estimate_loss()
                print(f"Epoch: {epoch + 1} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
            
            # Save checkpoint
            if save_model:
                chkpt_dir = Path(f"'output'/chkpt")
                chkpt_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = f"{chkpt_dir}/checkpoint_epoch_{epoch + 1}.pt"
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

        if save_model:
            self.model.save_model('output')
            logger.info("Model saved to 'output'")

    @torch.no_grad()
    def estimate_loss(self) -> dict:
        """
        Estimate the loss on training and validation data.
        """
        losses = {'train': 0, 'val': 0}
        self.model.eval()
        for split in ['train', 'val']:
            batch_losses = []
            for _ in range(100):
                xb, yb = self.get_batch(split)
                logits, loss = self.model(xb, yb)
                batch_losses.append(loss.item())
            losses[split] = sum(batch_losses) / len(batch_losses)
        return losses
    
    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data for training or validation.
        """
        dataloader = self.train_dataloader if split == 'train' else self.val_dataloader
        for x, y in dataloader:
            return x.to(self.device), y.to(self.device)




if __name__ == '__main__':
    pass

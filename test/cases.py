import torch
from trainer import MambaTrainer, get_or_build_tokenizer
from hypertune import HyperparameterOptimizer
from config_nd_params import MambaConfig, TrainerParams
from model import MambaModel
from sample_dataset import TextDataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20

    ds = open('train_nd_tune\input.txt', 'r').read()
    tokenizer = get_or_build_tokenizer(ds)

    data = tokenizer.encode(ds).ids
    train_size = int(len(data) * 0.9)
    train_data = data[:train_size]
    val_data = data[train_size:]

    config = MambaConfig(vocab_size=tokenizer.get_vocab_size())
    tparams = TrainerParams()

    train_dataset = TextDataset(train_data, config.block_size)
    val_dataset = TextDataset(val_data, config.block_size)

    trainer = MambaTrainer(config, tparams, tokenizer, device)
    trainer.train(train_dataset, val_dataset, num_epochs)
    
    optimizer = HyperparameterOptimizer(MambaModel, MambaTrainer, config, tparams, tokenizer, train_dataset, val_dataset, device)
    best_params = optimizer.run_optimization(n_trials=2)

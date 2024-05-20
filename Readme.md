# **Mamba-SSM Training and Hyperparameter Optimization**

## **Overview**
This repository provides code for training the Mamba Sequence-to-Sequence Model (SSM) and optimizing its hyperparameters using Optuna. The project includes a custom tokenizer, a dataset class, model training functionality, and hyperparameter optimization with visualization.

## **Repository Structure**
- model.py: Defines the MambaModel class.
- trainer.py: Contains the MambaTrainer class and the get_or_build_tokenizer function.
- config_nd_params.py: Contains the MambaConfig and TrainerParams classes.
- hypertune.py: Implements the HyperparameterOptimizer class for optimizing model hyperparameters.
- sample_dataset.py: Contains the TextDataset class for handling text data.
- main.py: Main script for training the model and running hyperparameter optimization.
- README.md: This file, providing an overview of the repository and usage instructions.

## **Installation**
Ensure you have the required packages installed:
```bash
pip install torch tokenizers optuna matplotlib kaleido
```

## **Components**
### **Tokenizer**
The tokenizer is built or loaded using the get_or_build_tokenizer function

### **Test Case**
**Here is a test case to verify the training and hyperparameter optimization process**:

1. Set Up the Environment: Ensure your environment is correctly set up with the necessary dependencies installed.
2. Prepare the Dataset: Ensure train_nd_tune/input.txt contains your training data.
3. Run the Main Script:

This script performs the following:

- Builds or loads the tokenizer.
- Prepares the dataset for training and validation.
- Initializes the model and trainer.
- Trains the model for a specified number of epochs.
- Runs hyperparameter optimization with a specified number of trials.
- The expected outcome includes a trained model and a set of optimized hyperparameters.

### **Example Usage**
#### **Training:**
```bash
from trainer import MambaTrainer, get_or_build_tokenizer
from config_nd_params import MambaConfig, TrainerParams
from sample_dataset import TextDataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20

ds = open('train_nd_tune/input.txt', 'r').read()
tokenizer = get_or_build_tokenizer(ds)

data = tokenizer.encode(ds).ids
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
val_data = data[train_size:]

config = MambaConfig(vocab_size=tokenizer.get_vocab_size())
tparams = TrainerParams()

train_dataset = TextDataset(train_data, config.block_size)
val_dataset = TextDataset(val_data, config.block_size)

trainer = MambaTrainer(config, tparams, device)
trainer.train(train_dataset, val_dataset, num_epochs)
```
#### **Hyperparameter Optimization:**
```bash
from hypertune import HyperparameterOptimizer
from config_nd_params import MambaConfig, TrainerParams
from sample_dataset import TextDataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 3

ds = open('train_nd_tune/input.txt', 'r').read()
tokenizer = get_or_build_tokenizer(ds)

data = tokenizer.encode(ds).ids
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
val_data = data[train_size:]

config = MambaConfig(vocab_size=tokenizer.get_vocab_size())
tparams = TrainerParams()

train_dataset = TextDataset(train_data, config.block_size)
val_dataset = TextDataset(val_data, config.block_size)

optimizer = HyperparameterOptimizer(MambaModel, MambaTrainer, config, tparams, train_dataset, val_dataset, device)
best_params = optimizer.run_optimization(n_trials=2)
```

### **Conclusion**
This repository provides a comprehensive setup for training the Mamba-SSM model and optimizing its hyperparameters. By following the provided instructions and using the test case, you can ensure that the training and optimization processes work correctly.
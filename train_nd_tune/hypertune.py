import optuna
from optuna.visualization import plot_parallel_coordinate, plot_slice
from pathlib import Path  # Import Path from pathlib
import matplotlib.pyplot as plt
import kaleido
import torch



class HyperparameterOptimizer:
    def __init__(self, model_class, trainer_class, configs, tparams, train_dataset, val_dataset, device, dtype=torch.float32, range=None, num_epochs=3):
        """
        Initialize the HyperparameterOptimizer.

        Parameters:
        - model_class: The class for the model to be optimized.
        - trainer_class: The class for the trainer associated with the model.
        - configs: Configuration object for the model.
        - tparams: Training parameters.
        - tokenizer: Tokenizer object for encoding input sequences.
        - device: Device on which the model will be trained.
        - ds: Raw dataset for training.
        - dtype (torch.dtype, optional): Data type for the model's parameters. Defaults to torch.float32.
        - output_dir (str, optional): Directory path for saving optimization-related files. Defaults to "output".
        - range (dict, optional): Range of hyperparameters for optimization. Defaults to None.
        """
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.configs = configs
        self.tparams = tparams
        self.device = device
        self.dtype = dtype
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.range = None

    def objective(self, trial):
        """
        Objective function for the hyperparameter optimization.

        Parameters:
        - trial: Optuna's Trial object.

        Returns:
        - val_loss: Validation loss obtained with the suggested hyperparameters.
        """
        if self.range is None:
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            d_model = trial.suggest_categorical('d_model', [256, 512, 768])
            n_layer = trial.suggest_int('n_layer', 4, 12)
        else:
            lr = trial.suggest_float('lr', self.range['low'], self.range['high'], log=True)
            batch_size = trial.suggest_categorical('batch_size', self.range['batch_size'])
            d_model = trial.suggest_categorical('d_model', self.range['d_model'])
            n_layer = trial.suggest_int('n_layer', self.range['n_layer'], self.range['n_layer'])

        # Define config with suggested hyperparameters
        self.configs.d_model = d_model
        self.configs.n_layer = n_layer
        self.tparams.batch_size = batch_size
        self.tparams.lr = lr

        # Initialize the model and trainer
        model = self.model_class(self.configs, self.device, self.dtype)
        trainer = self.trainer_class(self.configs, self.tparams, self.tokenizer, self.device, self.dtype, model=model)

        # Train the model
        trainer.train(self.train_dataset, self.val_dataset, self.num_epochs, verbose=0, save_model=False)

        # Estimate validation loss
        losses = trainer.estimate_loss()
        val_loss = losses['val']

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_loss

    def run_optimization(self, n_trials=1, timeout=None, pruner=optuna.pruners.MedianPruner(), output_dir="output"):
        """
        Run the hyperparameter optimization.

        Parameters:
        - n_trials (int, optional): Number of trials for optimization. Defaults to 1.
        - timeout (int, optional): Timeout in seconds. Defaults to None.
        - pruner (optuna.pruners.BasePruner, optional): Pruner object for early-stopping. Defaults to optuna.pruners.MedianPruner().

        Returns:
        - best_params: Dictionary containing the best hyperparameters found during optimization.
        """
        # Create the Optuna study
        study = optuna.create_study(direction='minimize', pruner=pruner)
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        # Print the best hyperparameters
        print("Best hyperparameters: ", study.best_params)
        
        # Create the output directory if it doesn't exist
        output_dir = Path(f"{self.output_dir}/hypertuning_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize the optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig_path = output_dir / "optimization_history.png"
        fig.write_image(str(fig_path))
        
        # Plot parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study, params=['lr', 'batch_size', 'd_model', 'n_layer'])
        fig_path = output_dir / "parallel_coordinate_plot.png"
        fig.write_image(str(fig_path))

        # Plot slice plot
        fig = optuna.visualization.plot_slice(study, params=['lr', 'batch_size', 'd_model', 'n_layer'])
        fig_path = output_dir / "slice_plot.png"
        fig.write_image(str(fig_path))

        return study.best_params

if __name__ == '__main__':
    pass
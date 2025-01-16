import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn


class BaseNeuralNet(nn.Module):
    """
    A base (super) neural network class that handles:
      - Forward pass interface
      - Basic training loop
      - Basic evaluation loop
      - Prediction
    Subclasses should implement the `forward()` method at minimum.

    Usage:
      1) Inherit from BaseNeuralNet.
      2) Override `forward()` with your model architecture.
      3) (Optionally) override or extend `train_step()` or `eval_step()`
         if you have specialized logic.
    """

    def __init__(self):
        super(BaseNeuralNet, self).__init__()

    def forward(self, x):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward() method.")

    def train_step(self, batch, criterion, optimizer):
        """
        Defines a single training step on one batch.
        Subclasses could override for specialized logic (e.g., multiple heads).

        Parameters
        ----------
        batch : tuple
            A tuple of (inputs, targets).
        criterion : callable
            A loss function, e.g., nn.MSELoss().
        optimizer : torch.optim.Optimizer
            An optimizer instance, e.g., optim.Adam().

        Returns
        -------
        float
            The training loss for this batch.
        """
        self.train()
        inputs, targets = batch

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = self(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backpropagation
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_step(self, batch, criterion):
        """
        Defines a single evaluation step (no gradient update).

        Parameters
        ----------
        batch : tuple
            A tuple of (inputs, targets).
        criterion : callable
            A loss function, e.g., nn.MSELoss().

        Returns
        -------
        float
            The evaluation loss for this batch.
        """
        self.eval()
        inputs, targets = batch
        with torch.no_grad():
            outputs = self(inputs)
            loss = criterion(outputs, targets)
        return loss.item()

    def fit(self, train_loader, val_loader, criterion, optimizer, epochs=10, verbose=True):
        """
        Trains the model for a given number of epochs, evaluating on val_loader if provided.

        Parameters
        ----------
        train_loader : DataLoader
            A PyTorch DataLoader for the training set.
        val_loader : DataLoader
            A PyTorch DataLoader for the validation set. Can be None if no validation is used.
        criterion : callable
            A loss function, e.g., nn.MSELoss().
        optimizer : torch.optim.Optimizer
            An optimizer instance, e.g., optim.Adam().
        epochs : int
            Number of epochs to train.
        verbose : bool
            Whether to print epoch losses.

        Returns
        -------
        dict
            A dictionary with keys "train_loss" and "val_loss" containing lists of per-epoch losses.
        """
        history = {"train_loss": [], "val_loss": [] if val_loader else None}

        for epoch in range(epochs):
            # Training loop
            train_losses = []
            for batch in train_loader:
                loss = self.train_step(batch, criterion, optimizer)
                train_losses.append(loss)

            avg_train_loss = sum(train_losses) / len(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation loop
            if val_loader:
                val_losses = []
                for batch in val_loader:
                    loss = self.eval_step(batch, criterion)
                    val_losses.append(loss)
                avg_val_loss = sum(val_losses) / len(val_losses)
                if history["val_loss"] is not None:
                    history["val_loss"].append(avg_val_loss)
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.4f}")

        return history

    def predict(self, inputs):
        """
        Generates predictions for a batch of inputs in eval mode.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data (e.g., shape [batch_size, input_dim]).

        Returns
        -------
        torch.Tensor
            Model outputs.
        """
        self.eval()
        with torch.no_grad():
            return self(inputs)


class MeanVarianceNet(BaseNeuralNet):
    def __init__(self, input_dim=6, hidden_dim=100):
        super(MeanVarianceNet, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_mean = nn.Linear(hidden_dim, 1)
        self.output_var = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()  # forces variance to be a positive value

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        mean = self.output_mean(x)
        variance = self.softplus(self.output_var(x))
        return mean, variance

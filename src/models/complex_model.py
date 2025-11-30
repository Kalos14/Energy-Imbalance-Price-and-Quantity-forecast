"""Complex LSTM model for imbalance price forecasting.

This module implements a deep learning model using LSTM layers for 
predicting imbalance prices from price and quantity sequences.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMNetwork(nn.Module):
    """LSTM neural network for time series forecasting.
    
    Architecture:
        - Input layer (2 features: price and quantity)
        - Stacked LSTM layers with dropout
        - Fully connected output layer
    """
    
    def __init__(
        self, 
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """Initialize the LSTM network.
        
        Args:
            input_size: Number of input features (price + quantity = 2)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between LSTM layers
        """
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)


class ComplexModel:
    """Complex LSTM-based model for imbalance price forecasting.
    
    Uses a multi-layer LSTM network to capture temporal patterns
    in imbalance price and quantity data.
    
    Attributes:
        hidden_size: Number of LSTM hidden units
        num_layers: Number of stacked LSTM layers
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """Initialize the complex model.
        
        Args:
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between layers
            learning_rate: Learning rate for Adam optimizer
            epochs: Number of training epochs
            batch_size: Mini-batch size for training
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = None
        self.is_fitted = False
        self.training_history: List[float] = []
        
    def _build_model(self, input_size: int = 2) -> None:
        """Build the LSTM network.
        
        Args:
            input_size: Number of input features
        """
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'ComplexModel':
        """Fit the model on training data.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, 2)
            y: Target values of shape (n_samples,)
            X_val: Optional validation input sequences
            y_val: Optional validation targets
            verbose: Whether to print training progress
            
        Returns:
            Self for method chaining
        """
        input_size = X.shape[2]
        self._build_model(input_size)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        self.training_history = []
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            self.training_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                val_msg = ""
                if X_val is not None and y_val is not None:
                    val_loss = self._calculate_validation_loss(
                        X_val, y_val, criterion
                    )
                    val_msg = f", Val Loss: {val_loss:.4f}"
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                      f"Train Loss: {avg_loss:.4f}{val_msg}")
                
        self.is_fitted = True
        return self
    
    def _calculate_validation_loss(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        criterion: nn.Module
    ) -> float:
        """Calculate validation loss.
        
        Args:
            X_val: Validation input sequences
            y_val: Validation targets
            criterion: Loss function
            
        Returns:
            Validation loss value
        """
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            outputs = self.model(X_val_tensor)
            loss = criterion(outputs, y_val_tensor)
        return loss.item()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict imbalance prices.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, 2)
            
        Returns:
            Predicted prices of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model configuration
        """
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'is_fitted': self.is_fitted
        }
    
    def get_training_history(self) -> List[float]:
        """Get the training loss history.
        
        Returns:
            List of loss values for each epoch
        """
        return self.training_history
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.get_params()
        }, filepath)
        
    def load_model(self, filepath: str) -> 'ComplexModel':
        """Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint['config']
        
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        return self

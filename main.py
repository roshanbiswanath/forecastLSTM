import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_horizon=1):
        self.data = data
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class TimeSeriesLSTM(L.LightningModule):
    def __init__(self, 
                 input_size=1, 
                 hidden_size=64, 
                 num_layers=2, 
                 dropout=0.2, 
                 sequence_length=60,  # 2 hours of data (60 * 2min)
                 prediction_horizon=1,
                 learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
        
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        prediction = self.output_layer(last_output)
        
        return prediction
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(-1))
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(-1))
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.squeeze(-1))
        
        # Calculate additional metrics
        mae = F.l1_loss(y_hat, y.squeeze(-1))
        
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

class TimeSeriesPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False
        
    def fit_transform(self, data):
        """Fit scaler and transform data"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        self.fitted = True
        return scaled_data.flatten()
    
    def transform(self, data):
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.scaler.transform(data.reshape(-1, 1)).flatten()
    
    def inverse_transform(self, data):
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

def load_and_prepare_data(csv_path, 
                         sequence_length=60, 
                         prediction_horizon=1,
                         train_ratio=0.7,
                         val_ratio=0.15):
    """Load CSV and prepare train/val/test datasets"""
    
    # Load data with encoding handling
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp1252')
    
    # Assuming CSV has columns: ['timestamp', 'value'] or just ['value']
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        data = df['value'].values
    else:
        # Assume first column is the target variable
        data = df.iloc[:, 0].values
    
    # Handle missing values
    data = pd.Series(data).ffill().bfill().values
    
    # Preprocessing
    preprocessor = TimeSeriesPreprocessor()
    scaled_data = preprocessor.fit_transform(data)
    
    # Split data
    n = len(scaled_data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = scaled_data[:train_end]
    val_data = scaled_data[train_end:val_end]
    test_data = scaled_data[val_end:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length, prediction_horizon)
    val_dataset = TimeSeriesDataset(val_data, sequence_length, prediction_horizon)
    test_dataset = TimeSeriesDataset(test_data, sequence_length, prediction_horizon)
    
    return train_dataset, val_dataset, test_dataset, preprocessor

def train_model(csv_path, config=None):
    """Main training function"""
    
    # Default configuration
    default_config = {
        'sequence_length': 60,      # 2 hours of 2-minute intervals
        'prediction_horizon': 1,    # Predict next value
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'max_epochs': 100
    }
    
    if config:
        default_config.update(config)
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, preprocessor = load_and_prepare_data(
        csv_path, 
        default_config['sequence_length'], 
        default_config['prediction_horizon']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=default_config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=default_config['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=default_config['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = TimeSeriesLSTM(
        hidden_size=default_config['hidden_size'],
        num_layers=default_config['num_layers'],
        dropout=default_config['dropout'],
        sequence_length=default_config['sequence_length'],
        prediction_horizon=default_config['prediction_horizon'],
        learning_rate=default_config['learning_rate']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=default_config['max_epochs'],
        callbacks=[early_stopping, checkpoint_callback],
        accelerator='auto',
        devices='auto',
        log_every_n_steps=50,
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    trainer.test(model, test_loader, ckpt_path='best')
    
    return model, preprocessor, trainer

# Example usage
if __name__ == "__main__":
    # Configuration for your specific use case
    config = {
        'sequence_length': 60,      # 2 hours lookback
        'prediction_horizon': 1,    # Predict next 2-minute value
        'hidden_size': 128,         # Increased for complex patterns
        'num_layers': 3,            # Deeper network
        'dropout': 0.3,             # Regularization
        'learning_rate': 0.0005,    # Conservative learning rate
        'batch_size': 64,           # Larger batch for stability
        'max_epochs': 200
    }
    
    # Replace with your CSV path
    csv_path = "EventReady.csv"
    
    model, preprocessor, trainer = train_model(csv_path, config)
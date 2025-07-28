import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import lightning as L

class LightningLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)
    def forward(self, input):
        print(len(input))
        if input.dim() == 1:
            seq_len = len(input)
            input_trans = input.view(seq_len, 1, 1)  # [4, 1, 1]
        else:
            batch_size, seq_len = input.shape
            input_trans = input.view(seq_len, batch_size, 1)  # [4, 1, 1]
        
        lstm_out, _ = self.lstm(input_trans)
        pred = lstm_out[-1]  # Get last time step
        return pred.squeeze()  # Remove extra dimensions
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = F.mse_loss(output_i, label_i)
        self.log('train_loss', loss)
        if label_i == 0:
            self.log('out_0', output_i)
        else:
            self.log('out_1', output_i)
        return loss


model = LightningLSTM()
print(model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

trainer = L.Trainer(max_epochs=300, enable_progress_bar=True, log_every_n_steps=2)

inputs = torch.tensor([[0., 0.5, 0.25, 1.],[1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer.fit(model, train_dataloaders=dataloader)

print(model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print(model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

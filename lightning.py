import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class DataModule:
    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, batch_size: int, train_test_split = 0.8):
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def _get_dataloader(self, train = True, shuffle = True):
        
        data = self.train_dataset if train else self.test_dataset
        return DataLoader(data, batch_size = self.batch_size, shuffle = shuffle)
    
    def train_dataloader(self):
        return self._get_dataloader(train = True)
    
    def test_dataloader(self):
        return self._get_dataloader(train = False)
    

class Trainer:
    '''
    The base class for training models with data
    '''
    def __init__(self, max_epochs, device, num_gpus = 0):
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.device = device

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = (len(self.test_dataloader)
                                if self.test_dataloader is not None else 0)
    
    def prepare_model(self, model: nn.Module):
        self.model = model.to(self.device)

    def fit(self, model, data): 
        self.prepare_data(data)
        self.prepare_model(model)
        self.model.train()
        self.optim = self.model.configure_optimizers()
        self.epoch = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        running_loss = 0.0
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optim["optimizer"].zero_grad()
            output = self.model(inputs)
            loss = self.optim["loss"](output, labels)
            loss.backward()
            self.optim["optimizer"].step()
            running_loss += loss.item()  # Add the loss value for monitoring
        print(f"Epoch [{self.epoch+1}/{self.max_epochs}], Loss: {running_loss/len(self.train_dataloader):.4f}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {correct/total}")

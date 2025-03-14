#!/usr/bin/env python
# coding: utf-8
import clip
import torch
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os

LOG_WANDB = False
model_name = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = "./data"

epochs = 100
learning_rate = 0.01

if LOG_WANDB:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="clip-for-caltech101",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
    )


class CLIPEmbeddedCaltech101(Dataset):
    '''
    A Dataset with each image embedded by an clip model. 
    Valid model_names can be seen from clip.available_models()
    Each entry has one embedding vector and one label. 
    '''
    def __init__(self, root, model_name, load_embeddings = None, download = True):
        """
        Initialize the dataset with CLIP embeddings.
        
        Args:
            root (str): Root directory of the dataset
            model_name (str): Name of the CLIP model to use
            load_embeddings (str, optional): Path to pre-computed embeddings file
            download (bool, optional): Whether to download the dataset
        """
        self.data = datasets.Caltech101(root = root, target_type = "category", download = download)
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.labels = [self.data[i][1] for i in range(len(self.data))]
       
        #try to load the embeddings from the embeddings_dir if file not found create new ones
        try:
            self.image_embeddings = torch.load(load_embeddings)
        except:
            print ("Embedding not provided, calculating new embeddings")
            self.image_embeddings = self._embed_images(save = True)

        self.labels = [self.data[i][1] for i in range(len(self.data))]
        assert(len(self.labels) == self.image_embeddings.shape[0])

    def _embed_images(self, save = False):
        print("Embedding Images using CLIP ...")
        batch_size = 32
        num_images = len(self.data)
        image_embeddings = []
        for i in range(0, num_images, batch_size):
            if i % (320) == 0:
                print(f"{i}/{num_images}")
            preprocessed_images = [self.preprocess(self.data[i][0]) for i in range(i, min(i+batch_size, num_images))]
            image_batch = torch.stack(preprocessed_images).to(device)
            with torch.no_grad():
                image_emb_batch = self.model.encode_image(image_batch).cpu()
            image_embeddings.append(image_emb_batch)
        
        image_embeddings = torch.cat(image_embeddings)
        if save:
            torch.save(image_embeddings, "image_embeddings.pt")

        return image_embeddings

    def __len__(self):
        return len(self.image_embeddings)
    
    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.labels[idx]
    
    @property
    def categories(self):
        return self.data.categories

class DataModule:
    def __init__(self, dataset: Dataset, batch_size: int, train_test_split = 0.8):
        self.batch_size = batch_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(dataset, [train_test_split, 1 - train_test_split])
    
    def _get_dataloader(self, train = True, shuffle = True):
        
        data = self.train_dataset if train else self.test_dataset
        return DataLoader(data, batch_size = self.batch_size, shuffle = shuffle)
    
    def train_dataloader(self):
        return self._get_dataloader(train = True)
    
    def test_dataloader(self):
        return self._get_dataloader(train = False)
    

class CLIPEmbeddingClassifier(nn.Module):
    '''
    A basic neural network with 3 linear layers for classification. Takes 
    CLIP embeddings as inputs and outputs one of the 101 classes.
    Requires: The training and inference should be take in embeddings of the 
    same CLIP model. 
    '''
    def __init__(self):
        """
        Initialize the classifier model.
        
        Args:
            model_name (str): Name of the CLIP model to use
            class_names (list): List of class names for the classifier
        """
        super(CLIPEmbeddingClassifier, self).__init__()
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 101)
        self._init_weights()
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Applies Xavier initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Sets biases to zero

    def configure_optimizers(self):
        return {'optimizer': optim.SGD(self.parameters(), lr = learning_rate),
                'loss': nn.CrossEntropyLoss()}

class Trainer:
    '''
    The base class for training models with data
    '''
    def __init__(self, max_epochs, num_gpus = 0):
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus

    def prepare_data(self, data: DataModule):
        self.train_dataloader = data.train_dataloader()
        self.test_dataloader = data.test_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = (len(self.test_dataloader)
                                if self.test_dataloader is not None else 0)
    
    def prepare_model(self, model: nn.Module):
        self.model = model

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
            inputs, labels = inputs.to(device), labels.to(device)
            self.optim["optimizer"].zero_grad()
            output = self.model(inputs)
            loss = self.optim["loss"](output, labels)
            loss.backward()
            self.optim["optimizer"].step()
            running_loss += loss.item()  # Add the loss value for monitoring
        print(f"Epoch [{self.epoch+1}/{self.max_epochs}], Loss: {running_loss/len(self.train_dataloader):.4f}")
        if LOG_WANDB: wandb.log({"loss": loss})

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {correct/total}")

def predict_image_class(model, clip_model_name, image_path, category_names = None):
    image = Image.open(image_path)
    clip_model, preprocess = clip.load(clip_model_name, device=device)
    preprocessed_image = preprocess(image)
    with torch.no_grad():
        #unsqueeze because model expects a batch
        image_embedding = clip_model.encode_image(preprocessed_image.unsqueeze(0))
        output = model(image_embedding)
        predicted_class = torch.argmax(output, 1)
    
    if category_names == None: 
        print(predicted_class)
    else:
        print(category_names[predicted_class])

def main():
    #load the dataset
    dataset = CLIPEmbeddedCaltech101(model_name = model_name, 
                                     root = data_path,
                                     load_embeddings = 'image_embeddings.pt', 
                                     download = True)
    
    model = CLIPEmbeddingClassifier()
    data = DataModule(dataset, batch_size = 32)
    trainer = Trainer(epochs)
    trainer.fit(model, data)
    trainer.evaluate()

    predict_image_class(
        model = model,
        clip_model_name= model_name,
        image_path="sample.jpg",
        category_names=dataset.categories
    )

if __name__ == "__main__":
    main()
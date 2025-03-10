#!/usr/bin/env python
# coding: utf-8
import clip
import scipy
import torch
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Verify the download by listing the directory contents.
# data = datasets.Caltech101(root = "./data", target_type = "category", download = True)
model_name = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = "./data"

class CLIPEmbeddedCaltech101(Dataset):
    '''
    A Dataset with each image embedded by an clip model. 
    Valid model_names can be seen from clip.available_models()
    Each entry has one embedding vector and one label. 
    '''
    def __init__(self, root, model_name, load_embeddings = None, download = True):
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

class CLIPEmbeddingClassifier(nn.Module):
    '''
    A basic neural network with 3 linear layers for classification. Takes 
    CLIP embeddings as inputs and outputs one of the 101 classes.
    '''
    def __init__(self, model_name, class_names):
        super(CLIPEmbeddingClassifier, self).__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.class_names = class_names
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

    def predict_image_class(self, image_path):
        image = Image.open(image_path)
        image = self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
            output = self.forward(image_embedding)
            predicted_class = torch.argmax(output, 1)
        return self.class_names[predicted_class]
    

def train(model, train_dataloader, n_epochs=100):
    '''
    train the model with the given train_dataloader for n_epochs.
    Returns the trained model.
    '''
    model.train()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # Add the loss value for monitoring
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")
    return model

def evaluate(model, test_dataloader):
    '''
    Evaluate the model with the given test_dataloader.
    Returns the accuracy'
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total

def main():
    dataset = CLIPEmbeddedCaltech101(model_name = model_name, 
                                     root = data_path,
                                     load_embeddings = 'image_embeddings.pt', 
                                     download = True)
    #split into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    model = CLIPEmbeddingClassifier(model_name=model_name, class_names=dataset.categories)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = : ", device)
    model = model.to(device)
    model = train(model, train_dataloader, n_epochs=20)
    train_accuracy = evaluate(model, train_dataloader)
    print(f"Train Accuracy: {train_accuracy}")
    test_accuracy = evaluate(model, test_dataloader)
    print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
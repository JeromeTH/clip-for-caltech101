#Load the Caltech 101 dataset, add a prediction head on the embedding model of clip, 
# set contrastive learning rates on pretrained model and embedding head, and train the model on the dataset.
import torchvision
from torchvision.datasets import Caltech101
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from lightning import Trainer, DataModule
from torch.utils.data import Subset

pretrained, preprocess = clip.load("ViT-B/32")

train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     preprocess])

test_augs = preprocess


class CLIPImageClassifier(nn.Module):
    def __init__(self, pretrained, num_classes, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super(CLIPImageClassifier, self).__init__()
        self.pretrained = pretrained
        self.fc = nn.Linear(self.pretrained.visual.output_dim, num_classes)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.pretrained.encode_image(x)
        x = self.fc(x)
        return x
    
    def configure_optimizers(self, learning_rate = 0.01):
        params_1x = [param for name, param in self.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                   {'params': self.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
        
        return {'optimizer': optimizer,
                'loss': nn.CrossEntropyLoss()}


    


def load_caltech101(train_augs, test_augs, batch_size):
    dataset = Caltech101(root='.', target_type = "category", transform= train_augs, download=True)
    #generate random indices to be the training set * 0.8, will split using SubSet later to be the training set 
    random_perm = torch.randperm(len(dataset))
    training_indices = random_perm[:int(0.8 * len(dataset))]  # 80% for training
    testing_indices = random_perm[int(0.8 * len(dataset)):]  # 20% for testing
    train_dataset = Subset(dataset, training_indices)
    test_dataset = Subset(dataset, testing_indices) 
    train_dataset.transform = train_augs
    test_dataset.transform = test_augs
    return train_dataset, test_dataset


def main():
    train_dataset, test_dataset = load_caltech101(train_augs, test_augs, batch_size=32)
    model = CLIPImageClassifier(pretrained, num_classes=101)
    # print(model.named_parameters())
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    print(preprocess.__str__())
    print(train_dataset.__getitem__(0))
    # data = DataModule(train_dataset, test_dataset, batch_size=32)
    # trainer = Trainer(max_epochs=10, device=model.device)
    # trainer.fit(model, data)
    # trainer.evaluate()

if __name__ == "__main__":
    main() 
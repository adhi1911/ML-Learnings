import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# network architecture
class CNN(nn.Module):

    def __init__(self, in_channels = 1, num_classes = 10):
        super(CNN,self).__init__()

        self.input_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels = self.input_channels, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        # shorter way to write same: Conv2d(1,8,3,1,1)  # in_channels, out_channels, kernel_size, stride, padding
        # output dim will be [batch_size, 8, 28, 28] because padding = 1 and stride = 1 considering image size as 28x28
        # using formula the output size = (W - F + 2P)/S + 1 = (28 - 3 + 2*1)/1 + 1 = 28
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))  # max pooling layer
        # output = [batch_size, 8, 14, 14] because kernel size = 2 and stride = 2. Using formula: (W - F)/S + 1 = (28 - 2)/2 + 1 = 14
        self.conv2 = nn.Conv2d(8,16,3,1,1)  
        # output = [batch_size, 16, 14, 14] because padding = 1 and stride = 1. Using formula: (W - F + 2P)/S + 1 = (14 - 3 + 2*1)/1 + 1 = 14
        self.fc1 = nn.Linear(16*7*7, num_classes)  # fully connected layer. input size = 16*7*7 because after conv2 and pooling we will have output of size [batch_size, 16, 7, 7]
 
    def forward(self, X):

        X = F.relu(self.conv1(X))       # [32,1,28,28] --conv(1,8)-> [32,8,28,28]
        X = self.pool(X)                # [32,8,28,28] --maxpool(2,2)-> [32,8,14,14]
        X = F.relu(self.conv2(X))       # [32,8,14,14] --conv(8,16)-> [32,16,14,14]
        X = self.pool(X)                # [32,16,14,14] --maxpool(2,2)-> [32,16,7,7]
        X = X.reshape(X.shape[0], -1)   # [32,16,7,7] --flatten-> [32,784]
        X = self.fc1(X)                 # [32,784] --fc(784,10)-> [32,10]

        return X


model = CNN(in_channels=1, num_classes=10)
x = torch.randn(32,1,28,28)
print(model(x).shape)

# set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
print(device)  # check if GPU is available

# hyperparams
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 10


# data loading 
train_D = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_D, batch_size=batch_size, shuffle=True)

test_D = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_D, batch_size=batch_size, shuffle=True)


# init network 
cnn = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)  # optimizer

# training 

for epoch in range(num_epochs):

    for batch_idx, (data, targets) in enumerate(train_loader): 

        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], in_channels, 28, 28)  # reshape data to [batch_size, in_channels, height, width]

        # forward 
        scores = cnn(data)  # get scores from the model
        loss = criterion(scores, targets)  # calculate loss

        # backward 
        optimizer.zero_grad()  # zero the gradients
        loss.backward()  # backpropagation

        # gd step 
        optimizer.step()  # update weights

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")  # print loss for each epoch



def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data ")
    else:
        print("Checking accuracy on test data ")

    num_correct = 0
    num_samples = 0

    model.eval()  # set model to evaluation mode

    with torch.no_grad(): 
        for x,y in loader: 
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], in_channels, 28, 28)  # reshape data to [batch_size, in_channels, height, width]

            scores = model(x)  # get scores from the model
            _, predictions = scores.max(1)  # get the index of the max log-probability

            num_correct += (predictions == y).sum()  # count correct predictions
            num_samples += predictions.size(0)  # count total samples

        acc = float(num_correct)/float(num_samples)  # calculate accuracy
        print(f'Got {num_correct}/{num_samples} with accuracy {acc*100:.2f}%')

    model.train()  


check_accuracy(train_loader, cnn)
check_accuracy(test_loader, cnn)



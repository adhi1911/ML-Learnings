# import packages
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms  # contains transformations we can perform on our data


# print(torch.cuda.is_available())  # check if GPU is available

# create fully connected network 
class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(input_size, 50) 
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X


# model = NN(input_size=784, num_classes=10)
# x = torch.randn(64,784)
# print(model(x).shape) 
## expected ([64,10]) i.e. 64 rows and 10 columns (one for each class)
# print(model(x)) 


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)  


# hyperparams
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64 
num_epochs = 10


# load data 
train_D = datasets.MNIST(root='dataset/', train=True, transform= transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_D, batch_size=batch_size, shuffle=True)

test_D = datasets.MNIST(root='dataset/', train=False, transform= transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_D, batch_size=batch_size, shuffle=True)

# initialize network 
model = NN(input_size=input_size, num_classes=num_classes).to(device)
    # so input required will be [#examples, 784] and output will be [#examples, 10] (one for each class)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
    # just create instance whatever loss function you want to use, and then pass the model parameters to the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # instance of optimizer we want. useful parameters are model.parameters() to update the weights and learning rate
    # additonally you can add weight decay to prevent overfitting, momentum, etc.
    # eg. optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # weight decay is L2 regularization

# training loop
for epoch in range(num_epochs): 
    for batch_idx, (data,targets) in enumerate(train_loader):
        print(f'epoch {epoch+1}/{num_epochs} batch {batch_idx+1}/{len(train_loader)}')  # print progress

        # move data to device
        data = data.to(device=device)
        targets = targets.to(device=device)

        # data shape here is [64,1,28,28] (batch_size, channels, height, width)
        # we need to flatten it to [64,784] (batch_size, input_size)
        data = data.reshape(data.shape[0], -1)  # or reshape(batch_size, 784) 

        # forward pass 
        scores = model(data)  # this will return [64,10] (batch_size, num_classes) , results of one iteration of forward pass
        loss = criterion(scores, targets)   # loss from forward pass

        # backward pass
        optimizer.zero_grad()  # set gradients to zero before backward pass
        loss.backward()  # compute gradients

        # gradient step
        optimizer.step()  # update weights


# check accuracy on training and test to see how good our model is
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    num_correct = 0
    num_samples = 0


    model.eval()  # set model to evaluation mode

    with torch.no_grad():  # no grad computing needed now 
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)  # flatten the input

            scores = model(x)  # forward pass
            # scoes : [64,10] . We want max of second dim.
            _, predictions = scores.max(1)  # returns max value and index of max value. we want index of max value
            num_correct += (predictions == y).sum()  # add number of correct predictions
            num_samples += predictions.size(0)  # add number of samples in this batch

        acc = float(num_correct)/float(num_samples)  # accuracy = correct predictions / total samples
        print(f'Got {num_correct}/{num_samples} with accuracy {acc*100:.2f}%')

    model.train() # set model back to training mode
    return acc 

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
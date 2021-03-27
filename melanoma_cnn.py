### CHANGE
import pdb
import torch
import torchvision
import torchvision.transforms as transforms
## import tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

### CHANGE ###
data_path = 'DermMel\\train'
transformations = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.RandomHorizontalFlip(p=0.5),           #resizing all of the pictures
    torchvision.transforms.ToTensor()
    ## other transformations? such as horizonal or vertical random flips
])
trainset = torchvision.datasets.ImageFolder(                #loading all of the data into the right place
    root=data_path,
    transform=transformations
)  #transforms images into a tensor
##############
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,           #8 data points at a time (every time it loads)
                                          shuffle=True, num_workers=0)

### CHANGE ###
data_path = 'DermMel\\test'
validset = torchvision.datasets.ImageFolder(
    root=data_path,
    transform=transformations
)
##############
validloader = torch.utils.data.DataLoader(validset, batch_size=10,
                                         shuffle=True, num_workers=0)



import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120) ## CHANGE
        self.fc2 = nn.Linear(120, 2)
        self.fc3 = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53) ## CHANGE
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net() #creates neural network

import torch.optim as optim

criterion = nn.CrossEntropyLoss() # cost function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5) #gradient descent
## tensorboard --logdir=runs
writer = SummaryWriter('runs/OT_trialrunoftransformation') #creates and labels graph for organization
iterations = 1
for epoch in range(50):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0): #enumerate creates trainloader into a "loopable" object #0 keeps count
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) #labels the images through the neural network
        loss = criterion(outputs, labels) #taking the cost function of the outputs and labels to compare
        loss.backward()
        optimizer.step() #taking the gradient descent
        writer.add_scalar('training loss', loss, iterations) #plots the iterations and loss on the graph as it is training

        y_true = labels.numpy()
        y_pred = torch.argmax(outputs, dim=1).numpy() # next week
        accuracy = accuracy_score(y_true, y_pred, normalize=False)  #calculating the accuracy
        writer.add_scalar('training accuracy', accuracy, iterations)

        iterations += 1 #adding 1 to each iteration to help keep track
        print(loss)



        if iterations % 50 == 0:
            ## load a batch of new data (validation data)
            inputs_valid, labels_valid = iter(validloader).next()
            outputs_valid = net(inputs_valid)
            loss_valid = criterion(outputs_valid, labels_valid)
            writer.add_scalar('validation loss', loss_valid, iterations)
            y_true = labels_valid.numpy()
            y_pred = torch.argmax(outputs_valid, dim=1).numpy()
            val_accuracy = accuracy_score(y_true, y_pred, normalize=False)
            writer.add_scalar('valication accuracy',  val_accuracy, iterations)

            ## put the prediction error on new data into the tensorboard


writer.close() #closes out of the graph
print('Finished Training')




import PIL  

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt

import time

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train = True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root="./data", train = False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv00 = nn.Conv2d(3,8,1)
    self.bn01 = nn.BatchNorm2d(8)
    self.conv02 = nn.Conv2d(8,24 , 1)
    self.bn03 = nn.BatchNorm2d(24)
    self.conv1 = nn.Conv2d(24, 64, 3,padding = 1) 
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 64, 3, padding = 1) 
    self.conv3 = nn.Conv2d(64, 64, 3, padding = 1) 
    self.pool = nn.MaxPool2d(2, 2)  
    self.conv4 = nn.Conv2d(64, 128, 3, padding = 1) 
    self.bn2 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 128, 3, padding = 1) 
    self.conv6 = nn.Conv2d(128, 128, 3, padding = 1)
    self.conv7 = nn.Conv2d(128, 256, 3, padding = 1) 
    self.conv8 = nn.Conv2d(256, 256, 3, padding = 1) 
    self.conv9 = nn.Conv2d(256, 256, 3, padding = 1)
    self.bn3 = nn.BatchNorm2d(256)
    self.conv10 = nn.Conv2d(256,100,1)
    self.conv11 = nn.Conv2d(100, 10, 4)

  def forward(self, x):
    
    x = self.bn03(F.relu(self.conv02(self.bn01(F.relu(self.conv00(x))))))    

    x = F.relu(self.conv1(x))#32*32
    x = self.bn1(x)
    x = F.relu(self.conv2(x))

    x = self.pool(F.relu(self.conv3(x)))#16*16
    x = F.relu(self.conv4(x))
    x = self.bn2(x)
    x = F.relu(self.conv5(x))
               
    x = self.pool(F.relu(self.conv6(x)))#8*8
    x = F.relu(self.conv7(x))
    x = self.bn3(x)
    x = F.relu(self.conv8(x))
               
    x = self.pool(F.relu(self.conv9(x)))#4*4
    x = self.bn3(x)
    x = F.relu(self.conv10(x))
    x = self.conv11(x)
    x = x.view(-1, 10)

    return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
net = Net()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay = 0.008)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
  

train_log = []
test_log=[]
test_acc=[]

path = "log.txt"
with open(path, mode = "w") as f:
    f.write("start\n")
for epoch in range(100):

    t1 = time.time()
    total_train_loss = 0.0
    total_test_loss = 0.0
    
    total = 0
    for i, data in enumerate(trainloader, 0):
        #get input
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
    
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        if i == 0:
          scheduler.step()
        else:
          optimizer.step()
        #scheduler.step()

        total_train_loss += loss.item()
        total += labels.size(0)        
    with open(path, mode = "a") as f:	 
        f.write("---------------------------------\n")
        f.write("epoch :{}  train loss :{:.4g}\n".format(epoch+1,loss.item()))      
    print("---------------------------------")
    print("epoch :",epoch+1, "train loss :","{:.4g}".format(loss.item()))      
    train_log.append(loss.item())
    correct = 0
    total = 0
    with torch.no_grad():
      for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).cpu().sum().item()

      test_log.append(loss.item())
      print("epoch :",epoch+1, "test  loss :","{:.4g}".format(loss.item()))
      print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))   
      print("{0} / {1}".format(correct , total))
      
      test_acc.append(int(100 * correct/total))
    with open(path, mode = "a") as f:	 
        f.write("epoch :{}  test  loss :{:.4g}\n".format(epoch+1,loss.item()))      
        f.write('Accuracy of the network on the 10000 test images: %d %%\n' % (100 * correct / total))   
        f.write("{0} / {1}\n".format(correct , total))
        f.write("time: {:.2g} [sec] \n".format(time.time() - t1))  
        f.write("---------------------------------\n")
      

print("finish")

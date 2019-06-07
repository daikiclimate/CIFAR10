import PIL  
import image

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib as plt

#data initial setting
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#load trainset and testset

trainset = torchvision.datasets.CIFAR10(root="./data", train = True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root="./data", train = False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

import matplotlib.pyplot as plt
import numpy as np

#if you want to show picture and labels
def imshow(img):
  img = img/2 + 0.5
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

#get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

#show images
#imshow(torchvision.utils.make_grid(images))

#print label
#print("".join("%5s" % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3) 
    self.bn1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 64, 3) 
    self.do1 = nn.Dropout2d(p=0.2) 
    self.conv3 = nn.Conv2d(64, 64, 3) 
    self.pool = nn.MaxPool2d(2, 2)  
    self.conv4 = nn.Conv2d(64, 128, 3) 
    self.bn2 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 128, 3) 
    self.conv6 = nn.Conv2d(128, 128, 3) 
    self.conv7 = nn.Conv2d(128, 256, 3) 
    self.conv8 = nn.Conv2d(256, 256, 3) 
    self.conv9 = nn.Conv2d(256, 256, 3) 

    #instead of FC layer
    self.conv10 = nn.Conv2d(256,100,1)
    self.bn3 = nn.BatchNorm2d(256)
    self.conv11 = nn.Conv2d(100, 10, 4)

    #zero padding
    self.zeropad = nn.ConstantPad2d(1, 0)

  def forward(self, x):
    x = F.relu(self.conv1(self.zeropad(x)))#32*32
    x = self.bn1(x)
    x = F.relu(self.conv2(self.zeropad(x)))
    x = self.do1(x)
    x = self.pool(F.relu(self.conv3(self.zeropad(x))))#16*16
    x = F.relu(self.conv4(self.zeropad(x)))
    x = self.bn2(x)
    x = self.do1(x)
    x = F.relu(self.conv5(self.zeropad(x)))
    x = self.pool(F.relu(self.conv6(self.zeropad(x))))#8*8
    x = F.relu(self.conv7(self.zeropad(x)))
    x = self.do1(x)
    x = F.relu(self.conv8(self.zeropad(x)))
    x = self.pool(F.relu(self.conv9(self.zeropad(x))))#4*4
    x = self.bn3(x)
    x = self.do1(x)
    x = F.relu(self.conv10(x))
    x = self.conv11(x)
    x = x.view(-1, 10)

    return x
#GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#optimizer and criterion
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
  

train_log = []
test_log=[]
test_acc=[]
for epoch in range(100):

    running_loss = 0.0
    total_train_loss = 0.0
    total_test_loss = 0.0
    total = 0

    for i, data in enumerate(trainloader, 0):
        #get input data
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
        
        running_loss += loss.item()
        total_train_loss += loss.item()
        total += labels.size(0)
        
        if i % 1500 == 1499:
            print("[%d, %5d] loss: %.3f" % (epoch +1, i+1, running_loss/ 1500))
            running_loss = 0.0
    #keep training loss log 
    train_log.append(loss.item())

    #keep validation loss log
    correct = 0
    total = 0
    with torch.no_grad():
      for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = net(images)
        loss = criterion(outputs, labels)

        #total validation loss
        total_test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #accurasy
        correct += (predicted == labels).cpu().sum().item()

      #validation loss log
      test_log.append(loss.item())
      print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))   
      print("{0} / {1}".format(correct , total))
      test_acc.append(int(100 * correct/total))

print("finish")

#loss changes
plt.plot(range(len(test_log)), test_log, label ="validation loss")
plt.plot(range(len(train_log)), train_log, label = "train loss")
plt.legend()
plt.show()

#accuracy changes
plt.figure()
plt.plot(range(len(test_acc)), test_acc, label = "test acc")
plt.legend()
plt.show()

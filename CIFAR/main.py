'''Test CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cpu' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#net = SimpleDLA()
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Load checkpoint.
print('==> Resuming from pretrained model..')
assert os.path.isdir('pretrained'), 'Error: no pretrained directory found!'
checkpoint = torch.load('./pretrained/resnet18.pth', map_location=device)
if device == 'cpu':
    a = checkpoint['net'].copy()
    for key in a.keys():
        checkpoint['net'][key[7:]] = checkpoint['net'].pop(key)


net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return 100.*correct/total


# get test accuracy
acc = test()

# read energy
f=open('./results/energy.txt','r')
lines = np.genfromtxt('./results/energy.txt', delimiter=',')

reading=[]
for i in range(len(lines)):
    reading.append(lines[i][0])

energy= np.sum(reading)
f.close()

# write result file
f=open('./results/result.txt','a')
np.savetxt(f, [acc], fmt='%1.2f', newline=", ")
f.write("\n")
np.savetxt(f, [int(energy)], fmt='%i', newline=", ")
f.write("\n")
f.close()

print('Accuracy: ' + str(acc) + '%\n' + 'Energy: ' + str(int(energy)) + '\n')


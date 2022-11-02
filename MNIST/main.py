import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np
import random
import os

import models

from torchvision import datasets, transforms
from functions import *

global test_loss
global test_best_accs


cuda_act = True

if cuda_act == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

def test():
    """Evaluate the neural network accuracy prediction"""
    
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    # load data and target
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # inference
        output = model(data)

        # calculate loss
        test_loss += criterion(output, target).data.item()

        # predict output
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    # track accuracy
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))


    return 100. * float(correct) / len(test_loader.dataset)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every lr_epochs"""
    
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__=='__main__':
    
    # parser settings
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure')
    args = parser.parse_args()

    print(args)
    
    # setting up random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if cuda_act:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # load MNIST dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_act else {}
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5()
    else:
        print('ERROR: specified arch is not supported')
        exit()    

    # load pretrained model
    pretrained_model = torch.load('./pretrained/LeNet_5.pth.tar')
    best_acc = pretrained_model['acc']
    model.load_state_dict(pretrained_model['state_dict'])

    if cuda_act:
        model.cuda()

    # define training hyperparameters
    print(model)
    param_dict = dict(model.named_parameters())
    params = []

    count_parameters(model)
    
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.lr,
            'weight_decay': args.weight_decay,
            'key':key}]
    
    optimizer = optim.Adam(params, lr=args.lr,
            weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # check if energy is deleted
    if os.path.exists('./results/energy.txt'):
        print('Energy file exists')

    # get test accuracy
    acc = test()

    # read energy
    f=open('./results/energy.txt','r')
    lines = np.genfromtxt('./results/energy.txt', delimiter=',')

    reading=[]
    for i in range(len(lines)):
        reading.append(lines[i][0])
    if len(lines) != 10000:
        print(len(lines))
        print('Double check the number of lines in energy')
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


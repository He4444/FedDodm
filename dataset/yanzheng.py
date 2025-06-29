import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


def separate_data1(data):
    num_classes = 10
    num_clients = 2
    class_per_client = 5000
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    idxs = np.array(range(len(dataset_label)))
    idx_for_each_class = []
    for i in range(num_classes):
        idx_for_each_class.append(idxs[dataset_label == i])

    class_num_per_client = [class_per_client for _ in range(num_clients)]
    for i in range(num_classes):
        selected_clients = []
        for client in range(num_clients):
            if class_num_per_client[client] > 0:
                selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients/num_classes)*class_per_client))]

        num_all_samples = len(idx_for_each_class[i])
        num_selected_clients = len(selected_clients)
        num_per = num_all_samples / num_selected_clients

        num_samples = np.random.randint(max(num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
        num_samples.append(num_all_samples-sum(num_samples))

        idx = 0
        for client, num_sample in zip(selected_clients, num_samples):
            if client not in dataidx_map.keys():
                dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
            else:
                dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
            idx += num_sample
            class_num_per_client[client] -= 1

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic







dir_path = "Cifar10/"

config_path = dir_path + "config.json"


# Get Cifar10 data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root=dir_path + "rawdata", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root=dir_path + "rawdata", train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=len(trainset.data), shuffle=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=len(testset.data), shuffle=False)

for _, train_data in enumerate(trainloader, 0):
    trainset.data, trainset.targets = train_data
for _, test_data in enumerate(testloader, 0):
    testset.data, testset.targets = test_data

dataset_image = []
dataset_label = []

dataset_image.extend(trainset.data.cpu().detach().numpy())
dataset_image.extend(testset.data.cpu().detach().numpy())
dataset_label.extend(trainset.targets.cpu().detach().numpy())
dataset_label.extend(testset.targets.cpu().detach().numpy())
dataset_image = np.array(dataset_image)
dataset_label = np.array(dataset_label)


X, y, statistic = separate_data1((dataset_image, dataset_label))
train_data, test_data = split_data(X, y)



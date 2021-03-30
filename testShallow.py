import numpy as np

import SpykeTorch.utils as utils
import SpykeTorch.functional as sf
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random

kernels = [utils.GaborKernel(window_size=3, orientation=45 + 22.5),
           utils.GaborKernel(3, 90 + 22.5),
           utils.GaborKernel(3, 135 + 22.5),
           utils.GaborKernel(3, 180 + 22.5)]
filter = utils.Filter(kernels, use_abs=True)


def time_dim(input):
    return input.unsqueeze(0)

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     time_dim,
     filter,
     sf.pointwise_inhibition,
     utils.Intensity2Latency(number_of_spike_bins=30, to_spike=True)])


dataset = ImageFolder("D:\PyCharm 2020.3.5\pythonProject\dataset\eth", transform)  # adding transform to the dataset

# splitting training and testing sets
indices = list(range(len(dataset)))
random.shuffle(indices)
split_point = int(0.75*len(indices))
train_indices = indices[:split_point]
test_indices = indices[split_point:]
print("Size of the training set:", len(train_indices))
print("Size of the  testing set:", len(test_indices))
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

dataset = utils.CacheDataset(dataset)
train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices))

import SpykeTorch.snn as snn

pool = snn.Pooling(kernel_size = 3, stride = 2)
conv = snn.Convolution(in_channels=4, out_channels=20, kernel_size=30)
stdp = snn.STDP(conv_layer = conv, learning_rate = (0.05, -0.015))
for iter in range(300):
    print('\rIteration:', iter, end="")
    for data,_ in train_loader:
        for x in data:
            x = pool(x)
            p = conv(x)
            o, p = sf.fire(p, 20, return_thresholded_potentials=True)
            winners = sf.get_k_winners(p, kwta=1, inhibition_radius=0, spikes=o)
            stdp(x, p, o, winners)
print()
print("Unsupervised Training is Done.")

#######################################################
train_x_spike = []
train_x_pot = []
train_y = []
for data,targets in train_loader:
    for x,t in zip(data, targets):
        x = pool(x)
        p = conv(x)
        o = sf.fire(p, 20)
        train_x_spike.append(o.reshape(-1).cpu().numpy())
        train_x_pot.append(p.reshape(-1).cpu().numpy())
        train_y.append(t)
train_x_spike = np.array(train_x_spike)
train_x_pot = np.array(train_x_pot)
train_y = np.array(train_y)
test_x_spike = []
test_x_pot = []
test_y = []
for data,targets in test_loader:
    for x,t in zip(data, targets):
        x = pool(x)
        p = conv(x)
        o = sf.fire(p, 20)
        test_x_spike.append(o.reshape(-1).cpu().numpy())
        test_x_pot.append(p.reshape(-1).cpu().numpy())
        test_y.append(t)
test_x_spike = np.array(test_x_spike)
test_x_pot = np.array(test_x_pot)
test_y = np.array(test_y)

from sklearn.svm import LinearSVC

clf_spike = LinearSVC(max_iter=100000)
clf_pot = LinearSVC(max_iter=100000)
clf_spike.fit(train_x_spike, train_y)
clf_pot.fit(train_x_pot, train_y)

predict_spike = clf_spike.predict(test_x_spike)
predict_pot = clf_pot.predict(test_x_pot)

error_spike = np.abs(test_y - predict_spike).sum()
error_pot = np.abs(test_y - predict_pot).sum()
print("    Spike-based error:", error_spike/len(predict_spike))
print("Potential-based error:", error_pot/len(predict_pot))
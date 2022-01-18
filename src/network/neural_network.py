# This file contains the following:
# 
# - code that converts pytorch model into a neural network
#   layer representation executable by  private inference methods
#   (just a list of "Layer" types)
#
# - code that executes the network in the given layer representation
#   (to be used for sanity checking output correctness)
#
# - code that evaluates a pretrained network on datasets (mnist, etc)

import sys
import pprint
import time
from collections import OrderedDict
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import math
from train.mnist.mnist import MnistNet, test as MnistTest
from train.mnist.mnist_bn import MnistBNNet
from train.resnet.pytorch_resnet_cifar10.resnet import *
from train.resnet.pytorch_cifar100.models.resnet import *
from train.resnet.pytorch_cifar100.models.vgg import *
from torchvision import datasets, transforms
import git
import torch.nn.functional as F

# Helper for finding pretrained model files
def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


class Layer(object):
    def __init__(self, name):
        self.name = name
        self.extra = {}

        # Tracks intermediate outputs
        self.output = None
        self.loaded = 0

    def __repr__(self):
        return "%s" % self.name

    def load(self, pytorch_module):
        self.pytorch_module = pytorch_module

    def clear(self):
        # Clear any revealing weight data of the network
        # Note, does not hide architecture.
        raise Exception("Not implemented")

class FCLayer(Layer):
    def __init__(self, name):
        super().__init__(name)

        self.weight = None
        self.bias = None        

    def load(self, pytorch_module):
        super().load(pytorch_module)
        self.weight = pytorch_module.weight.detach().numpy()
        self.weight_mat = self.weight.T
        if pytorch_module.bias is not None:
            self.bias = pytorch_module.bias.detach().numpy()
        else:
            self.bias = None
        self.loaded = 1

    def clear(self):
        if self.bias is not None:
            self.bias = np.zeros_like(self.bias)
        if self.pytorch_module.bias is not None:
            self.pytorch_module.bias = torch.nn.Parameter(torch.zeros_like(self.pytorch_module.bias))
            
        self.weight = np.zeros_like(self.weight)
        self.weight_mat = np.zeros_like(self.weight_mat)
        self.pytorch_module.weight = torch.nn.Parameter(torch.zeros_like(self.pytorch_module.weight))

    def __repr__(self):
        return "%s {shape: %s} loaded=%d" % (self.name, self.weight.T.shape, self.loaded)

class AvgPoolLayer(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.kernel_size = None
        
    def load(self, pytorch_module):
        super().load(pytorch_module)        
        self.kernel_size = pytorch_module.kernel_size
        self.loaded = 1

    def clear(self):
        pass

    def __repr__(self):
        return "%s {kernel_size: %s} loaded=%d" % (self.name, self.kernel_size, self.loaded)

class IdentityLayer(Layer):
    def __init__(self, name, indx_identity, conv_block=None):
        super().__init__(name)
        self.indx_identity = indx_identity

    def clear(self):
        pass

    def __repr__(self):
        return "%s {from layer: %s}" % (self.name, self.indx_identity)

class ConvLayer(Layer):
    def __init__(self, name, merge_bn=None):
        super().__init__(name)
        self.weight = None
        self.bias = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.unfold = None
        self.merge_bn = merge_bn

    def load(self, pytorch_module):
        super().load(pytorch_module)        
        self.weight = pytorch_module.weight.detach().numpy()
        if pytorch_module.bias is not None:
            self.bias = pytorch_module.bias.detach().numpy()
            self.bias_mat = self.bias.reshape((1, self.bias.shape[0], 1))
        else:
            self.bias = None
            self.bias_mat = None

        self.kernel_size = pytorch_module.kernel_size
        
        assert(pytorch_module.stride[0] ==
               pytorch_module.stride[1])
        self.stride = pytorch_module.stride[0]
        
        assert(pytorch_module.padding[0] ==
               pytorch_module.padding[1])
        self.padding = pytorch_module.padding[0]

        # Ops for execution
        self.unfold = torch.nn.Unfold(self.kernel_size,
                                      padding=self.padding,
                                      stride=self.stride)

        self.weight_mat = self.weight.reshape(self.weight.shape[0], -1).T
        self.loaded = 1

    def clear(self):
        if self.bias is not None:
            self.bias = np.zeros_like(self.bias)
        if self.pytorch_module.bias is not None:
            self.pytorch_module.bias = torch.nn.Parameter(torch.zeros_like(self.pytorch_module.bias))
        
        self.weight = np.zeros_like(self.weight)
        self.weight_mat = np.zeros_like(self.weight_mat)
        self.pytorch_module.weight = torch.nn.Parameter(torch.zeros_like(self.pytorch_module.weight))

    def __repr__(self):
        return "%s {kernel_size=%s, stride=%s, padding=%s, merge_bn=%d} loaded=%d" % (self.name,
                                                                                      self.kernel_size,
                                                                                      self.stride,
                                                                                      self.padding,
                                                                                      self.merge_bn is not None,
                                                                                      self.loaded)

class FlattenLayer(Layer):
    def __init__(self, name="flatten"):
        super().__init__(name)

    def clear(self):
        pass

class ReLULayer(Layer):
    def __init__(self, name="ReLU"):
        super().__init__(name)

    def clear(self):
        pass

class PushInputsToStack(Layer):
    # This layer is a hack to get resnet-34 on cifar100 shortcut connections working
    # (as shortcut connection preceded by conv which branches off from the main work).
    # Layer indicates to push the current inputs onto the stack
    def __init__(self, name="push"):
        super().__init__(name)

    def clear(self):
        pass

class PopInputsFromStack(Layer):
    def __init__(self, name="pop"):
        super().__init__(name)

    def clear(self):
        pass

def print_layers(layers):
    print("Layer Architecture {")
    for l in layers:
        print("\t" + str(l))
    print("}")

# Fuse pytorch conv and bn. Source: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
def fuse_conv_and_bn(conv, bn):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) )
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_( b_conv + b_bn )
	#
	# we're done
	return fusedconv    
        
def load_neural_network_from_pytorch(pytorch_model, layers, mute=False):
    # Extracts network weights from pytorch model and layer ordering (user defined)
    # - Exclude Softmaxs
    # - Fold batchnorm into conv (todo)
    # : Output should contain only
    #   - FCs
    #   - ReLUs
    #   - Convs
    #   - AvgPool
    #   - Skip connections (todo)    

    supported_pytorch_layer_types = [
        # type(torch.nn.ReLU()), ReLU has no weight/param data associated        
        torch.nn.Linear,
        torch.nn.Conv2d,
        torch.nn.AvgPool2d,
    ]

    # Index for layers
    name_to_index = {}
    for i,l in enumerate(layers):
        name_to_index[l.name] = i

    # Populate the layers weight data
    for name, module in pytorch_model.named_modules():

        if type(module) not in supported_pytorch_layer_types:
            continue

        if name not in name_to_index:
            continue
        layer = layers[name_to_index[name]]
        layer.load(module)

        # Handle merged batchnorm with conv
        if type(layer) == ConvLayer and layer.merge_bn is not None:
            
            # Find bn referenced by conv and fuse them
            with torch.no_grad():
                fused = fuse_conv_and_bn(module, layer.merge_bn)
                layer.load(fused)

    if not mute:
        print_layers(layers)
        
    return layers

def execute_neural_network(layers, x, timing_breakdown={}, misc={}):
    initial = x

    stack = []

    misc["shapes"] = {}

    tot_act_size = 0
    
    # Execute a neural network
    # For purposes of sanity checking and debugging
    for i, layer in enumerate(layers):

        misc["shapes"][i] = x.shape

        if i != 0:
            tot_act_size += np.prod(x.shape)
        
        t_start = time.time()
        if type(layer) == PushInputsToStack:
            stack.append(x)
        elif type(layer) == PopInputsFromStack:
            x = stack.pop(-1)
        elif type(layer) == FCLayer:
            x = x.dot(layer.weight_mat) 
            if layer.bias is not None:
                x = x + layer.bias
        elif type(layer) == ReLULayer:
            x[x<0] = 0
        elif type(layer) == ConvLayer:
            x = layer.pytorch_module(torch.from_numpy(x).type(layer.pytorch_module.weight.dtype)).detach().numpy()
        elif type(layer) == AvgPoolLayer:
            x = layer.pytorch_module(torch.from_numpy(x)).numpy()
        elif type(layer) == FlattenLayer:
            x = x.reshape((x.shape[0], -1))
        elif type(layer) == IdentityLayer:
            other = initial if layer.indx_identity == -1 else layers[layer.indx_identity].output

            if other.shape != x.shape:

                # This happens with cifar10 resnet
                p = (x.shape[1]-other.shape[1])//2
                other = np.pad(other[:,:,::2,::2], ((0,0), (p,p), (0,0), (0,0)), mode="constant", constant_values=0)
                assert(other.shape == x.shape)
                
            x = x + other
        else:
            raise Exception("execute_neural_network: Unknown type: %s" % type(layer))

        layer.output = x
        
        t_end = time.time()
        elapsed = t_end-t_start
        if str(layer) not in timing_breakdown:
            timing_breakdown[str(layer)] = 0
        timing_breakdown[str(layer)] += elapsed

    return x
    
####### Tests #######

def mean_err(x, y):
    return np.mean(np.abs(x-y))

def test_run_fc_network():
    class FCNet(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.L1 = nn.Linear(28*28, hidden_size)
            self.L2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.L3 = nn.Linear(hidden_size, 10)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.L1(x)
            x = self.relu(x)
            x = self.L2(x)
            x = self.relu(x)
            x = self.L3(x)
            return x

        
    layers = [
        FCLayer("L1"),
        ReLULayer("ReLU"),
        FCLayer("L2"),        
        ReLULayer("ReLU"),
        FCLayer("L3")
    ]

    net = FCNet(128)
    net.eval()    
    layers = load_neural_network_from_pytorch(net, layers)

    # Test batch != 1
    for b in [1, 16, 64, 128]:
        x = np.random.normal(0, 10, size=(b, 28*28))
        result = execute_neural_network(layers, x)
        golden = net(torch.from_numpy(x).float()).detach().numpy()

        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-6)

    print("Pass FCnet")        

def test_run_conv_network():
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.AvgPool2d(2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.AvgPool2d(2)
            self.fc1 = nn.Linear(256, 120)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
            self.relu4 = nn.ReLU()
            self.fc3 = nn.Linear(84, 10)
            self.relu5 = nn.ReLU()            

        def forward(self, x):
            y = self.conv1(x)
            y = self.relu1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.relu2(y)
            y = self.pool2(y)
            y = y.view(y.shape[0], -1)
            y = self.fc1(y)
            y = self.relu3(y)
            y = self.fc2(y)
            y = self.relu4(y)
            y = self.fc3(y)
            y = self.relu5(y)
            return y            

        
    layers = [
        ConvLayer("conv1"),
        ReLULayer(),
        AvgPoolLayer("pool1"),

        ConvLayer("conv2"),
        ReLULayer(),
        AvgPoolLayer("pool2"),

        FlattenLayer("flatten"),

        FCLayer("fc1"),
        ReLULayer(),

        FCLayer("fc2"),
        ReLULayer(),

        FCLayer("fc3"),
        ReLULayer(),        
    ]

    net = ConvNet()    
    net.eval()
    layers = load_neural_network_from_pytorch(net, layers)

    # Test batch != 1
    for b in [1, 16, 64, 128]:
        # Recall (N, C, H, W) convolution format
        x = np.random.normal(0, 10, size=(b, 1, 28, 28))
        golden = net(torch.from_numpy(x).float()).detach().numpy()
        result = execute_neural_network(layers, x)

        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-6)

    print("Pass Convnet")

def test_mnist_pretrained():
    # See train/mnist/mnist.py for network architecture
        
    layers = [
        ConvLayer("conv1"),
        ReLULayer(),

        ConvLayer("conv2"),
        ReLULayer(),
        
        AvgPoolLayer("pool2d"),

        FlattenLayer("flatten"),

        FCLayer("fc1"),
        ReLULayer(),

        FCLayer("fc2"),
        
        # Note there is a softmax, but don't need for classification
    ]    

    # Model
    net = MnistNet()
    net.eval()
    pretrained_model_path = get_git_root(__file__) + "/src/train/mnist/mnist_cnn.pt"
    net.load_state_dict(torch.load(pretrained_model_path))
    layers = load_neural_network_from_pytorch(net, layers)

    # Dataset
    transform=transforms.Compose([
	transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])    
    dataset1 = datasets.MNIST('/tmp', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/tmp', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000)

    # Evaluate
    tstart = time.time()
    timing_breakdown = {}
    correct, total = 0, 0
    for data, target in test_loader:
        data = data.numpy()
        target = target.numpy()

        pred = execute_neural_network(layers, data, timing_breakdown)
        pred = np.argmax(pred, axis=1)
        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # comment for full
    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total
    assert(acc >= .97)

    pprint.pprint(timing_breakdown)    
    print("Pass mnist pretrained with acc=%f time=%f s" % (acc, telapsed))

def test_run_conv_batchnorm_network():
    
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.bn1 = nn.BatchNorm2d(6)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.AvgPool2d(2)

        def forward(self, x):
            y = self.conv1(x)
            y = self.bn1(y)
            #y = self.relu1(y)
            #y = self.pool1(y)
            return y            
        
    net = ConvNet()
    net.eval()

    layers = [
        # Merged ConvBNLayer
        ConvLayer("conv1", merge_bn=net.bn1)
        #ReLULayer(),
        #AvgPoolLayer("pool1"),
    ]    
    
    layers = load_neural_network_from_pytorch(net, layers)
    
    # Test batch != 1
    for b in [1, 16, 64, 128]:
        # Recall (N, C, H, W) convolution format
        x = np.random.normal(0, 10, size=(b, 1, 28, 28))
        golden = net(torch.from_numpy(x).float()).detach().numpy()
        result = execute_neural_network(layers, x)

        
        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-6)

    print("Pass Convnet Batchnorm")

def test_mnist_bn_pretrained():
    # See train/mnist/mnist_bn.py for network architecture
        
    # Model
    net = MnistBNNet()
    net.eval()

    layers = [
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer(),

        ConvLayer("conv2", merge_bn=net.bn2),
        ReLULayer(),
        
        AvgPoolLayer("pool2d"),

        FlattenLayer("flatten"),

        FCLayer("fc1"),
        ReLULayer(),

        FCLayer("fc2"),
        
        # Note there is a softmax, but don't need for classification
    ]        
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/mnist/mnist_bn_cnn.pt"
    net.load_state_dict(torch.load(pretrained_model_path))
    layers = load_neural_network_from_pytorch(net, layers)

    # Dataset
    transform=transforms.Compose([
	transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])    
    dataset1 = datasets.MNIST('/tmp', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('/tmp', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000)

    # Evaluate
    tstart = time.time()
    timing_breakdown = {}
    correct, total = 0, 0
    for data, target in test_loader:
        data = data.numpy()
        target = target.numpy()

        pred = execute_neural_network(layers, data, timing_breakdown)
        pred = np.argmax(pred, axis=1)
        correct += np.sum(pred == target)
        total += pred.shape[0]
        break # Comment for full
    
    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total
    assert(acc >= .97)

    pprint.pprint(timing_breakdown)    
    print("Pass mnist bn pretrained with acc=%f time=%f s" % (acc, telapsed))

def test_run_resnet_block():

    def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)
    
    class BasicBlock(nn.Module):
        expansion: int = 1

        def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
        ) -> None:
            super(BasicBlock, self).__init__()
            norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out += identity
            out = self.relu(out)

            return out        
    
    net = BasicBlock(32, 32)
    net.eval()
    layers = [
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer("ReLU"),
        ConvLayer("conv2", merge_bn=net.bn2),
        IdentityLayer("Identity", -1),
        ReLULayer("ReLU")
    ]
    layers = load_neural_network_from_pytorch(net, layers)

    for b in [1, 16, 32]:
        x = np.random.normal(0, 1, size=(b, 32, 32, 32))
        golden = net(torch.from_numpy(x).float()).detach().numpy()
        result = execute_neural_network(layers, x)

        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-6)

    print("Pass ResNet block")

def test_resnet20_sanity():
    net = torch.nn.DataParallel(resnet20())
    net.eval()

    #net = torch.nn.DataParallel(resnet32())
    #pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
    #save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    #net.load_state_dict(save_data["state_dict"])

    #print(net)    
    #for a,b in net.named_modules():
    #    print(a)

    layers = [
        ConvLayer("module.conv1", merge_bn=net.module.bn1),
        ReLULayer(),
        
        ConvLayer("module.layer1.0.conv1", merge_bn=net.module.layer1[0].bn1),
        ReLULayer(),
        ConvLayer("module.layer1.0.conv2", merge_bn=net.module.layer1[0].bn2),
        IdentityLayer("shortcut", 1),
        ReLULayer(),
        
        ConvLayer("module.layer1.1.conv1", merge_bn=net.module.layer1[1].bn1),
        ReLULayer(),        
        ConvLayer("module.layer1.1.conv2", merge_bn=net.module.layer1[1].bn2),
        IdentityLayer("shortcut", 6),
        ReLULayer(),        
        
        ConvLayer("module.layer1.2.conv1", merge_bn=net.module.layer1[2].bn1),
        ReLULayer(),        
        ConvLayer("module.layer1.2.conv2", merge_bn=net.module.layer1[2].bn2),
        IdentityLayer("shortcut", 11),
        ReLULayer(),        
        
        ConvLayer("module.layer2.0.conv1", merge_bn=net.module.layer2[0].bn1),
        ReLULayer(),                
        ConvLayer("module.layer2.0.conv2", merge_bn=net.module.layer2[0].bn2),
        IdentityLayer("shortcut", 16),
        ReLULayer(),                
        
        ConvLayer("module.layer2.1.conv1", merge_bn=net.module.layer2[1].bn1),
        ReLULayer(),                
        ConvLayer("module.layer2.1.conv2", merge_bn=net.module.layer2[1].bn2),
        IdentityLayer("shortcut", 21),
        ReLULayer(),                
        
        ConvLayer("module.layer2.2.conv1", merge_bn=net.module.layer2[2].bn1),
        ReLULayer(),                
        ConvLayer("module.layer2.2.conv2", merge_bn=net.module.layer2[2].bn2),
        IdentityLayer("shortcut", 26),
        ReLULayer(),                        
        
        ConvLayer("module.layer3.0.conv1", merge_bn=net.module.layer3[0].bn1),
        ReLULayer(),                
        ConvLayer("module.layer3.0.conv2", merge_bn=net.module.layer3[0].bn2),
        IdentityLayer("shortcut", 31),
        ReLULayer(),                
        
        ConvLayer("module.layer3.1.conv1", merge_bn=net.module.layer3[1].bn1),
        ReLULayer(),
        ConvLayer("module.layer3.1.conv2", merge_bn=net.module.layer3[1].bn2),
        IdentityLayer("shortcut", 36),
        ReLULayer(),                
        
        ConvLayer("module.layer3.2.conv1", merge_bn=net.module.layer3[2].bn1),
        ReLULayer(),                
        ConvLayer("module.layer3.2.conv2", merge_bn=net.module.layer3[2].bn2),
        IdentityLayer("shortcut", 41),
        ReLULayer(),
        
        AvgPoolLayer("module.pool"),
        FlattenLayer(),
        
        FCLayer("module.linear")
    ]

    layers = load_neural_network_from_pytorch(net, layers)

    # Load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/tmp', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/tmp', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        pin_memory=True)

    for i, (input, target) in enumerate(val_loader):
        target = target
        input_var = input
        golden = net(input_var).detach().numpy()
        result = execute_neural_network(layers, input_var.numpy())

        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-5)
        break

    print("Pass ResNet20 sanity")

def test_resnet32_cifar10_pretrained():
    net = torch.nn.DataParallel(resnet32())
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data["state_dict"])

    print(net)    
    for a,b in net.named_modules():
        print(a)

    layers = [
        ConvLayer("module.conv1", merge_bn=net.module.bn1),
        ReLULayer(),
        
        ConvLayer("module.layer1.0.conv1", merge_bn=net.module.layer1[0].bn1),
        ReLULayer(),
        ConvLayer("module.layer1.0.conv2", merge_bn=net.module.layer1[0].bn2),
        IdentityLayer("shortcut", 1),
        ReLULayer(),
        
        ConvLayer("module.layer1.1.conv1", merge_bn=net.module.layer1[1].bn1),
        ReLULayer(),        
        ConvLayer("module.layer1.1.conv2", merge_bn=net.module.layer1[1].bn2),
        IdentityLayer("shortcut", 6),
        ReLULayer(),        
        
        ConvLayer("module.layer1.2.conv1", merge_bn=net.module.layer1[2].bn1),
        ReLULayer(),        
        ConvLayer("module.layer1.2.conv2", merge_bn=net.module.layer1[2].bn2),
        IdentityLayer("shortcut", 11),
        ReLULayer(),

        ConvLayer("module.layer1.3.conv1", merge_bn=net.module.layer1[3].bn1),
        ReLULayer(),        
        ConvLayer("module.layer1.3.conv2", merge_bn=net.module.layer1[3].bn2),
        IdentityLayer("shortcut", 16),
        ReLULayer(),        

        ConvLayer("module.layer1.4.conv1", merge_bn=net.module.layer1[4].bn1),
        ReLULayer(),        
        ConvLayer("module.layer1.4.conv2", merge_bn=net.module.layer1[4].bn2),
        IdentityLayer("shortcut", 21),
        ReLULayer(),        
             
        
        ConvLayer("module.layer2.0.conv1", merge_bn=net.module.layer2[0].bn1),
        ReLULayer(),                
        ConvLayer("module.layer2.0.conv2", merge_bn=net.module.layer2[0].bn2),
        IdentityLayer("shortcut", 26),
        ReLULayer(),                
        
        ConvLayer("module.layer2.1.conv1", merge_bn=net.module.layer2[1].bn1),
        ReLULayer(),                
        ConvLayer("module.layer2.1.conv2", merge_bn=net.module.layer2[1].bn2),
        IdentityLayer("shortcut", 31),
        ReLULayer(),                
        
        ConvLayer("module.layer2.2.conv1", merge_bn=net.module.layer2[2].bn1),
        ReLULayer(),                
        ConvLayer("module.layer2.2.conv2", merge_bn=net.module.layer2[2].bn2),
        IdentityLayer("shortcut", 36),
        ReLULayer(),

        ConvLayer("module.layer2.3.conv1", merge_bn=net.module.layer2[3].bn1),
        ReLULayer(),        
        ConvLayer("module.layer2.3.conv2", merge_bn=net.module.layer2[3].bn2),
        IdentityLayer("shortcut", 41),
        ReLULayer(),        

        ConvLayer("module.layer2.4.conv1", merge_bn=net.module.layer2[4].bn1),
        ReLULayer(),        
        ConvLayer("module.layer2.4.conv2", merge_bn=net.module.layer2[4].bn2),
        IdentityLayer("shortcut", 46),
        ReLULayer(),                
        
        ConvLayer("module.layer3.0.conv1", merge_bn=net.module.layer3[0].bn1),
        ReLULayer(),                
        ConvLayer("module.layer3.0.conv2", merge_bn=net.module.layer3[0].bn2),
        IdentityLayer("shortcut", 51),
        ReLULayer(),                
        
        ConvLayer("module.layer3.1.conv1", merge_bn=net.module.layer3[1].bn1),
        ReLULayer(),
        ConvLayer("module.layer3.1.conv2", merge_bn=net.module.layer3[1].bn2),
        IdentityLayer("shortcut", 56),
        ReLULayer(),                
        
        ConvLayer("module.layer3.2.conv1", merge_bn=net.module.layer3[2].bn1),
        ReLULayer(),                
        ConvLayer("module.layer3.2.conv2", merge_bn=net.module.layer3[2].bn2),
        IdentityLayer("shortcut", 61),
        ReLULayer(),

        ConvLayer("module.layer3.3.conv1", merge_bn=net.module.layer3[3].bn1),
        ReLULayer(),        
        ConvLayer("module.layer3.3.conv2", merge_bn=net.module.layer3[3].bn2),
        IdentityLayer("shortcut", 66),
        ReLULayer(),        

        ConvLayer("module.layer3.4.conv1", merge_bn=net.module.layer3[4].bn1),
        ReLULayer(),        
        ConvLayer("module.layer3.4.conv2", merge_bn=net.module.layer3[4].bn2),
        IdentityLayer("shortcut", 71),
        ReLULayer(),                
        
        AvgPoolLayer("module.pool"),
        FlattenLayer(),
        
        FCLayer("module.linear")
    ]

    layers = load_neural_network_from_pytorch(net, layers)

    # Load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/tmp', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='/tmp', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1024, shuffle=False,
        pin_memory=True)

    correct, total = 0, 0
    timing_breakdown = {}
    for i, (input, target) in enumerate(val_loader):
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_neural_network(layers, input_var, timing_breakdown)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval

    acc = correct/total
    assert(acc >= .90)

    pprint.pprint(timing_breakdown)

    print("Pass ResNet32 pretrained with acc=%f" % acc)

def test_resnet_34_cifar100_pretrained():
    net = resnet34()
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_cifar100/checkpoint/resnet34/resnet34-142-best.pth"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data)

    layers = [
        ConvLayer("conv1.0", merge_bn=net.conv1[1]),
        ReLULayer(),

        ##
        
        ConvLayer("conv2_x.0.residual_function.0", merge_bn=net.conv2_x[0].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv2_x.0.residual_function.3", merge_bn=net.conv2_x[0].residual_function[4]),
        ReLULayer(),
        IdentityLayer("shortcut", 1),
        ReLULayer(),
        
        ConvLayer("conv2_x.1.residual_function.0", merge_bn=net.conv2_x[1].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv2_x.1.residual_function.3", merge_bn=net.conv2_x[1].residual_function[4]),
        ReLULayer(),
        IdentityLayer("shortcut", 7),
        ReLULayer(),        
        
        ConvLayer("conv2_x.2.residual_function.0", merge_bn=net.conv2_x[2].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv2_x.2.residual_function.3", merge_bn=net.conv2_x[2].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 13),
        ReLULayer(),
        
        ###
        
        PushInputsToStack(),
        ConvLayer("conv3_x.0.shortcut.0", merge_bn=net.conv3_x[0].shortcut[1]),
        ReLULayer(),
        PopInputsFromStack(),
        
        ConvLayer("conv3_x.0.residual_function.0", merge_bn=net.conv3_x[0].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv3_x.0.residual_function.3", merge_bn=net.conv3_x[0].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 22), # TODO: there is a conv for identity here
        ReLULayer(),
        
        ConvLayer("conv3_x.1.residual_function.0", merge_bn=net.conv3_x[1].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv3_x.1.residual_function.3", merge_bn=net.conv3_x[1].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 29), 
        ReLULayer(),
        
        ConvLayer("conv3_x.2.residual_function.0", merge_bn=net.conv3_x[2].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv3_x.2.residual_function.3", merge_bn=net.conv3_x[2].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 35), 
        ReLULayer(),        

        ConvLayer("conv3_x.3.residual_function.0", merge_bn=net.conv3_x[3].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv3_x.3.residual_function.3", merge_bn=net.conv3_x[3].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 41), 
        ReLULayer(),                
        
        ###
        
        PushInputsToStack(),
        ConvLayer("conv4_x.0.shortcut.0", merge_bn=net.conv4_x[0].shortcut[1]),
        ReLULayer(),        
        PopInputsFromStack(),
        
        ConvLayer("conv4_x.0.residual_function.0", merge_bn=net.conv4_x[0].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv4_x.0.residual_function.3", merge_bn=net.conv4_x[0].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 50), # TODO: there is a conv for identity here
        ReLULayer(),
        
        ConvLayer("conv4_x.1.residual_function.0", merge_bn=net.conv4_x[1].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv4_x.1.residual_function.3", merge_bn=net.conv4_x[1].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 57), 
        ReLULayer(),
        
        ConvLayer("conv4_x.2.residual_function.0", merge_bn=net.conv4_x[2].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv4_x.2.residual_function.3", merge_bn=net.conv4_x[2].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 63), 
        ReLULayer(),

        ConvLayer("conv4_x.3.residual_function.0", merge_bn=net.conv4_x[3].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv4_x.3.residual_function.3", merge_bn=net.conv4_x[3].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 69), 
        ReLULayer(),

        ConvLayer("conv4_x.4.residual_function.0", merge_bn=net.conv4_x[4].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv4_x.4.residual_function.3", merge_bn=net.conv4_x[4].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 75), 
        ReLULayer(),                        


        ConvLayer("conv4_x.5.residual_function.0", merge_bn=net.conv4_x[5].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv4_x.5.residual_function.3", merge_bn=net.conv4_x[5].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 81), 
        ReLULayer(),                        
        
        
        ###
        
        PushInputsToStack(),
        ConvLayer("conv5_x.0.shortcut.0", merge_bn=net.conv5_x[0].shortcut[1]),
        ReLULayer(),        
        PopInputsFromStack(),        
        
        ConvLayer("conv5_x.0.residual_function.0", merge_bn=net.conv5_x[0].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv5_x.0.residual_function.3", merge_bn=net.conv5_x[0].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 90), # TODO: there is a conv for identity here
        ReLULayer(),
        
        ConvLayer("conv5_x.1.residual_function.0", merge_bn=net.conv5_x[1].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv5_x.1.residual_function.3", merge_bn=net.conv5_x[1].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 97), 
        ReLULayer(),
        
        ConvLayer("conv5_x.2.residual_function.0", merge_bn=net.conv5_x[2].residual_function[1]),
        ReLULayer(),
        ConvLayer("conv5_x.2.residual_function.3", merge_bn=net.conv5_x[2].residual_function[4]),
        ReLULayer(),        
        IdentityLayer("shortcut", 103), 
        ReLULayer(),
        
        ###
        AvgPoolLayer("avg_pool"),
        FlattenLayer(),
        FCLayer("fc")
    ]

    layers = load_neural_network_from_pytorch(net, layers)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=512)
    val_loader = cifar100_test_loader


    correct, total = 0, 0
    timing_breakdown = {}
    for i, (input, target) in enumerate(val_loader):
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_neural_network(layers, input_var, timing_breakdown)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval

    acc = correct/total
    print("ResNet34 Cifar100 acc=%f" % acc)
    assert(acc >= .75)

    pprint.pprint(timing_breakdown)

    print("Pass ResNet34 Cifar100 pretrained with acc=%f" % acc)    
    
if __name__=="__main__":
    print("Testing network.py...")

    test_run_resnet_block()
    test_run_fc_network()
    test_run_conv_network()
    test_run_conv_batchnorm_network()
    
    test_mnist_pretrained()
    test_mnist_bn_pretrained()
    
    test_resnet20_sanity()
    test_resnet32_cifar10_pretrained()    
    test_resnet_34_cifar100_pretrained()

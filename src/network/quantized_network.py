# This file contains the following
#
# - Code that converts a pytorch model into a quantized network
#   representation executable by the tabula algorithm
#
# - Code that quantizes the model (w/ various quantized optimization procedures)
#
# - Code that executes the quantized network
#   Execution of quantized network is done analagously to how it is done in tabula.
#   (should yield _exact_ same output as tabula -- used for sanity checking + accuracy).
#
# - Code that evaluates the quantized network on datasets
#

import sys
import copy
import pprint
import time
from collections import OrderedDict
from collections import namedtuple
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import math
from network.neural_network import *
from train.mnist.mnist import MnistNet, test as MnistTest
from train.mnist.mnist_bn import MnistBNNet
from torchvision import datasets, transforms
import git

np.random.seed(0)
torch.manual_seed(0)

DEFAULT_BITS = 32

def quantize(W, bits=DEFAULT_BITS, qrange=30):
    best_err = float("inf")
    for ss in np.arange(-qrange, qrange):
        cur_scale = 2**int(ss)
        cur_q_w = np.clip(np.round(W/cur_scale).astype(np.float64), -2**bits, 2**bits)
        q_err = np.mean(np.abs(W-cur_q_w*cur_scale))
        if q_err <= best_err:
            best_err = q_err
            scale, q_w = cur_scale, cur_q_w

    return scale, quantize_with_scale(W, scale), best_err

def rescale_shared(x_q, x_scale, rescale, fp=2**62-1):

    x_q += np.random.randint(0, 2, size=x_q.shape, dtype=np.int64)

    prevtype = x_q.dtype

    x_q = x_q.astype(np.int64)
    rescale = np.int64(rescale)

    # Simulated shared secret truncaation
    # This introduces some error +/-1
    
    def trunc(a, b, s, fp):
        a //= s
        b  = fp - (fp-b)//s

        a = a.astype(np.int64)
        b = b.astype(np.int64)
        return a,b

    def enc(v, fp):
        np.random.seed(0)
        r = np.random.randint(0, fp, size=x_q.shape, dtype=np.int64)
        return (v - r) % fp, r

    def dec(a, b, fp):
        x = (a+b) % fp
        np.putmask(x, x > (fp-1)//2, x-fp)        
        return x

    print("rescale_shared input bits: %f" % (np.ceil(np.log2(np.max(x_q)))))
    x_q_correct = x_q//rescale
    x_scale *= rescale

    a,b = enc(x_q, fp)
    a,b = trunc(a, b, rescale, fp)
    x_q = dec(a, b, fp)
    
    # dbg
    errs = dict(Counter((x_q-x_q_correct).flatten().tolist()))
    print("rescale_shared errs (should be 0, 1 (a few off ok)):" + str(errs))

    # there's a +1 bias to the secure truncation, subtract 1 with prob
    # proportional to the errors. Default is 50/50
    np.random.seed(0)
    x_q -= np.random.randint(0, 2, size=x_q.shape, dtype=np.int64)
    
    x_q = x_q.astype(prevtype)
    
    return x_q, x_scale

def quantize_with_scale(W, scale):
    #cur_q_w = np.clip(np.round(W/scale), -2**bits, 2**bits)
    cur_q_w = np.round(W/scale).astype(np.float64)
    return cur_q_w

def replace_relu_with_quantized_relu(layers, relu_bits=10):
    return [x if type(x) is not ReLULayer else QuantizedReLULayer(bits=relu_bits) for x in layers]

class QuantizedReLULayer(Layer):
    id = 0
    def __init__(self, name="ReLUQuantized", bits=10):
        super().__init__(name+str(QuantizedReLULayer.id))
        QuantizedReLULayer.id += 1
        self.bits = bits

    def clear(self):
        pass

class QuantizedInputLayer(Layer):
    def __init__(self, bits=DEFAULT_BITS, qrange=30):
        super().__init__("quantized_input_layer")
        self.bits = bits
        self.qrange = qrange

    def __repr__(self):
        return "%s (quantized bits=%d qrange=%f)" % (self.name, self.bits, self.qrange)

    def clear(self):
        pass

class QuantizedAvgPoolLayer(Layer):
    def __init__(self, avg_pool_layer, rescale=None):
        super().__init__(avg_pool_layer.name + "_quantized")
        self.unquantized = avg_pool_layer
        self.rescale = rescale

    def clear(self):
        pass

class QuantizedIdentityLayer(Layer):
    def __init__(self, identity_layer):
        super().__init__(identity_layer.name + "_quantized")
        self.unquantized = identity_layer

    def clear(self):
        pass
        
class QuantizedFCLayer(Layer):
    def __init__(self, fc_layer, bits=DEFAULT_BITS, rescale=2**DEFAULT_BITS, qrange=30):
        super().__init__(fc_layer.name + "_quantized")
        self.bits = bits
        self.qrange = qrange
        self.unquantized = fc_layer
        self.scale, self.weight_mat, self.err = quantize(self.unquantized.weight_mat,
                                                         bits=self.bits,
                                                         qrange=self.qrange)
        self.weight = self.weight_mat.T
        self.rescale = rescale

        with torch.no_grad():
            self.pytorch_module = copy.deepcopy(self.unquantized.pytorch_module)
            self.pytorch_module.bias = None
            self.pytorch_module.weight = torch.nn.parameter.Parameter(torch.from_numpy(self.weight).double(), requires_grad=False)
        
    def __repr__(self):
        return "%s (quantized bits=%d rescale=%f qrange=%f)" % (self.unquantized.__repr__(), self.bits, self.rescale, self.qrange)

    def clear(self):
        self.weight_mat = np.zeros_like(self.weight_mat)
        self.unquantized.clear()

class QuantizedConvLayer(Layer):
    def __init__(self, conv_layer, bits=DEFAULT_BITS, rescale=2**DEFAULT_BITS, qrange=30):
        super().__init__(conv_layer.name + "_quantized")
        self.bits = bits
        self.qrange = qrange
        self.unquantized = conv_layer
        self.scale, self.weight, self.err = quantize(self.unquantized.weight,
                                                     bits=self.bits,
                                                     qrange=self.qrange)
        self.rescale = rescale

        with torch.no_grad():
            self.pytorch_module = copy.deepcopy(self.unquantized.pytorch_module)
            self.pytorch_module.bias = None
            self.pytorch_module.weight = torch.nn.parameter.Parameter(torch.from_numpy(self.weight).double(), requires_grad=False)            

    def clear(self):
        self.weight = np.zeros_like(self.weight)
        self.unquantized.clear()

    def __repr__(self):
        return "%s (quantized bits=%d rescale=%f qrange=%f)" % (self.unquantized.__repr__(), self.bits, self.rescale, self.qrange)

def load_quantized_neural_network_from_pytorch(pytorch_model, layers,
                                               default_precision=DEFAULT_BITS,
                                               default_rescale=2**DEFAULT_BITS,
                                               default_qrange=30,
                                               precision_assignment=None):
    # Note that precision is for weights/activations/accumulator

    # Load normally
    layers = load_neural_network_from_pytorch(pytorch_model, layers, mute=True)

    # Quantize each layer
    q_layers = []
    for l in layers:
        bits = default_precision
        rescale = default_rescale
        qrange = default_qrange

        if precision_assignment is not None:
            if "rescale" in precision_assignment:
                rescale = precision_assignment["rescale"]
            if "qrange" in precision_assignment:
                qrange = precision_assignment["qrange"]
            if l.name in precision_assignment:
                if "bits" in precision_assignment[l.name]:
                    bits = precision_assignment[l.name]["bits"]
                if "qrange" in precision_assignment[l.name]:
                    qrange = precision_assignment[l.name]["qrange"]
                if "rescale" in precision_assignment[l.name]:
                    rescale = precision_assignment[l.name]["rescale"]

        if type(l) == FCLayer:
            l = QuantizedFCLayer(l, bits=bits, rescale=rescale, qrange=qrange)
        if type(l) == ConvLayer:
            l = QuantizedConvLayer(l, bits=bits, rescale=rescale, qrange=qrange)
        if type(l) == AvgPoolLayer:
            l = QuantizedAvgPoolLayer(l, rescale=rescale)
        if type(l) == IdentityLayer:
            l = QuantizedIdentityLayer(l)
        q_layers.append(l)
        
    # Add input layer
    input_bits = default_precision
    input_qrange = default_qrange
    if precision_assignment is not None and "qrange" in precision_assignment:
        input_qrange = precision_assignment["qrange"]
    if precision_assignment is not None and "input" in precision_assignment:
        if "bits" in precision_assignment["input"]:
            input_bits = precision_assignment["input"]["bits"]
        if "qrange" in precision_assignment["input"]:
            input_qrange = precision_assignment["input"]["qrange"]
        
    q_layers = [QuantizedInputLayer(bits=input_bits, qrange=input_qrange)] + q_layers
        
    print_layers(q_layers)
        
    return q_layers

def execute_quantized_neural_network(layers, x, timing_breakdown={}, misc={}):
    # Quantized network execution works as follows:
    # 
    # - Tensors are split into corresponding scale (scalar) and quantized_data (tensor) tuples.
    #   scale scalars are powers of 2.
    #
    # - Matmuls/convs involve multiplying the scale scalars, and performing
    #   the op on the quantized_data
    #
    # - After each matmul/conv, rescale the scale scalars and the quantized_data tensors
    #   so that quantized_data is again n-bits (as oppoed to 2n bits)
    #
    # - Addition of quantized values requires rescaling the scalars and quantized
    #   values so that the scalars match.
    #
    # - Avg pool sums the kernel, then divides the scale by the kernel size^2
    #
    # Note: Each of these steps should work with additively shared quantized_data tensors. Hence,
    #       this overall process should be analagously implemented in tabula, and will yield exact same outputs.
    #       e.g: rescaling additively shared values using lookup tables, etc
    stack = []
    misc["shapes"] = {}
    misc["n_relus"] = 0
    misc["n_linear_elements"] = 0
    x_q = np.zeros_like(x).astype(np.int64)
    tot_act_size = 0

    
    for i, layer in enumerate(layers):

        t_start = time.time()

        if i != 0:
            tot_act_size += np.prod(x_q.shape)

        misc["shapes"][i] = x_q.shape
        
        if type(layer) == FCLayer or type(layer) == ConvLayer or type(layer) == AvgPoolLayer:
            raise Exception("execute_quantized_neural_network: Found unquantized type: %s "% type(layer))

        if type(layer) == PushInputsToStack:
            stack.append((x_scale, x_q))
        elif type(layer) == PopInputsFromStack:
            x_scale, x_q = stack.pop(-1)
        
        elif type(layer) == QuantizedInputLayer:
            x_scale, x_q, _ = quantize(x, bits=layer.bits, qrange=layer.qrange)
            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1

        elif type(layer) == QuantizedIdentityLayer:
            # +1 since quantized layers add a quantized input layer
            indx = layer.unquantized.indx_identity + 1  
            other_scale, other_q = layers[indx].output

            if other_q.shape != x_q.shape:

                # This happens with cifar10 resnet
                p = (x_q.shape[1]-other_q.shape[1])//2
                other_q = np.pad(other_q[:,:,::2,::2], ((0,0), (p,p), (0,0), (0,0)), mode="constant", constant_values=0)
                assert(other_q.shape == x_q.shape)
                
            # Rescale: use the smaller scale
            #
            # This effectively multiplies the smaller weights by the diff in scale to enable adding
            #
            # In private inference, this multiplies the random share by the same diff in scale (which is known to both parties, part of arch)
            # The two parties still hold a secret share of the weights 
            smaller_scale, larger_scale = min(other_scale, x_scale), max(other_scale, x_scale)
            ratio = larger_scale / smaller_scale            

            assert(ratio.is_integer())
            if smaller_scale == x_scale:
                x_q = x_q + other_q*ratio
            else:
                x_q = x_q*ratio + other_q

            x_scale = smaller_scale

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1            

        elif type(layer) == QuantizedFCLayer:

            misc["n_linear_elements"] += int(np.prod(x_q.shape))            
            
            x_q = x_q.dot(layer.weight_mat)
            x_scale *= layer.scale

            # Bias
            if layer.unquantized.bias is not None:
                b_q = quantize_with_scale(layer.unquantized.bias, x_scale)
                x_q += b_q

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1

            # Rescale 
            x_q, x_scale = rescale_shared(x_q, x_scale, layer.rescale)
            
        elif type(layer) == QuantizedConvLayer:
            misc["n_linear_elements"] += int(np.prod(x_q.shape))            
            
            x_q = layer.pytorch_module(torch.from_numpy(x_q).double()).detach().numpy()
            x_scale *= layer.scale

            # Bias
            if layer.unquantized.bias_mat is not None:
                b_q = quantize_with_scale(layer.unquantized.bias_mat, x_scale)
                b_q = b_q.reshape(b_q.shape + (1,))
                b_q = b_q.astype(x_q.dtype)
                x_q += b_q

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1
            
            # Rescale 
            x_q, x_scale = rescale_shared(x_q, x_scale, layer.rescale)            
            
        elif type(layer) == QuantizedAvgPoolLayer:
            kernel_size_2 = layer.unquantized.kernel_size**2
            x_q = layer.unquantized.pytorch_module(torch.from_numpy(x_q)).numpy()*kernel_size_2
            x_scale /= kernel_size_2

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1

            # Rescale if necessary -- use for large kernel sizes
            # Also note, that this works because avg pool comes after relu
            # hence can be combined with relu lookup table
            if layer.rescale is not None:
                x_q = np.round(x_q/layer.rescale)
                x_scale *= layer.rescale            

        elif type(layer) == ReLULayer:
            x_q[x_q<0] = 0

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1
            misc["n_relus"] += int(np.prod(x_q.shape))

        elif type(layer) == QuantizedReLULayer:
            v = x_q*x_scale

            # Calc number of bits in x_q
            bits_of_x_q = np.ceil(np.log2(np.max(np.abs(x_q))))+1
            bits_in = bits_of_x_q

            # Track bits in
            if layer not in misc:
                misc[layer] = {}
            misc[layer]["bits_in"] = bits_in

            print("QuantizedReLULayer %s - Input with %d bits" % (layer, bits_in))

            # Scale down
            scale_factor = max(1, 2**(bits_in-layer.bits))
            x_q, x_scale = rescale_shared(x_q, x_scale, scale_factor)
            
            x_q[x_q<=0] = 0

            # Scale back up
            x_scale /= scale_factor
            x_q *= scale_factor

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1
            misc["n_relus"] += int(np.prod(x_q.shape))            
            
        elif type(layer) == FlattenLayer:
            x_q = x_q.reshape((x_q.shape[0], -1))

            bits_for_output = np.ceil(np.log2(np.max(np.abs(x_q))))+1            
        else:
            raise Exception("execute_quantized_neural_network: Unknown type: %s" % type(layer))
        t_end = time.time()
        elapsed = t_end-t_start
        if str(layer) not in timing_breakdown:
            timing_breakdown[str(layer)] = 0
        timing_breakdown[str(layer)] += elapsed

        # Reference output
        layer.output = (x_scale, x_q)

        if np.isnan(x_q).any() or not np.isfinite(x_q).all():
            raise Exception("Found nan at layer %s" % layer)

        # Track misc data 
        # e.g: number of bits to represent largest value
        # (used to verfify that numbers representable in small finite field during private inference)
        if layer not in misc:
            misc[layer] = {}
        if "bits_for_output" not in misc[layer]:
            misc[layer]["bits_for_output"] = []
            misc[layer]["bits_for_output"].append(bits_for_output)

        # Sanity check x_q
        assert(np.linalg.norm(np.round(x_q) - x_q) <= 1e-5)

    return x_q*x_scale

##### Tests #####

def test_run_quantized_fc_network_32bit():
    class FCNet(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.L1 = nn.Linear(28*28, hidden_size, bias=True)
            self.L2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.L3 = nn.Linear(hidden_size, 10, bias=True)
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
    net.double()
    net.eval()
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=16)

    # Test batch != 1
    for b in [1, 16, 64, 128]:
        x = np.random.normal(0, 10, size=(b, 28*28)).astype(np.float64)
        result = execute_quantized_neural_network(layers, x)
        golden = net(torch.from_numpy(x).double()).detach().numpy()


        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-5)

    print("Pass FCnet 1")

def test_mnist_quantized_pretrained_32bit():
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
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=2**16)

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
        
        pred = execute_quantized_neural_network(layers, data, timing_breakdown)
        pred = np.argmax(pred, axis=1)
        
        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # uncomment for full
    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total
    print("Mnist pretrained %f" % acc)
    assert(acc >= .97)

    pprint.pprint(timing_breakdown)    
    print("Pass mnist pretrained 32-bit with acc=%f time=%f s" % (acc, telapsed))

def test_run_conv_network_32bit():
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, bias=True)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.AvgPool2d(2)
            self.conv2 = nn.Conv2d(6, 16, 5, bias=True)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.AvgPool2d(2)
            self.fc1 = nn.Linear(256, 120)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
            self.relu4 = nn.ReLU()
            self.fc3 = nn.Linear(84, 10, bias=True)
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
    net.double()
    net.eval()    
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=2**16)

    # Test batch != 1
    for b in [1, 16, 64, 128]:
        # Recall (N, C, H, W) convolution format
        x = np.random.normal(0, 10, size=(b, 1, 28, 28)).astype(np.float64)
        golden = net(torch.from_numpy(x).double()).detach().numpy()
        result = execute_quantized_neural_network(layers, x)

        assert(result.shape == golden.shape)
        print(mean_err(result, golden))
        assert(mean_err(result, golden) <= 1e-6)

    print("Pass Convnet")

def test_mnist_quantized_pretrained(default_precision=DEFAULT_BITS,
                                    default_rescale=2**DEFAULT_BITS):
    
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
    layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                        default_precision=default_precision,
                                                        default_rescale=default_rescale)

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
        
        pred = execute_quantized_neural_network(layers, data, timing_breakdown)
        pred = np.argmax(pred, axis=1)

        
        correct += np.sum(pred == target)
        total += pred.shape[0]
        break # comment for full

    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total

    pprint.pprint(timing_breakdown)    
    print("Mnist pretrained precision=%d, rescale=%f  with acc=%f time=%f s" % (default_precision, default_rescale, acc, telapsed))
    assert(acc >= .97)
    print("Passed MNIST")

def test_mnist_quantized_pretrained_precision_assignments_1():
    
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

    passignment = {
        # Note tht rescale bits must be the same for all layers
        # because we fuse it with relu hash table (in tabula)
        "rescale" : 42,
        "input": {
            "bits": 4,
        },
        "conv1": {
            "bits":2,
        },
        "conv2": {
            "bits":2,
        },
        "fc1": {
            "bits":2,
        },
        "fc2": {
            "bits":2,
        },
        "pool2d": {
            "rescale": None
        }
    }    

    # Model
    net = MnistNet()
    net.eval()    
    pretrained_model_path = get_git_root(__file__) + "/src/train/mnist/mnist_cnn.pt"
    net.load_state_dict(torch.load(pretrained_model_path))
    layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                        precision_assignment=passignment)

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
    misc = {}
    correct, total = 0, 0
    for data, target in test_loader:
        data = data.numpy()
        target = target.numpy()
        
        pred = execute_quantized_neural_network(layers, data, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)
        
        correct += np.sum(pred == target)
        total += pred.shape[0]
        break # comment for full

    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total

    pprint.pprint(timing_breakdown)
    pprint.pprint(misc)
    print("Mnist  with acc=%f time=%f s" % (acc, telapsed))
    assert(acc >= .97)
    print("Passed MNIST")

def test_mnist_bn_quantized_pretrained_32bit():
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
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=16, default_rescale=2**16)

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

        pred = execute_quantized_neural_network(layers, data, timing_breakdown)
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

def test_mnist_bn_quantized_pretrained_precision_assignments():
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

    passignments = {
        "rescale" : 2**6,
        "input": {
            "bits": 4
        },
        "conv1": {
            "bits":4,
        },
        "conv2": {
            "bits":4,
        },
        "fc1": {
            "bits":3,
        },
        "fc2": {
            "bits":3,
        },
        "pool2d": {
            "rescale": None
        }
    }
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/mnist/mnist_bn_cnn.pt"
    net.load_state_dict(torch.load(pretrained_model_path))
    layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                        precision_assignment=passignments)

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
    misc = {}
    correct, total = 0, 0
    for data, target in test_loader:
        data = data.numpy()
        target = target.numpy()

        pred = execute_quantized_neural_network(layers, data, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)
        correct += np.sum(pred == target)
        total += pred.shape[0]
        break # Comment for full
    
    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total
    print("Mnist bn pretrained with acc=%f time=%f s" % (acc, telapsed))
    pprint.pprint(timing_breakdown)
    pprint.pprint(misc)    
    assert(acc >= .97)

    print('Pass MNIST bn pretrained precision assignment')

def test_run_resnet_block_32bit():

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
    net.double()
    layers = [
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer("ReLU"),
        ConvLayer("conv2", merge_bn=net.bn2),
        IdentityLayer("Identity", -1),
        ReLULayer("ReLU")
    ]
    
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=2**16)
    misc = {}

    for b in [1, 16, 32]:
        x = np.random.normal(0, 1, size=(b, 32, 32, 32)).astype(np.float64)
        golden = net(torch.from_numpy(x).double()).detach().numpy()
        result = execute_quantized_neural_network(layers, x, misc=misc)

        assert(result.shape == golden.shape)
        assert(mean_err(result, golden) <= 1e-6)

    print("Pass ResNet block")

def test_resnet20_sanity_32bit():
    net = torch.nn.DataParallel(resnet20())
    net.eval()

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

    precision_assignment = {
        "module.pool" : {
            "rescale": None
        }
    }

    # IMPORTANT: note for pretrained resnet, due to batchnorm, precision scale is higher
    # This is to avoid overflow. Tuning it is necessary and non-systematic 
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=2**29)

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
        result = execute_quantized_neural_network(layers, input_var.numpy())

        assert(result.shape == golden.shape)
        print(mean_err(result, golden))
        assert(mean_err(result, golden) <= 1e-5)
        break

    print("Pass ResNet20 sanity 32-bit")    

def test_resnet32_cifar10_pretrained_32bit():
    net = torch.nn.DataParallel(resnet32())
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data["state_dict"])

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

    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=2**32)

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

        pred = execute_quantized_neural_network(layers, input_var, timing_breakdown)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval


    acc = correct/total
    print("ResNet32 pretrained 32-bit acc: %f" % acc)
    assert(acc >= .90)

    pprint.pprint(timing_breakdown)

    print("Pass ResNet32 pretrained with acc=%f" % acc)

def test_resnet32_cifar10_pretrained_quantized():
    net = torch.nn.DataParallel(resnet32())
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data["state_dict"])

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

    passignments = {
        "rescale" : 2**6,
        "input" : {"bits": 5, "qrange": 30},
        "module.conv1" : {"bits": 4},
        
        "module.layer1.0.conv1" : {"bits": 4},
        "module.layer1.0.conv2" : {"bits": 4},

        "module.layer1.1.conv1" : {"bits": 4},
        "module.layer1.1.conv2" : {"bits": 4},

        "module.layer1.2.conv1" : {"bits": 4},
        "module.layer1.2.conv2" : {"bits": 4},

        "module.layer1.3.conv1" : {"bits": 4},
        "module.layer1.3.conv2" : {"bits": 4},

        "module.layer1.4.conv1" : {"bits": 4},
        "module.layer1.4.conv2" : {"bits": 4},

        #--

        "module.layer2.0.conv1" : {"bits": 4},
        "module.layer2.0.conv2" : {"bits": 4},

        "module.layer2.1.conv1" : {"bits": 4},
        "module.layer2.1.conv2" : {"bits": 4},

        "module.layer2.2.conv1" : {"bits": 4},
        "module.layer2.2.conv2" : {"bits": 4},

        "module.layer2.3.conv1" : {"bits": 4},
        "module.layer2.3.conv2" : {"bits": 4},

        "module.layer2.4.conv1" : {"bits": 4},
        "module.layer2.4.conv2" : {"bits": 4},

        #--

        "module.layer3.0.conv1" : {"bits": 4},
        "module.layer3.0.conv2" : {"bits": 4},

        "module.layer3.1.conv1" : {"bits": 4},
        "module.layer3.1.conv2" : {"bits": 4},

        "module.layer3.2.conv1" : {"bits": 4},
        "module.layer3.2.conv2" : {"bits": 4},

        "module.layer3.3.conv1" : {"bits": 4},
        "module.layer3.3.conv2" : {"bits": 4},

        "module.layer3.4.conv1" : {"bits": 4},
        "module.layer3.4.conv2" : {"bits": 1},

        #--

        "module.linear" : {"bits": 4}
    }    

    
    #layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=32, default_rescale=2**4)
    #layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=6, default_rescale=2**7)
    layers = load_quantized_neural_network_from_pytorch(net, layers, precision_assignment=passignments, default_qrange=6)

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
    misc = {}
    for i, (input, target) in enumerate(val_loader):
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, misc=misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval


    acc = correct/total
    pprint.pprint(misc)
    print("ResNet32 pretrained 32-bit acc: %f" % acc)
    assert(acc >= .89)

    print("Pass ResNet32 pretrained with acc=%f" % acc)

def test_resnet_34_cifar100_quantized():
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
    
    precision_assignment = {
        "input": {"bits": 5},
        
        "conv2_x.0.residual_function.0": {"bits": 5},
        "conv2_x.0.residual_function.3": {"bits": 5},
        "conv2_x.1.residual_function.0": {"bits": 5},
        "conv2_x.1.residual_function.3": {"bits": 5},
        "conv2_x.2.residual_function.0": {"bits": 5},
        "conv2_x.2.residual_function.3": {"bits": 5},
        
        "conv3_x.0.residual_function.0": {"bits": 5},
        "conv3_x.0.residual_function.3": {"bits": 5},
        "conv3_x.1.residual_function.0": {"bits": 4},
        "conv3_x.1.residual_function.3": {"bits": 2},
        "conv3_x.2.residual_function.0": {"bits": 5},
        "conv3_x.2.residual_function.3": {"bits": 5},
        "conv3_x.3.residual_function.0": {"bits": 4},
        "conv3_x.3.residual_function.3": {"bits": 3},

        "conv4_x.0.residual_function.0": {"bits": 5},
        "conv4_x.0.residual_function.3": {"bits": 5},
        "conv4_x.1.residual_function.0": {"bits": 5},
        "conv4_x.1.residual_function.3": {"bits": 5},
        "conv4_x.2.residual_function.0": {"bits": 5},
        "conv4_x.2.residual_function.3": {"bits": 5},
        "conv4_x.3.residual_function.0": {"bits": 5},
        "conv4_x.3.residual_function.3": {"bits": 5},
        "conv4_x.4.residual_function.0": {"bits": 5},
        "conv4_x.4.residual_function.3": {"bits": 5},
        "conv4_x.5.residual_function.0": {"bits": 4},
        "conv4_x.5.residual_function.3": {"bits": 3},        
        
        "conv5_x.0.residual_function.0": {"bits": 5},
        "conv5_x.0.residual_function.3": {"bits": 5},
        "conv5_x.1.residual_function.0": {"bits": 5},
        "conv5_x.1.residual_function.3": {"bits": 5},
        "conv5_x.2.residual_function.0": {"bits": 4},
        "conv5_x.2.residual_function.3": {"bits": 3},
        
        "fc": {"bits": 5},
    }

    layers = load_quantized_neural_network_from_pytorch(net, layers, precision_assignment=precision_assignment,
                                                        default_qrange=7, default_precision=5, default_rescale=2**7)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=128)
    val_loader = cifar100_test_loader


    correct, total = 0, 0
    timing_breakdown = {}
    misc = {}
    for i, (input, target) in enumerate(val_loader):
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval

    acc = correct/total
    pprint.pprint(misc)
    print("ResNet34 Cifar100 acc=%f" % acc)
    assert(acc >= .73)

    print("Pass ResNet34 Cifar100 pretrained with acc=%f" % acc)

def test_vgg_cifar100_pretrained():

    net = vgg16_bn()
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_cifar100/checkpoint/vgg16/vgg16-200-regular.pth"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data)

    layers = [
        ConvLayer("features.0",  merge_bn=net.features[1]),
        ReLULayer(),
        ConvLayer("features.3", merge_bn=net.features[4]),
        ReLULayer(),
        AvgPoolLayer("features.6"),

        ConvLayer("features.7",  merge_bn=net.features[8]),
        ReLULayer(),
        ConvLayer("features.10", merge_bn=net.features[11]),
        ReLULayer(),
        AvgPoolLayer("features.13"),

        ConvLayer("features.14",  merge_bn=net.features[15]),
        ReLULayer(),
        ConvLayer("features.17", merge_bn=net.features[18]),
        ReLULayer(),
        ConvLayer("features.20", merge_bn=net.features[21]),
        ReLULayer(),        
        AvgPoolLayer("features.23"),

        ConvLayer("features.24",  merge_bn=net.features[25]),
        ReLULayer(),
        ConvLayer("features.27", merge_bn=net.features[28]),
        ReLULayer(),
        ConvLayer("features.30", merge_bn=net.features[31]),
        ReLULayer(),        
        AvgPoolLayer("features.33"),
        
        ConvLayer("features.34",  merge_bn=net.features[35]),
        ReLULayer(),
        ConvLayer("features.37", merge_bn=net.features[38]),
        ReLULayer(),
        ConvLayer("features.40", merge_bn=net.features[41]),
        ReLULayer(),        
        AvgPoolLayer("features.43"),

        FlattenLayer("flatten"),

        FCLayer("classifier.0"),
    ]

    # This config gets .695 acc with 16 bits
    passignment = {
        "rescale" : 31,
        "input" : {
            "bits": 7,
        },
        "features.0": {
            "bits": 4
        },
        "features.3": {
            "bits": 4
        },
        "features.4": {
            "bits": 4
        },
        "features.4": {
            "bits": 4
        },
        "features.10": {
            "bits": 4
        },
        "features.13": {
            "bits": 4
        },
        "features.14": {
            "bits": 4
        },
        "features.14": {
            "bits": 4
        },
        "features.20": {
            "bits": 4
        },
        "features.23": {
            "bits": 4
        },
        "features.24": {
            "bits": 4
        },
        "features.24": {
            "bits": 4
        },
        "features.30": {
            "bits": 4
        },
        "features.33": {
            "bits": 4
        },
        "features.34": {
            "bits": 4
        },
        "features.34": {
            "bits": 4
        },
        "features.40": {
            "bits": 4
        },
        "features.32": {
            "bits": 4
        },
        "classifier.0": {
            "bits": 5
        },
    }
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=512)
    val_loader = cifar100_test_loader

    layers = load_quantized_neural_network_from_pytorch(net, layers, precision_assignment=passignment,
                                                        default_qrange=6, default_precision=5, default_rescale=15)
    
    
    correct, total = 0, 0
    timing_breakdown = {}
    misc = {}
    for i, (input, target) in enumerate(val_loader):

        """
        pred = net(input).detach().numpy()
        pred = np.argmax(pred, axis=1)
        print(pred)
        print(pred == target.numpy())
        correct += np.sum(pred == target.numpy())
        total += pred.shape[0]
        """

        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval

    acc = correct/total
    pprint.pprint(misc)
    print("ResNet34 Cifar100 acc=%f" % acc)
    assert(acc >= .69)

    print("Pass VGG16 Cifar100 pretrained with acc=%f" % acc)

def test_mnist_act_quantized():
    # See train/mnist/mnist.py for network architecture
        
    layers = [
        ConvLayer("conv1"),
        QuantizedReLULayer(),

        ConvLayer("conv2"),
        QuantizedReLULayer(),
        
        AvgPoolLayer("pool2d"),

        FlattenLayer("flatten"),

        FCLayer("fc1"),
        QuantizedReLULayer(),

        FCLayer("fc2"),
        
        # Note there is a softmax, but don't need for classification
    ]    

    # Model
    net = MnistNet()
    net.eval()    
    pretrained_model_path = get_git_root(__file__) + "/src/train/mnist/mnist_cnn.pt"
    net.load_state_dict(torch.load(pretrained_model_path))
    #layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=12, default_rescale=2**12)
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=6, default_rescale=2**6)

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
        
        pred = execute_quantized_neural_network(layers, data, timing_breakdown)
        pred = np.argmax(pred, axis=1)
        
        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # uncomment for full
    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total
    print("Mnist pretrained %f" % acc)
    assert(acc >= .97)

    pprint.pprint(timing_breakdown)    
    print("Pass mnist pretrained 32-bit with acc=%f time=%f s" % (acc, telapsed))

def test_resnet32_cifar10_act_quantized():
    net = torch.nn.DataParallel(resnet32())
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data["state_dict"])

    layers = [
        ConvLayer("module.conv1", merge_bn=net.module.bn1),
        QuantizedReLULayer(),
        
        ConvLayer("module.layer1.0.conv1", merge_bn=net.module.layer1[0].bn1),
        QuantizedReLULayer(),
        ConvLayer("module.layer1.0.conv2", merge_bn=net.module.layer1[0].bn2),
        IdentityLayer("shortcut", 1),
        QuantizedReLULayer(),
        
        ConvLayer("module.layer1.1.conv1", merge_bn=net.module.layer1[1].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer1.1.conv2", merge_bn=net.module.layer1[1].bn2),
        IdentityLayer("shortcut", 6),
        QuantizedReLULayer(),        
        
        ConvLayer("module.layer1.2.conv1", merge_bn=net.module.layer1[2].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer1.2.conv2", merge_bn=net.module.layer1[2].bn2),
        IdentityLayer("shortcut", 11),
        QuantizedReLULayer(),

        ConvLayer("module.layer1.3.conv1", merge_bn=net.module.layer1[3].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer1.3.conv2", merge_bn=net.module.layer1[3].bn2),
        IdentityLayer("shortcut", 16),
        QuantizedReLULayer(),        

        ConvLayer("module.layer1.4.conv1", merge_bn=net.module.layer1[4].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer1.4.conv2", merge_bn=net.module.layer1[4].bn2),
        IdentityLayer("shortcut", 21),
        QuantizedReLULayer(),        
             
        
        ConvLayer("module.layer2.0.conv1", merge_bn=net.module.layer2[0].bn1),
        QuantizedReLULayer(),                
        ConvLayer("module.layer2.0.conv2", merge_bn=net.module.layer2[0].bn2),
        IdentityLayer("shortcut", 26),
        QuantizedReLULayer(),                
        
        ConvLayer("module.layer2.1.conv1", merge_bn=net.module.layer2[1].bn1),
        QuantizedReLULayer(),                
        ConvLayer("module.layer2.1.conv2", merge_bn=net.module.layer2[1].bn2),
        IdentityLayer("shortcut", 31),
        QuantizedReLULayer(),                
        
        ConvLayer("module.layer2.2.conv1", merge_bn=net.module.layer2[2].bn1),
        QuantizedReLULayer(),                
        ConvLayer("module.layer2.2.conv2", merge_bn=net.module.layer2[2].bn2),
        IdentityLayer("shortcut", 36),
        QuantizedReLULayer(),

        ConvLayer("module.layer2.3.conv1", merge_bn=net.module.layer2[3].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer2.3.conv2", merge_bn=net.module.layer2[3].bn2),
        IdentityLayer("shortcut", 41),
        QuantizedReLULayer(),        

        ConvLayer("module.layer2.4.conv1", merge_bn=net.module.layer2[4].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer2.4.conv2", merge_bn=net.module.layer2[4].bn2),
        IdentityLayer("shortcut", 46),
        QuantizedReLULayer(),                
        
        ConvLayer("module.layer3.0.conv1", merge_bn=net.module.layer3[0].bn1),
        QuantizedReLULayer(),                
        ConvLayer("module.layer3.0.conv2", merge_bn=net.module.layer3[0].bn2),
        IdentityLayer("shortcut", 51),
        QuantizedReLULayer(),                
        
        ConvLayer("module.layer3.1.conv1", merge_bn=net.module.layer3[1].bn1),
        QuantizedReLULayer(),
        ConvLayer("module.layer3.1.conv2", merge_bn=net.module.layer3[1].bn2),
        IdentityLayer("shortcut", 56),
        QuantizedReLULayer(),                
        
        ConvLayer("module.layer3.2.conv1", merge_bn=net.module.layer3[2].bn1),
        QuantizedReLULayer(),                
        ConvLayer("module.layer3.2.conv2", merge_bn=net.module.layer3[2].bn2),
        IdentityLayer("shortcut", 61),
        QuantizedReLULayer(),

        ConvLayer("module.layer3.3.conv1", merge_bn=net.module.layer3[3].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer3.3.conv2", merge_bn=net.module.layer3[3].bn2),
        IdentityLayer("shortcut", 66),
        QuantizedReLULayer(),        

        ConvLayer("module.layer3.4.conv1", merge_bn=net.module.layer3[4].bn1),
        QuantizedReLULayer(),        
        ConvLayer("module.layer3.4.conv2", merge_bn=net.module.layer3[4].bn2),
        IdentityLayer("shortcut", 71),
        QuantizedReLULayer(),                
        
        AvgPoolLayer("module.pool"),
        FlattenLayer(),
        
        FCLayer("module.linear")
    ]

    # layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=10, default_rescale=2**12)
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=13, default_rescale=2**14)

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

        pred = execute_quantized_neural_network(layers, input_var, timing_breakdown)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval


    acc = correct/total
    print("ResNet32 pretrained 32-bit acc: %f" % acc)
    assert(acc >= .90)

    pprint.pprint(timing_breakdown)

    print("Pass ResNet32 pretrained with acc=%f" % acc)

def test_resnet34_cifar100_act_quantized():
    net = resnet34()
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_cifar100/checkpoint/resnet34/resnet34-142-best.pth"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data)

    layers = [
        ConvLayer("conv1.0", merge_bn=net.conv1[1]),
        QuantizedReLULayer(),

        ##
        
        ConvLayer("conv2_x.0.residual_function.0", merge_bn=net.conv2_x[0].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv2_x.0.residual_function.3", merge_bn=net.conv2_x[0].residual_function[4]),
        QuantizedReLULayer(),
        IdentityLayer("shortcut", 1),
        QuantizedReLULayer(),
        
        ConvLayer("conv2_x.1.residual_function.0", merge_bn=net.conv2_x[1].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv2_x.1.residual_function.3", merge_bn=net.conv2_x[1].residual_function[4]),
        QuantizedReLULayer(),
        IdentityLayer("shortcut", 7),
        QuantizedReLULayer(),        
        
        ConvLayer("conv2_x.2.residual_function.0", merge_bn=net.conv2_x[2].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv2_x.2.residual_function.3", merge_bn=net.conv2_x[2].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 13),
        QuantizedReLULayer(),
        
        ###
        
        PushInputsToStack(),
        ConvLayer("conv3_x.0.shortcut.0", merge_bn=net.conv3_x[0].shortcut[1]),
        QuantizedReLULayer(),
        PopInputsFromStack(),
        
        ConvLayer("conv3_x.0.residual_function.0", merge_bn=net.conv3_x[0].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv3_x.0.residual_function.3", merge_bn=net.conv3_x[0].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 22), # TODO: there is a conv for identity here
        QuantizedReLULayer(),
        
        ConvLayer("conv3_x.1.residual_function.0", merge_bn=net.conv3_x[1].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv3_x.1.residual_function.3", merge_bn=net.conv3_x[1].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 29), 
        QuantizedReLULayer(),
        
        ConvLayer("conv3_x.2.residual_function.0", merge_bn=net.conv3_x[2].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv3_x.2.residual_function.3", merge_bn=net.conv3_x[2].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 35), 
        QuantizedReLULayer(),        

        ConvLayer("conv3_x.3.residual_function.0", merge_bn=net.conv3_x[3].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv3_x.3.residual_function.3", merge_bn=net.conv3_x[3].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 41), 
        QuantizedReLULayer(),                
        
        ###
        
        PushInputsToStack(),
        ConvLayer("conv4_x.0.shortcut.0", merge_bn=net.conv4_x[0].shortcut[1]),
        QuantizedReLULayer(),        
        PopInputsFromStack(),
        
        ConvLayer("conv4_x.0.residual_function.0", merge_bn=net.conv4_x[0].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv4_x.0.residual_function.3", merge_bn=net.conv4_x[0].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 50), # TODO: there is a conv for identity here
        QuantizedReLULayer(),
        
        ConvLayer("conv4_x.1.residual_function.0", merge_bn=net.conv4_x[1].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv4_x.1.residual_function.3", merge_bn=net.conv4_x[1].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 57), 
        QuantizedReLULayer(),
        
        ConvLayer("conv4_x.2.residual_function.0", merge_bn=net.conv4_x[2].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv4_x.2.residual_function.3", merge_bn=net.conv4_x[2].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 63), 
        QuantizedReLULayer(),

        ConvLayer("conv4_x.3.residual_function.0", merge_bn=net.conv4_x[3].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv4_x.3.residual_function.3", merge_bn=net.conv4_x[3].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 69), 
        QuantizedReLULayer(),

        ConvLayer("conv4_x.4.residual_function.0", merge_bn=net.conv4_x[4].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv4_x.4.residual_function.3", merge_bn=net.conv4_x[4].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 75), 
        QuantizedReLULayer(),                        


        ConvLayer("conv4_x.5.residual_function.0", merge_bn=net.conv4_x[5].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv4_x.5.residual_function.3", merge_bn=net.conv4_x[5].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 81), 
        QuantizedReLULayer(),                        
        
        
        ###
        
        PushInputsToStack(),
        ConvLayer("conv5_x.0.shortcut.0", merge_bn=net.conv5_x[0].shortcut[1]),
        QuantizedReLULayer(),        
        PopInputsFromStack(),        
        
        ConvLayer("conv5_x.0.residual_function.0", merge_bn=net.conv5_x[0].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv5_x.0.residual_function.3", merge_bn=net.conv5_x[0].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 90), # TODO: there is a conv for identity here
        QuantizedReLULayer(),
        
        ConvLayer("conv5_x.1.residual_function.0", merge_bn=net.conv5_x[1].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv5_x.1.residual_function.3", merge_bn=net.conv5_x[1].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 97), 
        QuantizedReLULayer(),
        
        ConvLayer("conv5_x.2.residual_function.0", merge_bn=net.conv5_x[2].residual_function[1]),
        QuantizedReLULayer(),
        ConvLayer("conv5_x.2.residual_function.3", merge_bn=net.conv5_x[2].residual_function[4]),
        QuantizedReLULayer(),        
        IdentityLayer("shortcut", 103), 
        QuantizedReLULayer(),
        
        ###
        AvgPoolLayer("avg_pool"),
        FlattenLayer(),
        FCLayer("fc")
    ]
    
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=16, default_rescale=2**18, precision_assignment={"conv5_x.0.residual_function.0" : {"bits": 12}})

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=128)
    val_loader = cifar100_test_loader


    correct, total = 0, 0
    timing_breakdown = {}
    misc = {}
    for i, (input, target) in enumerate(val_loader):
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval

    acc = correct/total
    pprint.pprint(misc)
    print("ResNet34 Cifar100 acc=%f" % acc)
    assert(acc >= .73)

    print("Pass ResNet34 Cifar100 pretrained with acc=%f" % acc)

def test_vgg_cifar100_act_quantized():
    net = vgg16_bn()
    net.eval()
    
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_cifar100/checkpoint/vgg16/vgg16-200-regular.pth"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data)

    layers = [
        ConvLayer("features.0",  merge_bn=net.features[1]),
        QuantizedReLULayer(),
        ConvLayer("features.3", merge_bn=net.features[4]),
        QuantizedReLULayer(),
        AvgPoolLayer("features.6"),

        ConvLayer("features.7",  merge_bn=net.features[8]),
        QuantizedReLULayer(),
        ConvLayer("features.10", merge_bn=net.features[11]),
        QuantizedReLULayer(),
        AvgPoolLayer("features.13"),

        ConvLayer("features.14",  merge_bn=net.features[15]),
        QuantizedReLULayer(),
        ConvLayer("features.17", merge_bn=net.features[18]),
        QuantizedReLULayer(),
        ConvLayer("features.20", merge_bn=net.features[21]),
        QuantizedReLULayer(),        
        AvgPoolLayer("features.23"),

        ConvLayer("features.24",  merge_bn=net.features[25]),
        QuantizedReLULayer(),
        ConvLayer("features.27", merge_bn=net.features[28]),
        QuantizedReLULayer(),
        ConvLayer("features.30", merge_bn=net.features[31]),
        QuantizedReLULayer(),        
        AvgPoolLayer("features.33"),
        
        ConvLayer("features.34",  merge_bn=net.features[35]),
        QuantizedReLULayer(),
        ConvLayer("features.37", merge_bn=net.features[38]),
        QuantizedReLULayer(),
        ConvLayer("features.40", merge_bn=net.features[41]),
        QuantizedReLULayer(),        
        AvgPoolLayer("features.43"),

        FlattenLayer("flatten"),

        FCLayer("classifier.0"),
    ]

    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=512)
    val_loader = cifar100_test_loader

    layers = load_quantized_neural_network_from_pytorch(net, layers, 
                                                        default_precision=16, default_rescale=2**13,
                                                        precision_assignment={"features.34" : {"bits": 12}})
    
    
    correct, total = 0, 0
    timing_breakdown = {}
    misc = {}
    for i, (input, target) in enumerate(val_loader):


        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

        break # Comment for full eval

    acc = correct/total
    pprint.pprint(misc)
    print("ResNet34 Cifar100 acc=%f" % acc)
    assert(acc >= .69)

    print("Pass VGG16 Cifar100 pretrained with acc=%f" % acc)                                    

def full_test():

    # Blocks
    test_run_resnet_block_32bit()
    test_resnet20_sanity_32bit()
    test_run_quantized_fc_network_32bit()
    test_run_conv_network_32bit()
    

    # Networks    
    test_resnet32_cifar10_pretrained_32bit()
    test_resnet32_cifar10_pretrained_quantized()

    test_resnet32_cifar10_pretrained_32bit()
    test_resnet32_cifar10_pretrained_quantized()
    test_resnet_34_cifar100_quantized()    
    test_mnist_quantized_pretrained_32bit()
    test_mnist_bn_quantized_pretrained_32bit()
    test_mnist_quantized_pretrained(default_precision=3, default_rescale=2**5)
    test_mnist_quantized_pretrained_precision_assignments_1()
    test_mnist_bn_quantized_pretrained_precision_assignments()    
    test_vgg_cifar100_pretrained()

def dev_test():
    #test_mnist_act_quantized()
    #test_resnet32_cifar10_act_quantized()
    #test_resnet34_cifar100_act_quantized()
    test_vgg_cifar100_act_quantized()
    

if __name__=="__main__":

    dev_test()
    #full_test()

# File contains configuration params to achieve different accuracy vs bits scores

import sys
import pprint
from network.neural_network import *
from network.quantized_network import *
import copy
import multiprocessing
import json

groot = get_git_root(__file__)

# Dataset
transform_mnist=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])    
dataset2_mnist = datasets.MNIST('/tmp', train=False,
                                transform=transform_mnist,
                                download=True)
test_loader_mnist = torch.utils.data.DataLoader(dataset2_mnist, batch_size=1000)

# Cifar10 dataset
normalize_cifar10 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader_cifar10 = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='/tmp', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize_cifar10,
    ])),
    batch_size=128, shuffle=False,
    pin_memory=True)

# Cifar100 dataset
transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
])
cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_cifar100)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=128)


def evaluate_mnist(layers):

    # Evaluate
    tstart = time.time()
    timing_breakdown = {}
    misc = {}
    correct, total = 0, 0
    for data, target in test_loader_mnist:
        data = data.numpy()
        target = target.numpy()

        pred = execute_quantized_neural_network(layers, data, timing_breakdown, misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

    tend = time.time()
    telapsed = tend-tstart

    acc = correct / total

    return acc

def evaluate_cifar10(layers):
    correct, total = 0, 0
    misc = {}
    for i, (input, target) in enumerate(val_loader_cifar10):
        print("Cifar10 Iteration", i, len(val_loader_cifar10))
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, misc=misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]

    acc = correct/total
    
    return acc

def evaluate_cifar100(layers):
    correct, total = 0, 0
    misc = {}
    for i, (input, target) in enumerate(cifar100_test_loader):
        print("Cifar100 Iteration", i, len(cifar100_test_loader))
        target = target.numpy()
        input_var = input.numpy()

        pred = execute_quantized_neural_network(layers, input_var, misc=misc)
        pred = np.argmax(pred, axis=1)

        correct += np.sum(pred == target)
        total += pred.shape[0]
        
    acc = correct/total
    
    return acc

def mnist():

    def get_mnist_layers(precision, rescale, relu_bits):
    
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

        if relu_bits is not None:
            layers = replace_relu_with_quantized_relu(layers, relu_bits=relu_bits)

        net = MnistNet()
        net.eval()    
        pretrained_model_path = groot + "/src/train/mnist/mnist_cnn.pt"
        net.load_state_dict(torch.load(pretrained_model_path))    

        exec_layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                                 default_precision=precision,
                                                                 default_rescale=rescale)
        return exec_layers

    # Default precision rescale that does well
    precision, rescale = 12, 2**12

    for relu_bits in [None, 12, 11, 10, 9, 8, 7, 6, 5, 4]:    
        acc = evaluate_mnist(get_mnist_layers(precision, rescale, relu_bits))
        print("relu bits: %s, acc: %f" % (str(relu_bits), acc))

def cifar10():


    net = torch.nn.DataParallel(resnet32())
    net.eval()
    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data["state_dict"])
    

    def get_layers(precision, rescale, relu_bits):
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

        if relu_bits is not None:
            layers = replace_relu_with_quantized_relu(layers, relu_bits=relu_bits)

        exec_layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                                 default_precision=precision,
                                                                 default_rescale=rescale)
        return exec_layers

    # Default precision rescale that does well
    precision, rescale = 13, 2**14

    for relu_bits in [None, 12, 11, 10, 9, 8, 7, 6, 5, 4]:    
        acc = evaluate_cifar10(get_layers(precision, rescale, relu_bits))
        print("relu bits: %s, acc: %f" % (str(relu_bits), acc))

def cifar100():

    net = resnet34()
    net.eval()

    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_cifar100/checkpoint/resnet34/resnet34-142-best.pth"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data)
    

    def get_layers(precision, rescale, relu_bits, precision_assignment={}):
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
        
        
        if relu_bits is not None:
            layers = replace_relu_with_quantized_relu(layers, relu_bits=relu_bits)

        exec_layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                                 default_precision=precision,
                                                                 default_rescale=rescale,
                                                                 precision_assignment=precision_assignment)
        return exec_layers

    # Default precision rescale that does well
    precision, rescale = 16, 2**18
    precision_assignment={"conv5_x.0.residual_function.0" : {"bits": 12}}

    for relu_bits in [None, 12, 11, 10, 9, 8, 7, 6, 5, 4]:
        acc = evaluate_cifar100(get_layers(precision, rescale, relu_bits,
                                           precision_assignment=precision_assignment))
        print("relu bits: %s, acc: %f" % (str(relu_bits), acc))        


def cifar100_vgg():

    net = vgg16_bn()
    net.eval()

    pretrained_model_path = get_git_root(__file__) + "/src/train/resnet/pytorch_cifar100/checkpoint/vgg16/vgg16-200-regular.pth"
    save_data = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    net.load_state_dict(save_data)
    

    def get_layers(precision, rescale, relu_bits, precision_assignment={}):

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
        
        
        if relu_bits is not None:
            layers = replace_relu_with_quantized_relu(layers, relu_bits=relu_bits)
            
        exec_layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                                 default_precision=precision,
                                                                 default_rescale=rescale,
                                                                 precision_assignment=precision_assignment)
        return exec_layers

    # Default precision rescale that does well
    precision, rescale = 16, 2**13
    precision_assignment={"features.34" : {"bits": 12}}

    for relu_bits in [None, 12, 11, 10, 9, 8, 7, 6, 5, 4]:    
        acc = evaluate_cifar100(get_layers(precision, rescale, relu_bits,
                                           precision_assignment=precision_assignment))
        print("relu bits: %s, acc: %f" % (str(relu_bits), acc))        
                
if __name__=="__main__":

    mode = "mnist"
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
    assert(mode in ["mnist", "cifar10", "cifar100", "cifar100_vgg"])
    
    print(mode)
    if mode == "mnist":
        mnist()
    if mode == "cifar10":
        cifar10()
    if mode == "cifar100":
        cifar100()
    if mode == "cifar100_vgg":
        cifar100_vgg()

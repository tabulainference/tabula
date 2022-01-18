import sys
from private_inference.protocol import *
from private_inference.delphi import Delphi
from private_inference.tabula import Tabula
import argparse
from network.neural_network import *
from network.quantized_network import *
import numpy as np
import json
import pprint
import git
import os

root_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"

parser = argparse.ArgumentParser()
parser.add_argument('--bits', required=True, type=int) # bits for GCs (currently inoperable for tabula)
parser.add_argument('--mode', required=True, choices=[0,1], type=int)
parser.add_argument('--task', required=True, choices=["mnist", "cifar10", "cifar100", "cifar100_vgg"])
parser.add_argument("--server_host", default="127.0.0.1")
parser.add_argument("--method", default="gc", choices=["gc", "tabula"])

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def get_kwargs_for_method_task(method, task, bits):
    if method == "tabula":
        return {}

    gitroot = get_git_root(__file__)

    circuits_file = ""
    

    if task == "mnist":        
        circuits_file = "%s/src/experiments/delphi/lenet_mnist" % gitroot
    if task == "cifar10":
        circuits_file = "%s/src/experiments/delphi/resnet_32_cifar10" % gitroot
    if task == "cifar100":
        circuits_file = "%s/src/experiments/delphi/resnet_34_cifar100" % gitroot
    if task == "cifar100_vgg":
        circuits_file = "%s/src/experiments/delphi/vgg_cifar100" % gitroot

    circuits_file = "%s_q=%d" % (circuits_file, bits)

    print(circuits_file)
    assert(os.path.exists(circuits_file))

    return {"load_relu_circuits_from": circuits_file, "q" : bits}

def get_network_for_benchmark(task, method):
    if task == "mnist":
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
        ]    

        # Model
        net = MnistNet()
        net.eval()
        net.double()

        if method == "gc":
            layers = load_neural_network_from_pytorch(net, layers)
        else:
            layers = replace_relu_with_quantized_relu(layers)
            layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                                default_precision=12,
                                                                default_rescale=2**12)

        return layers
    elif task == "cifar10":

        net = torch.nn.DataParallel(resnet32())
        net.eval()
        net.double()                
        
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

        if method == "gc":
            layers = load_neural_network_from_pytorch(net, layers)
        else:
            layers = replace_relu_with_quantized_relu(layers) 
            layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                                default_precision=13,
                                                                default_rescale=2**14)
        return layers
    elif task == "cifar100":
        net = resnet34()
        net.eval()
        net.double()        

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
        
        if method == "gc":
            layers = load_neural_network_from_pytorch(net, layers)
        else:
            layers = replace_relu_with_quantized_relu(layers) 
            layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=13, default_rescale=2**18)
        return layers

    elif task == "cifar100_vgg":
        net = vgg16_bn()
        net.eval()
        net.double()

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

        if method == "gc":
            layers = load_neural_network_from_pytorch(net, layers)
        else:
            layers = replace_relu_with_quantized_relu(layers) 
            layers = load_quantized_neural_network_from_pytorch(net, layers, default_precision=16, default_rescale=2**13)
        return layers            

    else:        
        assert(0)

def get_inp_for_task(task):
    if task == "mnist":
        inp = np.random.normal(0, 1, size=(1, 1, 28, 28))
    elif task == "cifar10" or task == "cifar100" or task == "cifar100_vgg":
        inp = np.random.normal(0, 1, size=(1, 3, 32, 32))
    else:
        assert(0)
    inp = Input(inp)
    return inp        

if __name__=="__main__":
    args = parser.parse_args()        
    pi_method = Delphi if args.method == "gc" else Tabula
    print("Using %s" % str(pi_method))

    layers = get_network_for_benchmark(args.task, args.method)

    pi_method_kwargs = (
            {"fake_relu" : False} if args.method == "gc" else {"fake_scale" : False})
    
    p = pi_method(mode=args.mode, neural_network=layers,
                  host=args.server_host, **pi_method_kwargs)

    inp = get_inp_for_task(args.task)
    kwargs = get_kwargs_for_method_task(args.method, args.task, args.bits)

    misc = {}
    
    p.initialize()
    p.preprocess(inp, **kwargs)

    p.start_sniffing()
    p.barrier()
    t_start = time.time()
    p.run(inp, misc=misc)
    t_end = time.time()
    p.stop_sniffing()

    misc["total_elapsed_time"] = t_end-t_start
    misc["total_bytes_communicated_online"] = p.get_communicated_bytes()
    misc["total_bytes_communicated_gc_online"] = p.get_communicated_bytes_gc()    

    os.chdir(root_path)
    modestr = "server" if args.mode == PrivateInferenceProtocol.SERVER else "client"
    f_out = "%s_%s_b=%d_benchmark_out_%s" % (args.task, args.method, args.bits, modestr)
    misc["mode"] = modestr
    with open(f_out, "w") as f:
        json.dump(misc, f)
    pprint.pprint(misc)

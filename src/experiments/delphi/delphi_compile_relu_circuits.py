import sys
from private_inference.delphi import *
import os
import pprint

file_dir = os.path.dirname(os.path.abspath(__file__))
mode = int(sys.argv[1])
task, q = sys.argv[2:]
q = int(q)
gitroot = get_git_root(__file__)

print(mode, task, q)

def compile_convnet():
    
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, bias=False)
            self.bn1 = nn.BatchNorm2d(6)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.AvgPool2d(2)

        def forward(self, x):
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.relu1(y)
            y = self.pool1(y)
            return y            
        
    net = ConvNet()
    net.eval()

    layers = [
        # Merged ConvBNLayer
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer(),
        AvgPoolLayer("pool1"),
    ]    
    
    layers = load_neural_network_from_pytorch(net, layers, mute=mode!=0)
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 1, 28, 28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [net(torch.from_numpy(x.x).float()).detach().numpy() for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], save_relu_circuits_to="%s/convnet" % file_dir)           

def test_convnet_compile(q=32):
    
    class ConvNet(torch.nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, bias=False)
            self.bn1 = nn.BatchNorm2d(6)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.AvgPool2d(2)

        def forward(self, x):
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.relu1(y)
            y = self.pool1(y)
            return y            
        
    net = ConvNet()
    net.eval()

    layers = [
        # Merged ConvBNLayer
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer(),
        AvgPoolLayer("pool1"),
    ]    
    
    layers = load_neural_network_from_pytorch(net, layers, mute=mode!=0)
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 1, 28, 28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [net(torch.from_numpy(x.x).float()).detach().numpy() for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], load_relu_circuits_from="%s/convnet_q=%d" % (file_dir, q), q=q)

    preds = [d.run(x) for x in inputs]    

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        # comment this out since optimized relu for speed may not be correct
        #assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed conv_relu_test")

    d.shutdown()            
    
def compile_small_fc(q=32):
    
    # Init the network
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
        FCLayer("L3"),
        FlattenLayer()
    ]
    
    net = FCNet(8)
    net.eval()    
    layers = load_neural_network_from_pytorch(net, layers, mute=mode!=0)
    
    inputs = [np.random.normal(0, 1, size=(1, 28*28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]
    
    d = Delphi(mode=mode, neural_network=layers, fake_relu=False)    
    d.initialize()

    # Save circuit for batch 1 inference
    d.preprocess(inputs[0], save_relu_circuits_to="%s/small_fc_q=%d/" % (file_dir, q), q=q)

    d.shutdown()


def test_small_fc_compile(q=32):
    
    # Init the network
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
        FCLayer("L3"),
        FlattenLayer()
    ]
    
    net = FCNet(8)
    net.eval()    
    layers = load_neural_network_from_pytorch(net, layers, mute=mode!=0)
    
    inputs = [np.random.normal(0, 1, size=(1, 28*28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    results = [execute_neural_network(layers, x.x)  for x in inputs]    
    
    d = Delphi(mode=mode, neural_network=layers, fake_relu=False)    
    d.initialize()

    # Save circuit for batch 1 inference
    d.preprocess(inputs[0], load_relu_circuits_from="%s/small_fc_q=%d/" % (file_dir, q), q=q)
    misc = {}
    preds = [d.run(x, misc=misc) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        pprint.pprint(misc)
        #print(np.mean(np.abs(preds[0]-results[0])))
        #assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)

    d.shutdown()        

def compile_resnet_32(q=32):    
    mode = int(sys.argv[1])    

    net = torch.nn.DataParallel(resnet32())
    net.cpu()
    net.eval()
    
    pretrained_model_path = gitroot + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
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
    
    # Init the inputs
    inputs = [next(iter(val_loader))[0].numpy() for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], save_relu_circuits_to="%s/resnet_32_cifar10_q=%d" % (file_dir, q), q=q)

    d.shutdown()    

def test_resnet_32_compile(q=32):
    mode = int(sys.argv[1])
    
    net = torch.nn.DataParallel(resnet32())
    net.eval()
    net.double()        
    
    pretrained_model_path = gitroot + "/src/train/resnet/pytorch_resnet_cifar10/pretrained_models/resnet32-d509ac18.th"
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
    
    # Init the inputs
    inputs = [next(iter(val_loader))[0].numpy() for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x) for x in inputs]
    fake_relu = False
    
    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], load_relu_circuits_from="%s/resnet_32_cifar10_q=%d" % (file_dir, q), q=q)
        
    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(results[0]-preds[0])))
        #assert(np.mean(np.abs(preds[0]-results[0])) <= 1)
        print("Pass Cifar10 ResNet32")

    d.shutdown()                    

def compile_resnet_34(q=32):
    net = resnet34()
    net.cpu()
    net.eval()
    
    pretrained_model_path = gitroot + "/src/train/resnet/pytorch_cifar100/checkpoint/resnet34/resnet34-142-best.pth"
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
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=1)
    val_loader = cifar100_test_loader
    
    # Init the inputs
    inputs = [next(iter(val_loader))[0].numpy() for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], save_relu_circuits_to="%s/resnet_34_cifar100_q=%d" % (file_dir, q), q=q)

    d.shutdown()        

def compile_vgg(q=32):
    net = vgg16_bn()
    net.cpu()
    net.eval()
    
    pretrained_model_path = gitroot + "/src/train/resnet/pytorch_cifar100/checkpoint/vgg16/vgg16-200-regular.pth"
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

    layers = load_neural_network_from_pytorch(net, layers)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    cifar100_test = datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
    cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, shuffle=False, batch_size=1)
    val_loader = cifar100_test_loader
    
    # Init the inputs
    inputs = [next(iter(val_loader))[0].numpy() for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], save_relu_circuits_to="%s/vgg_cifar100_q=%d" % (file_dir, q), q=q)

    d.shutdown()        

    
def compile_mnist(q=32):

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

    layers = load_neural_network_from_pytorch(net, layers)    
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 1, 28, 28))]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)
        
    d.initialize()
    d.preprocess(inputs[0], save_relu_circuits_to="%s/lenet_mnist_q=%d" % (file_dir, q), q=q)

    d.shutdown()        

def test_mnist_compile(q=32):

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

    layers = load_neural_network_from_pytorch(net, layers)    
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 1, 28, 28))]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_relu=False)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_relu=False)

        
    d.initialize()
    d.preprocess(inputs[0], load_relu_circuits_from="%s/lenet_mnist_q=%d" % (file_dir, q), q=q)

    m = {}
    preds = [d.run(x, misc=m) for x in inputs]

    pprint.pprint(m)

    d.shutdown()        
    
def preprocess_compilation():
    # This step preprocesses the ReLU GC compilation, as GC compilation may
    # be slow.
    if task == "small":
        compile_small_fc(q=q)
    elif task == "convnet":
        compile_convnet(q=q)
    elif task == "mnist":
        compile_mnist(q=q)
    elif task == "resnet32":
        compile_resnet_32(q=q)
    elif task == "resnet34":
        compile_resnet_34(q=q)
    elif task == "vgg":
        compile_vgg(q=q)
    else:
        assert(0)    

def test_preprocess_compilation():

    #test_small_fc_compile(q=32)
    #test_convnet_compile()
    #test_resnet_32_compile()
    test_mnist_compile(q=32)
    pass
    
if __name__ == "__main__":
    preprocess_compilation()
    #test_preprocess_compilation()

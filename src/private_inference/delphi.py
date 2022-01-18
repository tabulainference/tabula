# This file contains the following:
#
# - Code for the Delphi private inference algorithm (baseline).
#   Supports ResNet-32/34 / convnet for mnist, cifar10, cifar100
#
# - Note that we do not use FHE in the preprocessing phase
#   since we don't benchmark this step (and is irrelevant to the
#   online phase).
#
# - We use finite field 2^32-1 as specified in the Delphi paper
#
# - Note that Delphi also quantizes input values, hence there will
#   be some error. Furthermore, truncation yields some error as well.
#   However, we assume that Delphi achieves full precision quality.

from private_inference.protocol import PrivateInferenceProtocol, Input
from network.neural_network import *
import torch
import copy
import os
from garbled_circuits.relu import GC_ReLU
from garbled_circuits.relu_optimized import GC_ReLU_opt
import flock
import distutils

file_dir = os.path.dirname(os.path.abspath(__file__))

def reset_seed():
    np.random.seed(1000)
    torch.manual_seed(422123)

class Delphi(PrivateInferenceProtocol):
    def __init__(self, mode=None, neural_network=None,
                 host="127.0.0.1", port="5556",
                 field=2**32-1,
                 #field=14293, # Does not work well for delphi due to truncation errors
                 scale=2**7,
                 fake_relu=True):
        super().__init__("delphi", mode=mode, neural_network=neural_network,
                         host=host, port=port)
        self.field = field
        self.scale = scale
        self.relus = {}
        self.fake_relu = fake_relu
        
    def encode(self, x):

        x = np.round(x*self.scale).astype(np.int64)

        # Encode into finite field
        assert(np.max(np.abs(x)) < self.field)

        np.putmask(x, x < 0, self.field+x)

        return x

    def decode(self, x):
        
        x = x % self.field
        np.putmask(x, x > (self.field-1)//2, x-self.field)

        x = x/(self.scale)

        return x

    def initialize(self):
        
        if self.mode == PrivateInferenceProtocol.CLIENT:
            return

        # Use initialization process to convert network weights and biases into
        # fixed point (server only), then encode into the finite field
        for i, layer in enumerate(self.layers):
            
            if type(layer) == FCLayer or type(layer) == ConvLayer:
                module = layer.pytorch_module                

                # Turn to fixed point with scale
                weight = module.weight.detach().numpy()                
                weight = self.encode(weight)
                module.weight = torch.nn.parameter.Parameter(torch.from_numpy(weight).long(), requires_grad=False)

                if module.bias is not None:
                    bias = module.bias.detach().numpy()
                    bias = self.encode(bias)
                    module.bias = torch.nn.parameter.Parameter(torch.from_numpy(bias).long(), requires_grad=False)

                # Reload
                layer.load(module)


    def preprocess(self, input,
                   save_relu_circuits_to=None,
                   load_relu_circuits_from=None,
                   q=32):

        # Note that q is relu quantization factor

        # Note that, in its current form, we do not fully implement
        # the preprocessing phase with linearly homomorphic encryption,
        # as this is irrelevant to the online phase, and we do not compare
        # preprocessing overheads between tabula and delphi.

        # Execute the network once to track shapes
        misc = {}
        execute_neural_network(self.layers, input.x, misc=misc)

        info = {}
        x = input.x

        # Load compiled relu circuits
        if load_relu_circuits_from is not None and not self.fake_relu:
            gc_dir = "%s/../garbled_circuits/" % file_dir
            os.chdir(gc_dir)
            lockfile = "/tmp/lockfile" 
            with open(lockfile, 'w') as fp:
                with flock.Flock(fp, flock.LOCK_EX) as lock:
                    load_relu_ciruits_from = "%s/%s" % (load_relu_circuits_from, "client" if self.mode == PrivateInferenceProtocol.CLIENT else "server")
                    assert(os.path.exists(load_relu_circuits_from))
                    distutils.dir_util.copy_tree(load_relu_ciruits_from,
                                                 "%s/Programs/" % gc_dir,
                                                 preserve_mode=0)                        

        # Preprocess the additive complements of linear layers
        for i, layer in enumerate(self.layers):

            print("Preprocessed %d of %d" % (i, len(self.layers)))

            x_shape = misc["shapes"][i]
            
            if type(layer) == FCLayer or type(layer) == ConvLayer:
                
                info[i] = {}
                
                if self.mode == PrivateInferenceProtocol.CLIENT:
                    
                    # Generate additive nonce for this layer
                    additive_nonce = np.random.randint(0, self.field, size=x_shape)
                    #additive_nonce = np.zeros_like(additive_nonce)

                    # This part should be done w/ linear homomorphic encryption to preserve privacy, but
                    # we do it without it since it's irrelevant to online phase
                    self.send_npy(additive_nonce)
                    additive_complement = self.recv_npy()

                    info[i]["additive_nonce"] = additive_nonce
                    info[i]["additive_complement"] = additive_complement                    

                elif self.mode == PrivateInferenceProtocol.SERVER:

                    additive_nonce = self.recv_npy()

                    # Perform linear layer on additive nonce
                    no_bias_module = copy.deepcopy(layer.pytorch_module)
                    no_bias_module.bias = None
                    
                    #additive_complement = additive_nonce.dot(layer.weight_mat) % self.field
                    additive_complement = no_bias_module(torch.from_numpy(additive_nonce).long()).numpy() % self.field
                    
                    additive_nonce_output = np.random.randint(0, self.field, size=additive_complement.shape) % self.field
                    # This basically hacks the truncation process to nearly always work, but is insecure
                    #additive_nonce_output = np.ones_like(additive_nonce_output) * ((self.field-1)//2) 
                    additive_complement = (additive_complement + additive_nonce_output) % self.field

                    info[i]["additive_nonce_output"] = additive_nonce_output
                    self.send_npy(additive_complement)
                else:
                    raise Exception("Unknown mode")

            if type(layer) == ReLULayer:

                if self.fake_relu:
                    continue

                relu_gc_op = GC_ReLU_opt(layer.name + "_%d" % i, x_shape,
                                         self, is_server=self.mode == PrivateInferenceProtocol.SERVER,
                                         server_host=self.host,
                                         port=str(8700+30*i), q=q
                                         )                
                relu_gc_op.preprocess(save_relu_circuits_to=save_relu_circuits_to,
                                      compile_gc=load_relu_circuits_from is None)
                self.relus[i] = relu_gc_op
                

        input.info = info

    def run(self, input, verbose=True, misc={}):
              
        info = input.info

        # At the end, `x` held by both parties should sum to the output
        x = self.encode(input.x)
        initial = x

        outputs = {}
        stack = []
        total_relus = 0

        for i, layer in enumerate(self.layers):

            t_start = time.time()

            print("Executing layer %s (%d of %d)" % (layer, i, len(self.layers)))
            
            last_layer = i == (len(self.layers)-1)

            if type(layer) == PushInputsToStack:
                stack.append(x)
            elif type(layer) == PopInputsFromStack:
                x = stack.pop(-1)
            elif type(layer) == FCLayer or type(layer) == ConvLayer:
                if self.mode == PrivateInferenceProtocol.CLIENT:

                    additive_nonce = info[i]["additive_nonce"]
                    additive_complement = info[i]["additive_complement"]

                    x_encr = (x+additive_nonce) % self.field
                    self.send_npy(x_encr)

                    x = -additive_complement

                    # This is the truncation trick (mirrored by other party).
                    # With probability 1-value/field output is within 1 bit from the actual.
                    # If field is large, then this is nearly always true.
                    # Trick from secureml / delphi.

                    x //= self.scale

                elif self.mode == PrivateInferenceProtocol.SERVER:
                    
                    additive_nonce_output = info[i]["additive_nonce_output"]

                    x_encr = self.recv_npy() + x 

                    no_bias_module = copy.deepcopy(layer.pytorch_module)
                    no_bias_module.bias = None

                    # Note: be very careful about overflows
                    x = no_bias_module(torch.from_numpy(x_encr).long()).numpy() % self.field
                    x = (x + additive_nonce_output) % self.field
                    
                    x //= self.scale

                    if layer.bias is not None:
                        bias = layer.bias
                        if type(layer) == ConvLayer:
                            bias = bias.reshape(1, bias.shape[0], 1, 1)
                        x = (x + bias) % self.field
                    
            if type(layer) == ReLULayer:

                if self.fake_relu:
                    # Insecure fake relu for testing
                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        other_x = self.recv_npy()
                        x = (other_x + x) % self.field
                        x[x > (self.field-1)//2] = 0

                    elif self.mode == PrivateInferenceProtocol.SERVER:
                        self.send_npy(x)
                        x = np.zeros_like(x)
                else:
                    # GC ReLU -- note this skips the one-time pad for simplicity
                    relu_gc_op = self.relus[i]
                    np.putmask(x, x > (self.field-1)//2, x-self.field)
                    x = relu_gc_op.run(x)

                    if self.mode == PrivateInferenceProtocol.SERVER:
                        x = np.zeros_like(x)

                    # Track some extra relu stats
                    if "data_sent" in relu_gc_op.stats:
                        misc["%s_%d_data_sent" % (str(layer), i)] = relu_gc_op.stats["data_sent"]

                misc["%s_%d_n_elements" % (str(layer), i)] = float(np.prod(x.shape))
                total_relus += float(np.prod(x.shape))

            if type(layer) == FlattenLayer:
                x = x.reshape((x.shape[0], -1))

            if type(layer) == AvgPoolLayer:
                x = np.round(layer.pytorch_module(torch.from_numpy(x)).numpy()*layer.kernel_size**2) % self.field
                x = (x // layer.kernel_size**2) % self.field

            if type(layer) == IdentityLayer:
                other = initial if layer.indx_identity == -1 else outputs[layer.indx_identity]

                if other.shape != x.shape:
                    # This happens with cifar10 resnet
                    p = (x.shape[1]-other.shape[1])//2
                    other = np.pad(other[:,:,::2,::2], ((0,0), (p,p), (0,0), (0,0)), mode="constant", constant_values=0)
                    assert(other.shape == x.shape)

                x = x + other


            # Track outputs per layer
            outputs[i] = x

            t_end = time.time()
            elapsed = t_end-t_start

            misc[str(layer)+"_"+str(i)] = elapsed
            misc["total_relus"] = total_relus

        if self.mode == PrivateInferenceProtocol.SERVER:
            self.send_npy(x)
            return None
        else:
            other_x = self.recv_npy()
            result = self.decode((x + other_x))
            return result

def fc_relu_test():
    reset_seed()
    mode = int(sys.argv[1])

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
    
    net = FCNet(32)
    net.eval()
    net.double()
    layers = load_neural_network_from_pytorch(net, layers, mute=mode!=0)
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(32, 28*28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_neural_network(layers, x.x)  for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers)
        
    d.initialize()
    for inp in inputs:
        d.preprocess(inp)
        
    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed fc_relu_test")

    d.shutdown()                

def conv_relu_test():
    reset_seed()    
    mode = int(sys.argv[1])

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
    net.double()    

    layers = [
        # Merged ConvBNLayer
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer(),
        AvgPoolLayer("pool1"),
    ]    
    
    layers = load_neural_network_from_pytorch(net, layers, mute=mode!=0)
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(4, 1, 28, 28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [net(torch.from_numpy(x.x).double()).detach().numpy() for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers)
        
    d.initialize()
    for inp in inputs:
        d.preprocess(inp)
        
    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed conv_relu_test")

    d.shutdown()                

def resnet_block_test():
    reset_seed()    
    
    mode = int(sys.argv[1])    

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
    layers = load_neural_network_from_pytorch(net, layers)

    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 32, 32, 32)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [net(torch.from_numpy(x.x).double()).detach().numpy() for x in inputs]    

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers)
        
    d.initialize()
    for inp in inputs:
        d.preprocess(inp)
        
    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Pass ResNet block")

    d.shutdown()                

def resnet32_cifar10_pretrained_test():
    reset_seed()    

    mode = int(sys.argv[1])    

    net = torch.nn.DataParallel(resnet32())
    net.eval()
    net.double()        
    
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
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers)
        
    d.initialize()
    for inp in inputs:
        d.preprocess(inp)
        
    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(results[0]-preds[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1)
        print("Pass Cifar10 ResNet32")

    d.shutdown()                

def resnet34_cifar100_pretrained_test():
    reset_seed()    

    mode = int(sys.argv[1])    

    net = resnet34()
    net.eval()
    net.double()        
    
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
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers)
        
    d.initialize()
    for inp in inputs:
        d.preprocess(inp)
        
    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(results[0]-preds[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1)
        print("Pass Cifar100 ResNet34")

    d.shutdown()

def test_relu():
    reset_seed()
    mode = int(sys.argv[1])

    # Init the network
    class ReLUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(x)
            return x        

    layers = [
        ReLULayer("ReLU"),
    ]
    
    net = ReLUNet()
    net.eval()    
    layers = load_neural_network_from_pytorch(net, layers)
    
    # Init the inputs
    inp = np.random.normal(0, 1, size=(1, 100000))
    inp = Input(inp)

    golden = execute_neural_network(layers, inp.x)

    if mode == PrivateInferenceProtocol.SERVER:

        # Server        
        d = Delphi(mode=PrivateInferenceProtocol.SERVER, neural_network=layers)
        
    else:

        inp.clear()

        # Client        
        d = Delphi(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers)

    d.initialize()
    d.preprocess(inp)
    misc = {}
            
    pred = d.run(inp, misc=misc)

    print(misc)            
    
    if mode == PrivateInferenceProtocol.CLIENT:
        print(pred)
        print(golden)
        print(np.mean(np.abs(pred-golden)))
        assert(np.mean(np.abs(pred-golden)) <= 1e-1)
        print("Passed fc_relu_test")

    d.shutdown()    
            
if __name__=="__main__":

    fc_relu_test()
    conv_relu_test()
    resnet_block_test()
    resnet32_cifar10_pretrained_test()    
    #resnet34_cifar100_pretrained_test()

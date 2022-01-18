# This file contains the following
#
# - Code for the Tabula algorithm, which uses hash tables to
#   reduce ReLU communication by > 50x from prior state of the art.
#   Supports resnet-32/34 and convnet on mnist, cifar10, cifar100.
#
# - Tabula algorithm yields _exact_ outputs to simulation
#   execution code in networks/quantized_network 
#
# - Use networks/quantized_network to experiment with
#   per layer quantization assignments to ensure high quality
#
# - Ensure that accumulator bits reported by networks/quantized_network
#   is less than the bits of the finite field (see `FIELDS` variable).
#   If this condition is satisfied, the code will yield _exact_ predictions
#   to that in networks/quantized_network
#
# Note that to benchmark communication use tshark:
# sudo tshark -z conv,ip -p -f "port 5556" -i any

from private_inference.protocol import PrivateInferenceProtocol, Input
import network.quantized_network as quantized_network
from network.quantized_network import *
from network.neural_network import *
import random
import torch
import copy
import sys
import hmac
import hashlib
import itertools
from multiprocessing import Process, Manager, Pool
import tqdm
import json

def reset_seed():
    np.random.seed(1000)
    torch.manual_seed(422123)

class Tabula(PrivateInferenceProtocol):
    def __init__(self, mode=None, neural_network=None,
                 host="127.0.0.1", port="1234",
                 fake_scale=False, field=2**42-1):
        super().__init__("delphi", mode=mode, neural_network=neural_network,
                         host=host, port=port)
        self.field = field
        self.fake_scale = fake_scale
        self.misc = {}
        self.relu_table_size = None

    def encode(self, x):

        #x = np.int64(x)        
        
        # Assume x is int
        np.putmask(x, x < 0, self.field+x)
        
        return x

    def decode(self, x):
        x = x % self.field
        np.putmask(x, x > (self.field-1)//2, x-self.field)
        
        return x
        
    def initialize(self):

        # IMPORTANT: THIS STEP SHOULD BE DONE SECURELY BY A TRUSTED THIRD PARTY.
        
        # Server returns
        if self.mode != PrivateInferenceProtocol.SERVER:
            return        


    def preprocess(self, input):        

        # Note that, in its current form, we do not fully implement
        # the preprocessing phase as this is irrelevant to the online phase, and we do not compare
        # preprocessing overheads between tabula and gc.

        # Execute the network once to track shapes and quant data ranges
        misc = {}
        execute_quantized_neural_network(self.layers, input.x, misc=misc)
        self.misc = misc

        info = {}
        x = input.x

        # Preprocess the additive complements of linear layers
        for i, layer in enumerate(self.layers):

            x_shape = misc["shapes"][i]
            
            if type(layer) == QuantizedFCLayer or type(layer) == QuantizedConvLayer:
                
                info[i] = {}
                
                if self.mode == PrivateInferenceProtocol.CLIENT:
                    
                    # Generate additive nonce for this layer
                    additive_nonce = np.random.randint(0, self.field, size=x_shape, dtype=np.int64)

                    # This part should be done w/ linear homomorphic encryption to preserve privacy, but
                    # we do it without it since it's irrelevant to online phase
                    self.send_npy(additive_nonce)
                    additive_complement =self.recv_npy()

                    info[i]["additive_nonce"] = additive_nonce
                    info[i]["additive_complement"] = additive_complement                    

                elif self.mode == PrivateInferenceProtocol.SERVER:

                    additive_nonce = self.recv_npy()

                    # Perform linear layer on additive nonce
                    no_bias_module = copy.deepcopy(layer.pytorch_module)
                    no_bias_module.bias = None
                    
                    #additive_complement = additive_nonce.dot(layer.weight_mat) % self.field
                    #additive_complement = no_bias_module(torch.from_numpy(additive_nonce).double()).numpy().astype(np.int64) % self.field
                    additive_complement = no_bias_module(torch.from_numpy(additive_nonce).double()).numpy().astype(np.int64) % self.field
                    
                    additive_nonce_output = np.random.randint(0, self.field, size=additive_complement.shape, dtype=np.int64) % self.field
                    additive_complement = (additive_complement + additive_nonce_output) % self.field

                    info[i]["additive_nonce_output"] = additive_nonce_output
                    self.send_npy(additive_complement)
                else:
                    raise Exception("Unknown mode")

        input.info = info

        #################################################

        # Initialize relu tables for each element in the network
        print('Initialize RELU tables')
        self.relu_tables = self.initialize_relu_tables(self.mode == PrivateInferenceProtocol.SERVER)
        self.relu_table_size = int(np.sum([sys.getsizeof(x[1]) for x in self.relu_tables.values()]))

        
    def initialize_relu_tables(self, is_server):        

        relu_tables = {}
        for i, layer in enumerate(self.layers):
            if type(layer) == QuantizedReLULayer:
                input_shape = self.misc["shapes"][i]
                relu_bits = layer.bits
                n_relus = np.prod(input_shape)

                # Share secrets (insecure, but for purposes of preprocessing)
                # Important note: since we are encoding everything in self.field bits
                # we choosel local secrets from self.field but they sum to 2**relu_bits
                # numbers
                np.random.seed(0)
                global_secrets = np.random.randint(0, 2**relu_bits, size=n_relus, dtype=np.int64)                
                
                if self.mode == PrivateInferenceProtocol.CLIENT:
                    local_secrets = np.random.randint(0, self.field, size=n_relus, dtype=np.int64)
                    self.send_npy(local_secrets)                    
                    other_secrets = self.recv_npy()
                else:
                    other_secrets = self.recv_npy()
                    local_secrets = (global_secrets - other_secrets) % self.field
                    self.send_npy(local_secrets)
                    
                relu_tables[layer] = []

                t_start = time.time()

                relu_v = np.arange(0, 2**relu_bits)
                relu_v[relu_v >= 2**(relu_bits-1)] = 0
                
                enc = np.arange(0, 2**relu_bits)
                enc = (enc.reshape((1, 2**relu_bits)) - global_secrets.reshape((n_relus, 1))) % 2**relu_bits

                t = relu_v[enc]

                np.random.seed(10)                
                if is_server:
                    t = (np.random.randint(0, self.field, size=t.shape) + t) % self.field
                else:
                    t = -np.random.randint(0, self.field, size=t.shape) % self.field

                # Sanity
                #print(t[3, (global_secrets[3]+42) % 2**relu_bits])
                relu_tables[layer] = (local_secrets, t)
                
                t_end = time.time()
                print("Init tables elapsed: %f s" % (t_end-t_start))
                    
                                    
        return relu_tables
                                        

    def run(self, input, misc={}):
              
        info = input.info

        # At the end, `x` held by both parties should sum to the output
        x = input.x

        outputs = {}
        stack = []
        total_relus = 0
        
        for i, layer in enumerate(self.layers):

            print("Executing layer %s (%d of %d)" % (layer, i, len(self.layers)))
            if i > 0:
                print(x_q.shape)
                x_q %= self.field
                
            t_start = time.time()
            last_layer = i == (len(self.layers)-1)

            if type(layer) == PushInputsToStack:
                stack.append((x_scale, x_q))
                
            elif type(layer) == PopInputsFromStack:
                x_scale, x_q = stack.pop(-1)
                
            elif type(layer) == QuantizedInputLayer:
                x_scale, x_q, _ = quantized_network.quantize(x, bits=layer.bits, qrange=layer.qrange)
                x_q = self.encode(x_q)

                if self.mode == PrivateInferenceProtocol.SERVER:
                    x_scale = self.recv_npy()[0]
                    x_q = self.recv_npy()

                    ###
                    x_q = -self.recv_npy() % self.field
                else:
                    self.send_npy(np.array([x_scale]))
                    self.send_npy(np.zeros_like(x_q))

                    ###
                    nonce = np.random.randint(0, self.field, size=x_q.shape, dtype=np.int64)
                    x_q = (x_q + nonce) % self.field
                    self.send_npy(nonce)                    

            elif type(layer) == QuantizedFCLayer or type(layer) == QuantizedConvLayer:

                
                if self.mode == PrivateInferenceProtocol.CLIENT:

                    additive_nonce = info[i]["additive_nonce"]
                    additive_complement = info[i]["additive_complement"]

                    x_encr = (x_q+additive_nonce) % self.field
                    self.send_npy(x_encr)

                    x_q = -additive_complement
                    x_scale *= layer.scale

                    x_q %= self.field
                    
                    if self.fake_scale:
                        # This uses the rescale from quantized_network
                        # Note the real truncation trick incurs slight
                        # errors different from the truncation errors
                        # in quantized_network as the secret shares are diff
                        # (leading to diff 0/1 errors). This code ensures they
                        # are the same and is used for testing / verification
                        
                        # Note that fake relu does full precision relu
                        other_x_q = self.recv_npy()

                        x_q += other_x_q
                        x_q = self.decode(x_q)

                        x_q, x_scale = rescale_shared(x_q, x_scale, layer.rescale)
                        x_q = self.encode(x_q)

                        nonce = np.random.randint(0, self.field, size=x_q.shape, dtype=np.int64)
                        x_q = (x_q + nonce) % self.field
                        self.send_npy(nonce)
                    else:
                        # Truncation trick                    
                        x_scale *= layer.rescale
                        x_q //= layer.rescale

                        np.random.seed(0)
                        x_q -= np.random.randint(0, 2, size=x_q.shape)

                elif self.mode == PrivateInferenceProtocol.SERVER:
                    
                    additive_nonce_output = info[i]["additive_nonce_output"]

                    x_encr = self.recv_npy() + x_q

                    no_bias_module = copy.deepcopy(layer.pytorch_module)
                    no_bias_module.bias = None

                    #x_q = no_bias_module(torch.from_numpy(x_encr).double()).numpy().astype(np.int64) % self.field
                    x_q = no_bias_module(torch.from_numpy(x_encr).double()).numpy().astype(np.int64) % self.field
                    x_q = (x_q + additive_nonce_output) % self.field
                    x_scale *= layer.scale

                    if layer.unquantized.bias is not None:                        
                        bias = layer.unquantized.bias
                        b_q = quantized_network.quantize_with_scale(bias, x_scale)
                        if type(layer) == QuantizedConvLayer:
                            b_q = b_q.reshape(1, b_q.shape[0], 1, 1)
                        x_q = (x_q + b_q) % self.field                    

                    x_q = x_q.astype(np.int64)                    
                    
                    if self.fake_scale:
                        self.send_npy(x_q)
                        x_q = np.zeros_like(x_q)
                        x_scale *= layer.rescale

                        x_q = self.recv_npy()
                        x_q = -x_q
                    else:
                        # Truncation trick
                        x_scale *= layer.rescale
                        x_q = (self.field - (self.field - x_q)//layer.rescale) % self.field
                    
            if type(layer) == QuantizedReLULayer:
                total_relus += float(np.prod(x_q.shape))
                if self.fake_scale:

                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        other_x_q = self.recv_npy()
                        local_x_q = (other_x_q + x_q) % self.field
                        
                        bits_in = self.misc[layer]["bits_in"]
                        #print(sorted(np.log2(np.abs(self.decode(local_x_q))).flatten().tolist()))
                        print(np.ceil(np.log2(np.max(np.abs(self.decode(local_x_q))))), bits_in)
                        assert(np.ceil(np.log2(np.max(np.abs(self.decode(local_x_q))))) <= bits_in)

                        # Fake relu
                        local_x_q[local_x_q > (self.field-1)//2] = 0                                                
                        
                        bits_in = self.misc[layer]["bits_in"]

                        # Scale down and back up to simulate loss of precision
                        scale_factor = max(1, 2**(bits_in-layer.bits))
                        scale_factor = np.double(scale_factor).astype(local_x_q.dtype)
                        local_x_q //= scale_factor                    
                        
                        # Scale back up
                        local_x_q *= scale_factor
                        
                        nonce = np.random.randint(0, self.field, size=x.shape, dtype=np.int64)
                        self.send_npy(nonce)
                        x = (x - nonce) % self.field

                        x_q = local_x_q
                        
                    else:
                        self.send_npy(x_q)
                        x_q = np.zeros_like(x_q)

                        x = self.recv_npy()

                else:
                    
                    shape = x_q.shape

                    
                    ########################
                    # First truncate x_q
                    ########################
                    bits_in = self.misc[layer]["bits_in"]
                    scale_factor = max(1, 2**(bits_in-layer.bits))
                    #scale_factor = np.int64(scale_factor)
                    
                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        x_q_trunc = x_q // scale_factor
                    else:
                        x_q_trunc = self.field - (self.field-x_q) // scale_factor
                    x_q_trunc = x_q_trunc.flatten()

                    ########################################
                    # ReLU lookup on truncated activations
                    ########################################
                    secret, relu_table = self.relu_tables[layer]
                    x_q_trunc_enc = x_q_trunc % self.field
                    x_q_trunc_enc = (x_q_trunc_enc + secret) % self.field
                        
                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        self.send_npy(x_q_trunc_enc)
                        other_x_q_trunc_enc = self.recv_npy()
                    else:
                        other_x_q_trunc_enc = self.recv_npy()
                        self.send_npy(x_q_trunc_enc)

                    # This contains truncated activation + global secret
                    x_q_trunc_enc_whole = (x_q_trunc_enc + other_x_q_trunc_enc) % self.field #% layer.bits
                    

                    def maxbits(x):
                        return np.log2(np.max(np.abs(x)))
                    
                    # Table lookup on x_q_trunc_enc_whole
                    # At this point x_q_trunc_enc_whole contains
                    # actual activation values (bits_in bis)
                    # + secret (layer.bits)
                    # encoded in self.field bits.
                    # decode it and reencode it in layer.bits bits
                    np.putmask(x_q_trunc_enc_whole, x_q_trunc_enc_whole > (self.field-1)//2, 2**layer.bits-(self.field-x_q_trunc_enc_whole))

                    # Truncation trick sometimes adds +1 noise.
                    # To balance, sometimes subtract 1 noise
                    np.random.seed(0)                    
                    x_q_trunc_enc_whole -= np.random.randint(0, 2, size=x_q_trunc_enc_whole.shape, dtype=np.int64)                    
                        
                    # At this point maxbits(x_q_trunc_enc_whole) should be less than 2**(layer.bits+1)                                        
                    x_q_trunc_enc_whole %= 2**layer.bits

                    # Do table lookup
                    indx = np.arange(0, x_q_trunc_enc_whole.shape[0])
                    relu_res = relu_table[indx,x_q_trunc_enc_whole.astype(np.int32)]

                    assert(relu_res.shape == x_q_trunc_enc_whole.shape)

                    # Scale back up
                    relu_res *= np.double(scale_factor).astype(relu_res.dtype)

                    """
                    #####################
                    # DBG

                    if self.mode ==PrivateInferenceProtocol.CLIENT:
                        self.send_npy(relu_res)                        
                        other_relu_res = self.recv_npy()
                        pass
                    else:
                        other_relu_res = self.recv_npy()
                        self.send_npy(relu_res)                        

                    relu_res_whole = (other_relu_res + relu_res) % (self.field)
                    
                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        other_x_q = self.recv_npy()
                        other_secret = self.recv_npy()
                        ss = (secret + other_secret) % self.field

                        # Result with perfect scaling down
                        baseline =  ((self.decode(other_x_q + x_q)) // scale_factor)
                        np.random.seed(0)
                        baseline -= np.random.randint(0, 2, size=baseline.shape, dtype=np.int64)                                            
                        baseline[baseline < 0] = 0
                        baseline *= np.double(scale_factor).astype(relu_res_whole.dtype)
                        
                        print("SANITY: FINAL RELU diff (should be small < 1):", np.mean(np.abs(baseline.flatten()-relu_res_whole)))

                        # Result with shared truncation
                        simulated = (other_x_q//scale_factor + (self.field-(self.field-x_q)//scale_factor)) % self.field
                        simulated = self.decode(simulated)
                        np.random.seed(0)
                        simulated -= np.random.randint(0, 2, size=simulated.shape, dtype=np.int64)
                        simulated[simulated < 0] = 0
                        simulated *= np.double(scale_factor).astype(relu_res_whole.dtype)

                        print("SANITY: FINAL RELU diff (simulated trunc):", np.mean(np.abs(simulated.flatten()-relu_res_whole)))
                        
                        other_x_q = self.recv_npy()
                        x_q = (simulated - other_x_q) % self.field
                        
                    else:
                        self.send_npy(x_q)
                        self.send_npy(secret)

                        x_q = np.random.randint(0, self.field, size=x_q.shape)
                        self.send_npy(x_q)
                    """
                    
                    #####################
                    #x_q = relu_res.reshape(x_q.shape)

                    #if self.mode == PrivateInferenceProtocol.SERVER:
                    #    x_q = np.zeros_like(x_q)
                    
                    #assert(0)

            if type(layer) == ReLULayer:
                total_relus += float(np.prod(x_q.shape))    
                
                if self.fake_scale:
                
                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        other_x_q = self.recv_npy()
                        x_q = (other_x_q + x_q) % self.field
                        x_q[x_q > (self.field-1)//2] = 0

                        nonce = np.random.randint(0, self.field, size=x.shape, dtype=np.int64)
                        self.send_npy(nonce)
                        x = (x - nonce) % self.field                                        

                    elif self.mode == PrivateInferenceProtocol.SERVER:
                        self.send_npy(x_q)
                        x_q = np.zeros_like(x_q)

                        x = self.recv_npy()

                else:
                    # Please use QuantizedReLULayer for private nonlinear activation function
                    assert(0)
                        
            if type(layer) == FlattenLayer:
                x_q = x_q.reshape((x_q.shape[0], -1))

            if type(layer) == QuantizedAvgPoolLayer:
                x_q = np.round(layer.unquantized.pytorch_module(torch.from_numpy(x_q)).numpy()*layer.unquantized.kernel_size**2) % self.field
                x_scale = x_scale / layer.unquantized.kernel_size**2

                if layer.rescale is not None:
                    if self.mode == PrivateInferenceProtocol.CLIENT:
                        other_x_q = self.recv_npy()
                        x_q += other_x_q
                        x_q_dec = self.decode(x_q)
                        x_q_dec = np.round(x_q_dec/layer.rescale)
                        x_q = self.encode(x_q_dec)
                        x_scale *= layer.rescale

                        nonce = np.random.randint(0, self.field, size=x_q.shape, dtype=np.int64)
                        x_q = (x_q + nonce) % self.field
                        self.send_npy(nonce)
                    else:
                        self.send_npy(x_q)
                        x_q = np.zeros_like(x_q)
                        x_scale *= layer.rescale

                        x_q = self.recv_npy()
                        x_q = -x_q                    

            if type(layer) == QuantizedIdentityLayer:
                indx = layer.unquantized.indx_identity + 1
                other_scale, other_q = outputs[indx]

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
                #
                # Note we reveal scales of the weights (which reveals miniscule info about the actual weights).
                smaller_scale, larger_scale = min(other_scale, x_scale), max(other_scale, x_scale)
                ratio = larger_scale / smaller_scale            

                assert(ratio.is_integer())                
                if smaller_scale == x_scale:
                    x_q = x_q + other_q*ratio
                else:
                    x_q = x_q*ratio + other_q

                x_scale = smaller_scale

            assert(np.linalg.norm(np.round(x_q) - x_q) <= 1e-8)
            
            x_q = x_q % self.field

            t_end = time.time()
            elapsed = t_end-t_start
            misc[str(layer)+"_"+str(i)] = elapsed
            misc["total_relus"] = total_relus
            misc["tabula_relu_table_size"] = self.relu_table_size            
            
            # Track outputs per layer
            outputs[i] = x_scale, x_q

        if self.mode == PrivateInferenceProtocol.SERVER:
            self.send_npy(x_q)
            return None
        else:
            other_x_q = self.recv_npy()
            x_q = self.decode(x_q + other_x_q)
            result = x_scale * x_q
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
    
    net = FCNet(128)
    net.eval()    
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_qrange=8, default_precision=8, default_rescale=2**7)
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 28*28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_quantized_neural_network(layers, x.x) for x in inputs]    

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Tabula(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_scale=True)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Tabula(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_scale=True)

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

    passignment = {
        "pool1": {"rescale": None},
        "pool2": {"rescale": None}
    }

    net = ConvNet()
    net.eval()

    layers = load_quantized_neural_network_from_pytorch(net, layers,
                                                        default_qrange=8, default_precision=8, default_rescale=2**8,
                                                        precision_assignment=passignment)
    
    # Init the inputs
    inputs = [np.random.normal(0, 1, size=(1, 1, 28, 28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    misc = {}
    results = [execute_quantized_neural_network(layers, x.x, misc=misc) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Tabula(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_scale=True)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Tabula(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_scale=True)

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
    layers = [
        ConvLayer("conv1", merge_bn=net.bn1),
        ReLULayer("ReLU"),
        ConvLayer("conv2", merge_bn=net.bn2),
        IdentityLayer("Identity", -1),
        ReLULayer("ReLU")
    ]
    
    layers = load_quantized_neural_network_from_pytorch(net, layers, default_qrange=8, default_precision=8, default_rescale=2**7)
    
    # Init the inputs
    inputs = [np.random.normal(0, .1, size=(32, 32, 28, 28)) for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_quantized_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Tabula(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_scale=True)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Tabula(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_scale=True)

    d.initialize()
    for inp in inputs:
        d.preprocess(inp)

    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed resnet_block_test")

    d.shutdown()                

def resnet_32_pretrained_test():
    reset_seed()    
    mode = int(sys.argv[1])

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
        batch_size=1, shuffle=False,
        pin_memory=True)
    
    # Init the inputs
    inputs = [next(iter(val_loader))[0].numpy() for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    misc = {}
    results = [execute_quantized_neural_network(layers, x.x, misc=misc) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Tabula(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_scale=True)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Tabula(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_scale=True)

    d.initialize()
    if mode == PrivateInferenceProtocol.SERVER:
        print("Communicated bytes", d.get_communicated_bytes())
    
    for inp in inputs:
        d.preprocess(inp)

    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed resnet_32_pretrained_test")

    if mode == PrivateInferenceProtocol.SERVER:
        print("Communicated bytes", d.get_communicated_bytes())

    d.shutdown()                

def resnet_34_pretrained_cifar100_test():
    reset_seed()    
    mode = int(sys.argv[1])

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
        IdentityLayer("shortcut", 22), 
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
        IdentityLayer("shortcut", 50), 
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
        IdentityLayer("shortcut", 90), 
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
    results = [execute_quantized_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # Clear any input data from inputs
        for inp in inputs:
            inp.clear()
        
        # Server        
        d = Tabula(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_scale=True)
        
    else:

        # Clear any weight data from neural network layers
        for layer in layers:
            layer.clear()
        
        # Client        
        d = Tabula(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_scale=True)
    d.initialize()
    
    for inp in inputs:
        d.preprocess(inp)

    preds = [d.run(x) for x in inputs]

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        print(preds[0],results[0])
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed resnet_34_pretrained_cifar100_test")

    if mode == PrivateInferenceProtocol.SERVER:
        print("Communicated bytes", d.get_communicated_bytes())

    d.shutdown()

def test_mnist_act_quantized():
    reset_seed()
    mode = int(sys.argv[1])
    
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

    layers = replace_relu_with_quantized_relu(layers)

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
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1)

    # Init the inputs
    inputs = [next(iter(train_loader))[0].numpy() for i in range(1)]
    inputs = [Input(x) for x in inputs]

    # Compute results for sanity check
    results = [execute_quantized_neural_network(layers, x.x) for x in inputs]

    if mode == PrivateInferenceProtocol.SERVER:

        # For now, do not clear input data as server
        # uses this to calculate range of activation values of a typical input.
        # This usually should be done offline and tracked separately
        
        # Server        
        d = Tabula(mode=PrivateInferenceProtocol.SERVER, neural_network=layers, fake_scale=False)
        
    else:

        # For now, do not clear input data as client
        # uses this to calculate range of activation values of a typical input.
        # This usually should be done offline and tracked separately
        
        # Client        
        d = Tabula(mode=PrivateInferenceProtocol.CLIENT, neural_network=layers, fake_scale=False)
    d.initialize()

    
    for inp in inputs:
        d.preprocess(inp)

    misc = {}
    t_start = time.time()
    preds = [d.run(x, misc=misc) for x in inputs]
    t_end = time.time()

    if mode == PrivateInferenceProtocol.CLIENT:
        print(np.mean(np.abs(preds[0]-results[0])))
        print(preds[0], results[0])
        assert(np.mean(np.abs(preds[0]-results[0])) <= 1e-1)
        print("Passed test_mnist_act_quantized")
    if mode == PrivateInferenceProtocol.SERVER:
        print("Communicated bytes", d.get_communicated_bytes())
        print("Elapsed time", t_end-t_start)
        pprint.pprint(misc)

    d.shutdown()    

def basic_tests():

    fc_relu_test()
    conv_relu_test()
    resnet_block_test()

    # The below tests have exactly 0 err diff between the execution engine with larger field (e.g: 13.8 bit field).
    resnet_34_pretrained_cifar100_test()
    resnet_32_pretrained_test()
    

def relu_qactivation_tests():
    test_mnist_act_quantized()
    
if __name__=="__main__":
    #basic_tests()
    relu_qactivation_tests()
    

import sys
import pathlib
import os
import glob
import numpy as np
import subprocess
import time
import mmap
import distutils.dir_util
import flock
import re
import signal
from private_inference.protocol import PrivateInferenceProtocol
from subprocess import *
import time

file_dir = os.path.dirname(os.path.abspath(__file__))

np.random.seed(0)

class GC_ReLU_opt(object):

    # Different from relu.py in that it starts the program interactively during preprocessing phase
    # then runs the online only phase during online phase.
    #
    # In this implementation, we precompute and save the GCS ahead of time (done in a separate run)
    # so that GCs no longer need to be transferred during online phase, reducing communication.
    # Note that this makes the output of the ReLU incorrect, as some of the GCs depend on the inputs,
    # so this file is primarily used for benchmarking. This gives GCs and advantage.

    def __init__(self, name, shape, protoc, is_server=False,
                 port="1456", server_host="127.0.0.1",
                 print_stats=False,
                 q=32):
        super().__init__()
        self.n_relus = np.prod(shape)
        self.name = name
        self.shape = shape
        self.port = port
        self.server_host = server_host
        self.file_dir = file_dir
        self.protoc = protoc
        self.template_path = "%s/template.mpc" % self.file_dir
        self.is_server = is_server
        self.mode_str = "server" if is_server else "client"
        self.bin_name = "relu-%d,q=%d-relu" % (self.n_relus, q)
        self.inputs_file = "Player-Data/Input-P0-0" if self.is_server else "Player-Data/Input-P1-0"
        self.print_stats = print_stats
        with open(self.template_path, "r") as f:
            self.template = f.read()
        os.chdir(self.file_dir)

        self.p = None
        self.fw = None
        self.fr = None
        self.stats = {}

        # quantization factor
        self.q = q

    def preprocess(self, save_relu_circuits_to=None,
                   compile_gc=True):

        os.chdir(self.file_dir)
        code = self.template.replace("{{N}}", str(int(np.prod(self.shape))))
        code = code.replace("{{Q}}", str(int(np.prod(self.q))))        

        code_name = self.bin_name

        # Write layer code
        with open(code_name, "w") as f:
            f.write(code)

        def already_compiled():
            progs = glob.glob("%s/Programs/Schedules/*" % file_dir)
            for n in progs:
                if code_name in n:
                    return True
            return False

        compiled_already = already_compiled()
        if compiled_already:
            print("Skipping compilation, already found: %s" % code_name)

        # Compile code
        if compile_gc and not compiled_already:
            if self.mode_str == "server":
                t_start = time.time()
                #os.system("./compile.py --output %s -B 32 %s > /dev/null 2>&1" % (self.mode_str, self.name))
                os.system("./compile.py --budget 30000 -B 32 %s" % (code_name))
                t_elaps = time.time()-t_start
                if self.print_stats:
                    print("Compilation elapsed: %s" % str(t_elaps))

                if save_relu_circuits_to is not None:
                    dest = str(save_relu_circuits_to)

                    save_relu_circuits_to = "%s/%s" % (dest, self.mode_str) 
                    os.makedirs(save_relu_circuits_to, exist_ok=True)
                    distutils.dir_util.copy_tree("%s/Programs/" % self.file_dir,
                                                 save_relu_circuits_to, preserve_mode=0
                    )

                    save_relu_circuits_to = "%s/%s" % (dest, "client")
                    os.makedirs(save_relu_circuits_to, exist_ok=True)
                    distutils.dir_util.copy_tree("%s/Programs/" % self.file_dir,
                                                 save_relu_circuits_to, preserve_mode=0
                    ) 
            print("Waiting barrier %s" % self.mode_str)
            self.protoc.barrier()
            print("Past barrier %s" % self.mode_str)

        # Start running interactively
        self.fw = open("/tmp/tmpout_%s_%s" % (self.name, self.mode_str), "wb")
        self.fr = open("/tmp/tmpout_%s_%s" % (self.name, self.mode_str), "r")
        if self.is_server:
            cmd = "./yao-party-interactive.x -O -p 0  -pn %s -gcs Programs/Gcs/gc_%s %s" % (self.port, self.name, self.bin_name)
            print("Exec cmd: %s" % cmd)            
            self.p = Popen(cmd, stdin=PIPE, stdout=self.fw, stderr=self.fw, bufsize=1, shell=True)
        else:
            cmd = "./yao-party-interactive.x -O -p 1  -pn %s -h %s -gcs Programs/Gcs/gc_%s %s" % (self.port, self.server_host, self.name, self.bin_name)
            print("Exec cmd: %s" % cmd)            
            self.p = Popen(cmd, stdin=PIPE, stdout=self.fw, stderr=self.fw, bufsize=1, shell=True)

        # Wait for "start running" message
        out = ''
        while True:
            out += self.fr.read()
            time.sleep(.3)
            if "Start running" in out:
                break
            print(".", end="")
            sys.stdout.flush()
        time.sleep(1)
        self.p.send_signal(signal.SIGSTOP)
        time.sleep(1)
        print("Done loading %s, awaiting input" % self.name)

    def run(self, x):
        t_start = time.time()
        
        os.chdir(self.file_dir)
        assert(x.shape == self.shape)

        # Send prog inputs
        self.p.send_signal(signal.SIGCONT)        
        #self.p.stdin.write(str.encode(" ".join([str(int(z)) for z in list(x.flatten().astype(np.int32))])+"\n"))
        # Use as placeholder for input
        self.p.stdin.write(str.encode(" ".join([str(int(z)) for z in list(np.random.randint(0, 100, size=(self.n_relus)))])+"\n"))
        _, _ = self.p.communicate()
        result = self.fr.read()
        self.p.wait()

        print("ReLU Optimized Elapsed: ", time.time()-t_start, self.mode_str)
        
        if self.is_server:
            # Extract some stats from output
            try:
                elapsed = float(re.findall("Time = ([0-9\.]+)", str(result))[0])
                data_sent = float(re.findall("Data sent = ([0-9\.]+)", str(result))[0])*1024*1024
                self.stats["elapsed"] = elapsed
                self.stats["data_sent"] = data_sent
            except:
                pass

            self.protoc.barrier()            
            return x
        else:

            try:
                # This may fail if GC files are not appropriately generated (on first calls).
                # Make sure to run the network locally to generate the gc files first.
                
                # Extract some stats from output
                elapsed = float(re.findall("Time = ([0-9\.]+)", str(result))[0])
                data_sent = float(re.findall("Data sent = ([0-9\.]+)", str(result))[0])*1024*1024
                self.stats["elapsed"] = elapsed
                self.stats["data_sent"] = data_sent

                if self.print_stats:
                    print("\n".join([x for x in result.splitlines()[1:]]))
                    t_elaps = time.time()-t_start
                    print("Elapsed: %s" % str(t_elaps))
                result = np.array(eval(result.splitlines()[-4]))
                result = result.reshape(self.shape)
            except:
                result = np.zeros(shape=self.shape)

            self.protoc.barrier()                
            return result

def test_gc():
    np.random.seed(0)
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    if mode == 0:        
        #x = np.array([[0, 1, -1, 5, 21]])
        #x = np.random.normal(0, 1, size=(10000)).astype(np.int)
        #x = np.random.normal(0, 1, size=(8192)).astype(np.int)
        x = np.random.normal(0, 1, size=(1234)).astype(np.int)
    else:
        #x = np.array([[0, -30, 3, -20, 21]])
        #x = np.random.normal(0, 1, size=(10000)).astype(np.int)
        #x = np.random.normal(0, 1, size=(8192)).astype(np.int)
        x = np.random.normal(0, 1, size=(1234)).astype(np.int)        
    
    g = GC_ReLU_opt("testing", x.shape, protoc, is_server=mode, print_stats=True)
    g.preprocess()
    g.protoc.barrier()
    result = g.run(x)

    print("Done")
    
def test_gc_quantized():
    np.random.seed(0)
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    if mode == 0:        
        #x = np.array([[0, 1, -1, 5, 21]])
        #x = np.random.normal(0, 1, size=(10000)).astype(np.int)
        #x = np.random.normal(0, 1, size=(8192)).astype(np.int)
        x = np.random.normal(0, 1, size=(1234)).astype(np.int)
    else:
        #x = np.array([[0, -30, 3, -20, 21]])
        #x = np.random.normal(0, 1, size=(10000)).astype(np.int)
        #x = np.random.normal(0, 1, size=(8192)).astype(np.int)
        x = np.random.normal(0, 1, size=(1234)).astype(np.int)        
    
    g = GC_ReLU_opt("testing", x.shape, protoc, is_server=mode, print_stats=True, q=32)
    g.preprocess()
    g.protoc.barrier()
    result = g.run(x)

    #

    g = GC_ReLU_opt("testing", x.shape, protoc, is_server=mode, print_stats=True, q=8)
    g.preprocess()
    g.protoc.barrier()
    result = g.run(x)

    
    print("Done")
    
def test_gc_multi():
    np.random.seed(0)
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    if mode == 0:        
        #x = np.array([[0, 1, -1, 5, 21]])
        #x = np.random.normal(0, 1, size=(21632)).astype(np.int)
        x = np.random.normal(0, 1, size=(8192)).astype(np.int)
    else:
        #x = np.array([[0, -30, 3, -20, 21]])
        #x = np.random.normal(0, 1, size=(21632)).astype(np.int)
        x = np.random.normal(0, 1, size=(8192)).astype(np.int)        

    gcs = [GC_ReLU_opt("t%d" % i, x.shape, protoc, is_server=mode, print_stats=False, port=str(1456+30*i)) for i in range(50)]
    for g in gcs:
        g.preprocess()

    for g in gcs:
        g.protoc.barrier()
        g.run(x)        

def test_compile():
    np.random.seed(0)
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    x = np.random.normal(0, 1, size=(5)).astype(np.int)
    x2 = np.random.normal(0, 1, size=(10)).astype(np.int)
    
    g = GC_ReLU_opt("testing", x.shape, protoc, is_server=mode, print_stats=True, port=str(1456+3*0))
    g.preprocess()

    g2 = GC_ReLU_opt("testing2", x.shape, protoc, is_server=mode, print_stats=True, port=str(1456+3*1))
    g2.preprocess()

    g3 = GC_ReLU_opt("testing3", x.shape, protoc, is_server=mode, print_stats=True, port=str(1456+3*2))
    g3.preprocess()

    g4 = GC_ReLU_opt("testing4", x2.shape, protoc, is_server=mode, print_stats=True, port=str(1456+3*3))
    g4.preprocess()

    g5 = GC_ReLU_opt("testing5", x2.shape, protoc, is_server=mode, print_stats=True, port=str(1456+3*4))
    g5.preprocess()

    g.run(x)
    g2.run(x)
    g3.run(x)

    g4.run(x2)
    g5.run(x2)        

if __name__=="__main__":
    #test_gc_multi()
    #test_gc()
    test_gc_quantized()
    #test_compile()
    


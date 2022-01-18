import sys
import glob
import pathlib
import os
import numpy as np
import subprocess
import time
import mmap
import distutils.dir_util
import flock
import re
from private_inference.protocol import PrivateInferenceProtocol

file_dir = os.path.dirname(os.path.abspath(__file__))

np.random.seed(0)

class GC_ReLU(object):

    def __init__(self, name, shape, protoc, is_server=False,
                 port="1456", server_host="127.0.0.1",
                 print_stats=False):
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
        self.bin_name = "relu-%d-relu" % (self.n_relus)
        self.inputs_file = "Player-Data/Input-P0-0" if self.is_server else "Player-Data/Input-P1-0"
        self.print_stats = print_stats
        with open(self.template_path, "r") as f:
            self.template = f.read()
        os.chdir(self.file_dir)

        self.stats = {}


    def preprocess(self, save_relu_circuits_to=None,
                   compile_gc=True):
        
        os.chdir(self.file_dir)
        code = self.template.replace("{{N}}", str(int(np.prod(self.shape))))

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
                os.system("./compile.py --budget=1000000000 -B 32 %s" % (code_name))
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

    def run(self, x):
        t_start = time.time()
        
        os.chdir(self.file_dir)
        assert(x.shape == self.shape)

        
        # Write to file. Not a runtime bottleneck
        with open(self.inputs_file, "w") as f:
            f.write(" ".join([str(z) for z in list(x.flatten())]))
        write_elaps = time.time()-t_start            
        
        if self.is_server:            
            #os.system("./yao-party.x -p 0 -pn %s %s -O > /dev/null 2>&1" % (self.port, self.bin_name))
            p = subprocess.Popen("./yao-party.x -p 0 -pn %s %s -O" % (self.port, self.bin_name),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True)
            result, _ = p.communicate()

            # Extract some stats from output                                                                                                                 
            elapsed = float(re.findall("Time = ([0-9\.]+)", str(result))[0])
            data_sent = float(re.findall("Data sent = ([0-9\.]+)", str(result))[0])*1024*1024
            self.stats["elapsed"] = elapsed
            self.stats["data_sent"] = data_sent            
            
            return x
        else:
            #os.system("./yao-party.x -p 1 -pn %s %s > /dev/null 2>&1" % (self.port, self.bin_name))            
            p = subprocess.Popen("./yao-party.x -p 1 -pn %s -h %s %s -O" % (self.port, self.server_host,
                                                                            self.bin_name),
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True)
            result, _ = p.communicate()

            # Extract some stats from output
            elapsed = float(re.findall("Time = ([0-9\.]+)", str(result))[0])
            data_sent = float(re.findall("Data sent = ([0-9\.]+)", str(result))[0])*1024*1024
            self.stats["elapsed"] = elapsed
            self.stats["data_sent"] = data_sent
            
            if self.print_stats:
                print("\n".join([x.decode("utf-8") for x in result.splitlines()[1:]]))
                t_elaps = time.time()-t_start
                print("Elapsed: %s" % str(t_elaps))
                print("Write time: %s" % str(write_elaps))
            result = np.array(eval(result.splitlines()[0]))
            result = result.reshape(self.shape)
            
            return result

def test_gc():
    np.random.seed(0)
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    if mode == 0:        
        x = np.array([[0, 1, -1, 5, 21]])
        x = np.random.normal(0, 1, size=(163)).astype(np.int)
    else:
        x = np.array([[0, -30, 3, -20, 21]])
        x = np.random.normal(0, 1, size=(163)).astype(np.int)
    golden = x*2
    golden[golden < 0] = 0
    
    g = GC_ReLU("testing", x.shape, protoc, is_server=mode, print_stats=True)
    g.preprocess()
    result = g.run(x)

    if not mode:
        # client
        print(golden)
        print(result)
        assert(np.linalg.norm(golden-result) <= 1e-8)
        print("Passed")

def test_compile():
    np.random.seed(0)
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    x = np.random.normal(0, 1, size=(5)).astype(np.int)
    x2 = np.random.normal(0, 1, size=(10)).astype(np.int)
    
    g = GC_ReLU("testing", x.shape, protoc, is_server=mode, print_stats=True)
    g.preprocess()

    g2 = GC_ReLU("testing2", x.shape, protoc, is_server=mode, print_stats=True)
    g2.preprocess()

    g3 = GC_ReLU("testing3", x.shape, protoc, is_server=mode, print_stats=True)
    g3.preprocess()

    g4 = GC_ReLU("testing4", x2.shape, protoc, is_server=mode, print_stats=True)
    g4.preprocess()

    g5 = GC_ReLU("testing5", x2.shape, protoc, is_server=mode, print_stats=True)
    g5.preprocess()

def test_gc_N():
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    if mode == 0:
        x = np.random.randint(-20, 20, size=(100, ))
    else:
        x = np.random.randint(-20, 20, size=(100, ))
    
    g = GC_ReLU("testing", x.shape, protoc, is_server=mode, print_stats=True)
    g.preprocess()
    t_start = time.time()
    result = g.run(x)
    t_end = time.time()

    print("Time", t_end-t_start)

    if not mode:
        # client
        print("Done")

def test_gc_N_large():
    mode = int(sys.argv[1])
    protoc = PrivateInferenceProtocol("testing", mode=mode)

    if mode == 0:
        x = np.random.randint(-20, 20, size=(65536, ))
    else:
        x = np.random.randint(-20, 20, size=(65536, ))
    
    g = GC_ReLU("testing", x.shape, protoc, is_server=mode, print_stats=True)
    g.preprocess()
    result = g.run(x)

    if not mode:
        # client
        print("Done")        
        

if __name__=="__main__":
    #test_compile()
    #test_gc()
    test_gc_N()
    #test_gc_N_large()
    


import sys
import zmq
import numpy as np
import zlib
import scapy
import threading
import subprocess
import select
import signal
import time
import json
from multiprocessing import Process, Manager, Value
import subprocess

input_id = 0

GC_PORT="1457" # Not sure why this is 1 more than the port specified

def sniff(terminate, ports, metrics):
    print("Start Sniffing...")
    ps = []
    for port in ports:
        ps.append(subprocess.Popen(['tshark  -z conv,ip -p -f "tcp port %s" -i any -q' % port],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=True))
    while not terminate.value:
        time.sleep(1)

    for p,metric,port in zip(ps,metrics,ports):
        print("")        
        p.send_signal(signal.SIGINT)
        print("")
        print("Sleeping 60 seconds to gather stat output from tshark...")
        time.sleep(30)
        p.wait()
        print("")
        out = p.stdout.read()
        print(out)
        try:
            stats = out.splitlines()[-2].split()
            total_bytes = int(stats[-3])
            print(port, total_bytes)
            metric.value = total_bytes
        except:
            pass
    print("Finished Sniffing...")    

class Input(object):

    def __init__(self, x):
        global input_id
        self.id = input_id
        input_id += 1
        self.x = x
        self.info = {}

    def clear(self):
        self.x = np.zeros_like(self.x)
        
class PrivateInferenceProtocol(object):
    CLIENT=0
    SERVER=1
    
    def __init__(self, name, mode=None, neural_network=None,
                 host="127.0.0.1", port="1234"):
        self.name = name
        assert(mode in [PrivateInferenceProtocol.CLIENT,
                        PrivateInferenceProtocol.SERVER])
        self.mode = mode
        self.host = host
        self.port = port
        self.layers = neural_network
        self.bytes_sent = 0

        # Bind tcp network
        self.context = zmq.Context()
        if self.mode == PrivateInferenceProtocol.CLIENT:
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.connect("tcp://%s:%s" % (self.host, self.port))
        else:            
            self.socket = self.context.socket(zmq.PAIR)
            #self.socket.bind("tcp://%s:%s" % (self.host, self.port))
            self.socket.bind("tcp://*:%s" % (self.port))

        # Handshake
        if self.mode == PrivateInferenceProtocol.CLIENT:
            secret = np.array([[42]])
            self.send_npy(secret)
            response_secret = self.recv_npy()
            assert(np.linalg.norm(response_secret - np.array([[84]])) <= 1e-8)
            print("Client successfully connected")
        if self.mode == PrivateInferenceProtocol.SERVER:
            message = self.recv_npy()
            assert(np.linalg.norm(message - np.array([[42]])) <= 1e-8)
            self.send_npy(np.array([[84]]))
            print("Server successfully connected")

        # Sniff
        self.sniffer = None
        self.sniffer_manager = Manager()
        self.terminate_sniffer = self.sniffer_manager.Value('i', 0)

        self.gc_bytes = self.sniffer_manager.Value('i', 0)
        self.online_bytes = self.sniffer_manager.Value('i', 0)

        self.ports = [GC_PORT, self.port]
        self.metrics = [self.gc_bytes, self.online_bytes]

    def initialize(self):
        # One time initialization ever
        raise Exception("Not implemented")

    def preprocess(self, input):
        # Done multiple times per input
        raise Exception("Not implemented")

    def run(self, input):
        # Online preprocessing step
        raise Exception("Not implemented")

    #### Sniffing
                
    def get_communicated_bytes(self):
        return self.online_bytes.value

    def get_communicated_bytes_gc(self):
        return self.gc_bytes.value
        
    def start_sniffing(self):
        if self.mode == PrivateInferenceProtocol.SERVER:
            self.sniffer = Process(target=sniff, args=(self.terminate_sniffer, self.ports, self.metrics))
            self.sniffer.start()
        time.sleep(10)

    def stop_sniffing(self):
        if self.mode == PrivateInferenceProtocol.SERVER:
            self.terminate_sniffer.value = 1
            self.sniffer.join()            

    #### End Sniffing            

    def shutdown(self):
        self.barrier()
        if self.mode == PrivateInferenceProtocol.CLIENT:
            self.socket.disconnect("tcp://%s:%s" % (self.host, self.port))
        else:
            print(self.socket.LAST_ENDPOINT)
            self.socket.unbind(self.socket.LAST_ENDPOINT)
            
        self.socket.close()
        self.context.term()
        time.sleep(1)
    
    ### Helpers

    def send_npy(self, A, flags=0, copy=True, track=False, use_int16=False):
        prevtype = A.dtype
        
        if use_int16:
            A = A.astype(np.uint16)

        self.bytes_sent += sys.getsizeof(A)
        print("SENDING BYTES:", sys.getsizeof(A), self.bytes_sent)        
            
        socket = self.socket
        md = dict(
            dtype = str(A.dtype),
            shape = A.shape,
            prevtype = str(prevtype),
        )
        socket.send_json(md, flags|zmq.SNDMORE)
        return socket.send(A, flags, copy=copy, track=track)

    def recv_npy(self, flags=0, copy=True, track=False):
        socket = self.socket
        md = socket.recv_json(flags=flags)
        msg = socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        A = A.reshape(md['shape']).astype(md['prevtype'])
        return A

    def barrier(self):
        if self.mode == PrivateInferenceProtocol.CLIENT:
            self.send_npy(np.array([0]))
            self.recv_npy()
        else:
            self.recv_npy()
            self.send_npy(np.array([0]))

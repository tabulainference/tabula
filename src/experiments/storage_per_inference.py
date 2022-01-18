import sys
import re
import numpy as np
from private_inference.protocol import *
from private_inference.delphi import Delphi
from private_inference.tabula import Tabula
from experiments.benchmark import *
import matplotlib.pyplot as plt
import matplotlib.style as style
#style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

def plot_bits_vs_storage():
    bits = [4,5,6,7,8,9,10,11,12]
    storage = [(2**x*8)/1024 for x in bits]
    storage_gc = [17*x/32 for x in bits]
    plt.plot(bits, storage, label="Tabula", linewidth=4, marker="o", markersize=9)
    plt.xlabel("Precision for Activation", fontsize=32)
    plt.ylabel("Storage/Mem. (KB)", fontsize=32)
    plt.plot(bits, storage_gc, label="Garbled Circuits", linewidth=4, marker="o", markersize=9)
    plt.xticks(fontsize=22)
    plt.yscale("log", base=2)
    plt.yticks([.125, .25, .5, 1,2,4,8,16], fontsize=22)    
    plt.legend(loc="best", fontsize=20)
    plt.tight_layout()
    plt.savefig("bits_vs_storage.pdf", bbox_inches="tight")

def plot_comm_vs_accuracy(fpath, name):
    plt.cla()
    plt.clf()

    tabula_comm = {
        "cifar10": 14,
        "cifar100": 59.5,
        "lenet": 3.5,
        "cifar100_vgg": 12.1
    }[name]

    gc_comm_32 = {
        "cifar10": 311,
        "cifar100": 1.5*1024,
        "lenet": 124,
        "cifar100_vgg": 286,
    }[name]

    def readfile(fpath):
        d = {}
        with open(fpath, "r") as f:
            for line in f:
                if "acc" in line and "relu bits:" in line:
                    matches = re.findall("relu bits: ([a-zA-Z0-9\.]+), acc: ([a-zA-Z0-9\.]+)", line)
                    bits, acc = matches[0]
                    bits, acc = eval(bits), eval(acc)
                    d[bits] = acc
        return d
    d = readfile(fpath)
    
    baseline_acc = d[None]
    
    bits = sorted([x for x in d.keys() if x is not None])
    accs = [d[x] for x in bits]
    comm_mb = [tabula_comm for x in bits]

    acc_spread = max(accs)-min(accs)
    text_offset = acc_spread / 20    

    plt.axhline(y=baseline_acc, color='b', linestyle='--', label="Baseline Acc", linewidth=4)

    accs = [d[x] for x in bits]
    comm_gcs = [gc_comm_32*(x/32) for x in bits]

    comm_red = [a/b for a,b in zip(comm_gcs, comm_mb)]
    
    plt.plot(comm_red, accs, linewidth=4, marker="o", markersize=8)

    for i, bit in enumerate(bits):
        plt.annotate("$A_{%d}$" % bit, (comm_red[i], accs[i]), fontsize=24,
                     xytext=(comm_red[i], accs[i]+text_offset))
    
    #plt.xscale("log", base=2)
    plt.xlabel("Communication Reduction", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="lower right", fontsize=20)
    plt.tight_layout()        
    plt.savefig("%s_comm_vs_acc.pdf" % name, bbox_inches="tight")
    
        
def plot_storage_vs_accuracy(fpath, name):    
    
    plt.cla()
    plt.clf()

    n_relus = {
        "cifar10": 303104.0,
        "cifar100": 1474560.0,
        "lenet": 58624.0,
        "cifar100_vgg": 276480.0,
    }[name]
    
    def readfile(fpath):
        d = {}
        with open(fpath, "r") as f:
            for line in f:
                if "acc" in line and "relu bits:" in line:
                    matches = re.findall("relu bits: ([a-zA-Z0-9\.]+), acc: ([a-zA-Z0-9\.]+)", line)
                    bits, acc = matches[0]
                    bits, acc = eval(bits), eval(acc)
                    d[bits] = acc
        return d
    d = readfile(fpath)
    
    baseline_acc = d[None]
    
    bits = sorted([x for x in d.keys() if x is not None])
    accs = [d[x] for x in bits]
    storage_kb = [2**x*8/1024 * n_relus for x in bits]

    acc_spread = max(accs)-min(accs)
    text_offset = acc_spread / 20

    plt.axhline(y=baseline_acc, color='b', linestyle='--', label="Baseline Acc", linewidth=4)

    for i, bit in enumerate(bits):
        plt.annotate("$A_{%d}$" % bit, (storage_kb[i], accs[i]), fontsize=24, xytext=(storage_kb[i], accs[i]+text_offset))

    plt.plot(storage_kb, accs, linewidth=4, marker="o", markersize=8, label="Tabula")


    bits = [4, 6, 8, 12]
    accs = [d[x] for x in bits]
    storage_gcs = [17*n_relus*(x/32) for x in bits]
    plt.plot(storage_gcs, accs, linewidth=4, marker="o", markersize=8, label="Garbled Circuits")

    for i, bit in enumerate(bits):
        plt.annotate("$A_{%d}$" % bit, (storage_gcs[i], accs[i]), fontsize=24, xytext=(storage_gcs[i], accs[i]-text_offset))    
    
    plt.xscale("log", base=2)
    plt.xlabel("Storage (KB)", fontsize=32)
    plt.ylabel("Accuracy", fontsize=32)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="lower right", fontsize=20)
    plt.tight_layout()        
    plt.savefig("%s_storage_vs_acc.pdf" % name, bbox_inches="tight")

def plot_runtime_vs_accuracy(name):    
    
    plt.cla()
    plt.clf()
    
    def readfile(fpath):
        d = {}
        with open(fpath, "r") as f:
            for line in f:
                if "acc" in line and "relu bits:" in line:
                    matches = re.findall("relu bits: ([a-zA-Z0-9\.]+), acc: ([a-zA-Z0-9\.]+)", line)
                    bits, acc = matches[0]
                    bits, acc = eval(bits), eval(acc)
                    d[bits] = acc
        return d

    def readfile2(fpath):
        with open(fpath, "r") as f:
            return eval(f.read())

    bits = [32,16,8]
    
    gc_fpaths = [
        "logfiles/runtimes/new/%s_gc_b=%d_benchmark_out_client" % (name, b)
        for b in bits
    ]
    gc_runtimes = [
        readfile2(x)["total_elapsed_time"] for x in gc_fpaths
    ]

    tabula_runtime = (
        readfile2("logfiles/runtimes/new/%s_tabula_b=0_benchmark_out_client" % name)["total_elapsed_time"])

    gc_accs_orig = readfile("logfiles/table_size_vs_acc/%s_bits_vs_acc_final" % name)
    gc_accs = [gc_accs_orig[None] if b == 32 else gc_accs_orig[b] for b in bits]

    acc_spread = max(gc_accs) - min(gc_accs)
    text_offset = acc_spread / 20

    #speedups = [x/tabula_runtime for x in gc_runtimes]
    #print(speedups)
    
    plt.plot(gc_runtimes, gc_accs, linewidth=4, marker="o", markersize=8, label="Garbled Circuits")
    for i, bit in enumerate(bits):
        plt.annotate("$A_{%d}$" % bit, (gc_runtimes[i], gc_accs[i]), fontsize=24, xytext=(gc_runtimes[i], gc_accs[i]-text_offset))

    t_bits = [12]
    t_accs = [gc_accs_orig[b] for b in t_bits]
    plt.plot([tabula_runtime]*len(t_accs), t_accs, linewidth=4, marker="*", markersize=16, label="Tabula")
    for i, bit in enumerate(t_bits):
        plt.annotate("$A_{%d}$" % bit, (tabula_runtime, t_accs[i]), fontsize=24, xytext=(tabula_runtime, t_accs[i]-text_offset))    

    plt.xscale("log", base=10)
    plt.xlabel("Runtime (s)", fontsize=32)
    plt.ylabel("Accuracy", fontsize=32)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(loc="best", fontsize=15)
    plt.tight_layout()        
    plt.savefig("%s_speedup_vs_acc.pdf" % name, bbox_inches="tight")


if __name__=="__main__":

    """
    plot_bits_vs_storage()
    
    plot_storage_vs_accuracy("logfiles/table_size_vs_acc/mnist_bits_vs_acc_final", "lenet")        
    plot_storage_vs_accuracy("logfiles/table_size_vs_acc/cifar10_bits_vs_acc_final", "cifar10")
    plot_storage_vs_accuracy("logfiles/table_size_vs_acc/cifar100_bits_vs_acc_final", "cifar100")
    plot_storage_vs_accuracy("logfiles/table_size_vs_acc/cifar100vgg_bits_vs_acc_final", "cifar100_vgg")

    plot_comm_vs_accuracy("logfiles/table_size_vs_acc/mnist_bits_vs_acc_final", "lenet")
    plot_comm_vs_accuracy("logfiles/table_size_vs_acc/cifar10_bits_vs_acc_final", "cifar10")
    plot_comm_vs_accuracy("logfiles/table_size_vs_acc/cifar100_bits_vs_acc_final", "cifar100")
    plot_comm_vs_accuracy("logfiles/table_size_vs_acc/cifar100vgg_bits_vs_acc_final", "cifar100_vgg")
    """

    plot_runtime_vs_accuracy("mnist")
    plot_runtime_vs_accuracy("cifar10")
    plot_runtime_vs_accuracy("cifar100")
    plot_runtime_vs_accuracy("cifar100_vgg")            

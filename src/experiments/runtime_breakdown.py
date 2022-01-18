import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
#style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

def read_data(f):
    with open(f) as ff:
        return eval(ff.read())

def breakdown(d):
    relu_time = np.sum([v for k,v in d.items() if "ReLU" in k and "n_elements" not in k and "data_sent" not in k])
    linear_time = np.sum([v for k,v in d.items() if "conv" in k or "fc" in k or "features" in k])
    return linear_time*1000, relu_time*1000

def plot_compare(a, b, name):
    plt.cla()
    plt.clf()    
    
    a = read_data(a)
    b = read_data(b)

    labels = ["Garbled Circuits", "Tabula"]
    
    linear_time, nonlinear_time = breakdown(a)
    linear_time_2, nonlinear_time_2 = breakdown(b)

    x = np.arange(len(labels))  
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [linear_time_2, linear_time], width, label='Linear Ops')
    rects2 = ax.bar(x + width/2, [nonlinear_time_2, nonlinear_time], width, label='ReLU')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Runtime (ms)', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=18)
    ax.set_yscale('log')
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)    

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()


    print(linear_time, nonlinear_time)
    print(linear_time_2, nonlinear_time_2)

    fig.savefig("%s.pdf" % name)

if __name__=="__main__":


    plot_compare("logfiles/runtimes/mnist_tabula",
                 "logfiles/runtimes/mnist_delphi_benchmark_out_client",
                 "mnist_runtime_breakdown")
    plot_compare("logfiles/runtimes/cifar100_tabula",
                 "logfiles/runtimes/cifar100_delphi_benchmark_out_client",
                 "cifar100_runtime_breakdown")

    plot_compare("logfiles/runtimes/cifar10_tabula",
                 "logfiles/runtimes/cifar10_delphi_benchmark_out_client",
                 "cifar10_runtime_breakdown")

    plot_compare("logfiles/runtimes/cifar100_vgg_tabula",
                 "logfiles/runtimes/cifar100_vgg_delphi_benchmark_out_client",
                 "cifar100_vgg_runtime_breakdown")

    pass

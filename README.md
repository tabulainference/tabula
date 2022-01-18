# Tabula: Efficiently Computing Nonlinear Activation Functions for Private Neural Network Inference

## Requirements

Install conda and create environment from the environment file:

```
conda env create -f environment.yml
```

```
conda activate tabula
```

Also install tshark.

## Install

Install garbled circuits code, unpack precompiled garbled circuits, and add python code to python path:

```
bash install.sh
```

```
bash install_gc.sh
```

```
source setup.sh
```

## Local benchmarks

Run tabula vs gc locally (note, when measuring communication with tshark, prepend sudo):

```
bash scripts/run_benchmark_local.sh {tabula|gc} {mnist|cifar10|cifar100|cifar100_vgg}
```

Example output for Tabula:
```
{
 'ReLU_2': 0.0853738784790039,
 'ReLU_4': 0.13007211685180664,
 'ReLU_8': 0.008369207382202148,
 'conv1 {kernel_size=(3, 3), stride=1, padding=0, merge_bn=0} loaded=1 (quantized bits=32 rescale=65536.000000 qrange=30.000000)_1': 0.0008199214935302734,
 'conv2 {kernel_size=(3, 3), stride=1, padding=0, merge_bn=0} loaded=1 (quantized bits=32 rescale=65536.000000 qrange=30.000000)_3': 0.001979827880859375,
 'fc1 {shape: (9216, 128)} loaded=1 (quantized bits=32 rescale=65536.000000 qrange=30.000000)_7': 0.0005328655242919922,
 'fc2 {shape: (128, 10)} loaded=1 (quantized bits=32 rescale=65536.000000 qrange=30.000000)_9': 0.0004868507385253906,
 'flatten_6': 0.00018715858459472656,
 'mode': 'client',
 'pool2d_quantized_5': 0.002482175827026367,
 'quantized_input_layer (quantized bits=32 qrange=30.000000)_0': 0.0051729679107666016,
 'total_bytes_communicated_gc_online': 0,
 'total_bytes_communicated_online': 695322,
 'total_elapsed_time': 0.23865413665771484,
 'total_relus': 58624.0
}
```
method=${1:-delphi} # gc tabula
task=${2:-mnist} # mnist cifar10 cifar100
bits=${3:-32}

pkill -f tshark
pkill -f yao
pkill -f python
python src/experiments/benchmark.py --mode 0 --task $task --method $method --bits $bits &
python src/experiments/benchmark.py --mode 1 --task $task --method $method --bits $bits 

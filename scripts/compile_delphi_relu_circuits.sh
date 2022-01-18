rm -rf src/garbled_circuits/Programs/*/*b

task=${1:-mnist} # gc tabula
bits=${2:-32} # mnist cifar10 cifar100

pkill -f python
python src/experiments/delphi/delphi_compile_relu_circuits.py 0 $task $bits &
BACK_PID=$!
python src/experiments/delphi/delphi_compile_relu_circuits.py 1 $task $bits &
BACK_PID_2=$!
wait $BACK_PID
wait $BACK_PID_2

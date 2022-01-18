pkill -f python
pkill -f yao

python src/garbled_circuits/relu_optimized.py 0 &
BACK_PID=$!
python src/garbled_circuits/relu_optimized.py 1 &
BACK_PID_2=$!

wait $BACK_PID
wait $BACK_PID_2

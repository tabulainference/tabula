pkill -f python
python src/private_inference/delphi.py 0 &
BACK_PID=$!
python src/private_inference/delphi.py 1 &
BACK_PID_2=$!
wait $BACK_PID
wait $BACK_PID_2

for delay in 0 2 4 6 8 10
do
  nohup python3 train.py --algorithm IS --stochastic False --verbose True --delay $delay &
  nohup python3 train.py --algorithm normal --stochastic False --verbose True --delay $delay &
  nohup python3 train.py --algorithm delay --stochastic False --verbose True --delay $delay
  wait
done
Recurrent neural networks with differentiable structure. The number of neurons
in the network undergoes gradient descent, just like the weights of the
network. Source code for an upcoming paper (more information soon).

This code is based on Andrej Karpathy's [`min-char-rnn.py`](https://gist.github.com/karpathy/d4dee566867f8291f086) program.

`rnn.py` is the main program. You can run it "as is" (`python rnn.py`) to run
the model on the "hard" problem for 300000 cycles.  It will generate an output
file called `output.txt`, updated every 1000 cycles, which logs the current
cycle number, position in the input file, loss, number of neurons, and total absolute sum of multipliers. (see code).

Other
python files in the repository generate inputs or figures, or submit jobs to a cluster.




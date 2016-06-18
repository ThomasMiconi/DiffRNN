Recurrent neural networks with differentiable structure. The number of neurons in the network undergoes and gradient descent, just like the weights of the network. Source code for an upcoming paper (more information soon).

`rnn.py` is the main program. You can simply run it as is. It will generate an
output file called `test.txt`, updated every 1000 cycles, which contains the
current loss, number of neurons, sum of multipliers, etc. (see code) Other
python files generate inputs or figures, or submit jobs to a cluster.


This code is largely based on Andrej Karpathy's [`min-char-rnn.py`](https://gist.github.com/karpathy/d4dee566867f8291f086) program.


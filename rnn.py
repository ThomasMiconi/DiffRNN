"""
Differentiable-structure RNN, by Thomas Miconi.

Largely based on minimal character-level Vanilla RNN model by Andrej Karpathy (@karpathy): https://gist.github.com/karpathy/d4dee566867f8291f086

BSD License

"""
import numpy as np
import math
import sys

# Global meta-parameters, modifiable by command line
g = {
'NBSTEPS' : 300000,
'COEFFMULTIPNORM' : 3e-5,
'EXPTYPE' : 'HARD',
'DELETIONTHRESHOLD': .05,
'MINMULTIP': .025,  # Must be lower than DELETIONTHRESHOLD !
'NBMARGIN' : 1,
'PROBADEL': .25,
'PROBAADD': .05,
'RNGSEED' : 0
}

# Command line parameters parsing

argpairs = [sys.argv[i:i+2] for i in range(1, len(sys.argv), 2)]
for argpair in argpairs:
    if not (argpair[0] in g):
        sys.exit("Error, tried to pass value of non-existent parameter "+argpair[0])
    if argpair[0] == 'EXPTYPE':
        g['EXPTYPE'] = argpair[1]
    else:
        g[argpair[0]] = float(argpair[1])

if (g['EXPTYPE'] not in ['HARD', 'EASY', 'HARDEASY', 'EASYHARDEASY']):
    sys.exit('Wrong EXPTYPE value')
g['NBMARGIN'] = int(g['NBMARGIN'])
g['RNGSEED'] = int(g['RNGSEED'])
print g

np.random.seed(g['RNGSEED'])


# data I/O
# NOTE: the input files are specified two directories up because I generally use the program with a different working directory. Modify as needed.
myf = open("test.txt", "w")
myf.close()
if (g['EXPTYPE'] == 'EASY') | (g['EXPTYPE'] == 'EASYHARDEASY'):
    data = open('../../inputeasy.txt', 'r').read() # should be simple plain text file
else:
    data = open('../../inputhard.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has', data_size, 'characters,', vocab_size, 'unique.'# % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
MAX_HIDDEN_SIZE = 100 # Maximum size of hidden layer of neurons (same as fixed size in original min-char-rnn.py)
hidden_size = 1 # size of hidden layer of neurons - start from 1 node.
seq_length = 40 # number of steps to unroll the RNN for 
learning_rate = 1e-1

# network parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
multips = .001 * np.ones((hidden_size, 1)); # multipliers 
multips[0,0] = 1.0 # Start with a multiplier of 1 on the single starting node.
Wiy = np.random.randn(vocab_size, hidden_size)*0.01 # hidden (after multiplier) to output. See below
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

ages = np.zeros(hidden_size) # Ages of all neurons. Not used at present.

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, intoys, ys, ps = {}, {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    intoys[t] = multips * hs[t]  # "intoy" is the output of the hidden layer after the multipliers, which is to be fed "into" y (through the Wiy weight matrix)
    ys[t] = np.dot(Wiy, intoys[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dmultips, dWiy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(multips), np.zeros_like(Wiy)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWiy += np.dot(dy, intoys[t].T) 
    dby += dy
    dintoy = np.dot(Wiy.T, dy) # dE/dIntoY, as a function of dE/dy
    
    # Gradient to be applied to the multipliers
    dmultips += (1.0 * dintoy * multips # This part descends the error gradient
            + g['COEFFMULTIPNORM'] * np.sign(multips)) # L1-norm regularization. The derivative of abs(x) is sign(x). Thus, descending the gradient of abs(x) over x is simply subtracting a constant multiple of sign(x). 
            # + .001 * multips) # This would add an L2-regularization term, which we don't use here.
    
    dh = dintoy * multips + dhnext
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dmultips, dWiy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dmultips, dWiy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Wiy, multips * h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mmultips, mWiy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(multips), np.zeros_like(Wiy)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0


while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dmultips, dWiy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: 
      print 'iter %d, position in data %d, loss: %f , nb hidden neurons %d, sum-abs multips: %f' % (n, p, smooth_loss, hidden_size, sum(abs(multips))), # print progress
      print multips.T
  if n % 1000 == 0: 
      with open("test.txt", "a") as myf:
        msg = "%d %d %f  %d %f" % (n, p, smooth_loss, hidden_size, sum(abs(multips))) # print progress
        myf.write(msg+"\n")
 

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, multips, Wiy, bh, by], 
                                [dWxh, dWhh, dmultips, dWiy, dbh, dby], 
                                [mWxh, mWhh, mmultips, mWiy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update


  # Neuron addition / deletion
  # Deletable neurons are those whose multipliers fall below threshold.
  # We want to delete excess below-threshold neurons, keeping only NBMARGIN below-threshold neuron at any time; or add one new neuron if no below-threshold neuron remains.

  ages += 1

  multips[multips < g['MINMULTIP']] = g['MINMULTIP']  # multipliers are clipped from below
  

  # Which neurons are above threshold ('selected' for preservation) ?
  sel = (abs(multips) > g['DELETIONTHRESHOLD'])[:,0] # | (ages < 500)
  
  if sum(sel) < hidden_size - g['NBMARGIN'] :
   
    # Preserve 1-PROBADEL% of the below-threshold neurons, in addition to NBMARGIN below-threshold neurons (NBMARGIN is usually set to 1).
    # (Perhaps select the most recent neurons for deletion? Future work.)
    deletable = np.where(sel == False)[0]
    np.random.shuffle(deletable)
    for xx in range(g['NBMARGIN']):
        sel[deletable[xx]] = True
    deletable = deletable[g['NBMARGIN']:]
    for x in deletable:
        if np.random.rand() > g['PROBADEL']: # Note that this is a test for preservation rather than deletion, hence the >
            sel[x] = True


    # Delete all other deletable neurons
    hidden_size = sum(sel)
    Whh = Whh[sel,:][:, sel]
    Wxh = Wxh[sel, :]
    multips = multips[sel]
    Wiy = Wiy[:, sel]
    bh = bh[sel]
    hprev = hprev[sel]
    ages = ages[sel]
    
    mWxh = mWxh[sel, :]
    mWhh = mWhh[sel,:][:, sel]
    mmultips = mmultips[sel]
    mWiy = mWiy[:, sel]
    mbh = mbh[sel]
    
  if hidden_size < MAX_HIDDEN_SIZE -1:
      if ( (sum((abs(multips) > g['DELETIONTHRESHOLD'])[:,0]) > hidden_size - g['NBMARGIN']) & (np.random.rand() < g['PROBAADD']))  \
        | (np.random.rand() < 1e-4):
      # Add a new neuron
          Whh = np.append(Whh, np.random.randn(1, hidden_size)*0.01, axis=0)
          Whh = np.append(Whh, np.random.randn(hidden_size+1, 1)*0.01, axis=1)
          Wxh = np.append(Wxh, np.random.randn(1, vocab_size)*0.01, axis=0)
          Wiy = np.append(Wiy, np.random.randn(vocab_size,1)*0.01, axis=1)
          bh = np.append(bh, np.zeros((1,1)), axis=0)
          hprev = np.append(hprev, np.zeros((1,1)), axis=0)
          multips = np.append(multips,  g['DELETIONTHRESHOLD'] * np.ones((1,1)), axis=0)  # Initial multiplier for new neurons is set to deletion threshold
          ages = np.append(ages, 0)

          mWhh = np.append(mWhh, np.zeros((1, hidden_size)), axis=0)
          mWhh = np.append(mWhh, np.zeros((hidden_size+1, 1)), axis=1)
          mWxh = np.append(mWxh, np.zeros((1, vocab_size)), axis=0)
          mWiy = np.append(mWiy, np.zeros((vocab_size,1)), axis=1)
          mbh = np.append(mbh, np.zeros((1,1)), axis=0)
          mmultips = np.append(mmultips, np.zeros((1,1)), axis=0)

          hidden_size += 1
          print "Adding Neuron"



  p += seq_length # move data pointer
  n += 1 # iteration counter 
  if (n == 100000) & (g['EXPTYPE'] == 'EASYHARDEASY'):
      data = open('../../inputhard.txt', 'r').read() # should be simple plain text file
      p = 0
  if (n == 100000) & (g['EXPTYPE'] == 'HARDEASY'):
      data = open('../../inputeasy.txt', 'r').read() # should be simple plain text file
      p = 0
  if (n == 200000) & (g['EXPTYPE'] == 'EASYHARDEASY'):
      data = open('../../inputeasy.txt', 'r').read() # should be simple plain text file
      p = 0
  if n > g['NBSTEPS']:
      sys.exit(0)


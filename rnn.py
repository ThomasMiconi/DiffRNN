"""
Differentiable-structure RNN, by Thomas Miconi.

Mostly based on minimal character-level Vanilla RNN model by Andrej Karpathy
(@karpathy): https://gist.github.com/karpathy/d4dee566867f8291f086

BSD License

"""
import numpy as np
import math
import sys

# Global meta-parameters, modifiable by command line
g = {
'ADDDEL': 1,
'ETA' : .01,
'NBNEUR': 40,   # Number of neurons for fixed-size experiments (ignored if adddel is 1)
'MAXDW': .01,  
'DIR' : '.',  # The directory of input text files
'NBSTEPS' : 100000,
'COEFFWPEN' : 1e-4, 
'EXPTYPE' : 'HARD',
'DELETIONTHRESHOLD': .05,
'MINMULTIP': .025,  # Must be lower than DELETIONTHRESHOLD ! NOTE: Has no effect in the current version of the code.
'NBMARGIN' : 1,
'PROBADEL': .05,
'PROBAADD': .01,
'RNGSEED' : 0
}

# Command line parameters parsing

argpairs = [sys.argv[i:i+2] for i in range(1, len(sys.argv), 2)]
for argpair in argpairs:
    if not (argpair[0] in g):
        raise Exception("Error, tried to pass value of non-existent parameter "+argpair[0])
    if argpair[0] == 'EXPTYPE' or argpair[0] == 'DIR':
        g[argpair[0]] = argpair[1]
    else:
        g[argpair[0]] = float(argpair[1])

if (g['EXPTYPE'] not in ['HARD', 'EASY', 'HARDEASY', 'EASYHARDEASY']):
    raise Exception('Wrong EXPTYPE value')
g['NBMARGIN'] = int(g['NBMARGIN'])
g['RNGSEED'] = int(g['RNGSEED'])
print g

np.random.seed(g['RNGSEED'])


# data I/O
myf = open("output.txt", "w")
myf.close()
if (g['EXPTYPE'] == 'EASY') | (g['EXPTYPE'] == 'EASYHARDEASY'):
    data = open(g['DIR'] + '/inputeasy.txt', 'r').read() # should be simple plain text file
else:
    data = open(g['DIR'] + '/inputhard.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has', data_size, 'characters,', vocab_size, 'unique.'# % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
MAX_HIDDEN_SIZE = 100 # Maximum size of hidden layer of neurons (same as fixed size in original min-char-rnn.py)
if g['ADDDEL']:
    hidden_size = 1 # size of hidden layer of neurons - start from 1 node.
else:
    hidden_size = g['NBNEUR'] # fixed size
seq_length = 40 # number of steps to unroll the RNN for 
learning_rate = g['ETA']

# network parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden (after multiplier) to output. See below
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias
normz = np.zeros_like(bh)

ages = np.zeros(hidden_size) # Ages of all neurons. Not used at present.

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh),  np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T) 
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  #for dparam in [dWxh, dWhh,  dWhy, dbh, dby]:
  #  np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients - clipping is actually done just before the update, after learning_rate has been applied - see below
  return loss, dWxh, dWhh,  dWhy, dbh, dby, hs[len(inputs)-1]

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
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh,  mWhy = .01 * np.ones_like(Wxh), .01 * np.zeros_like(Whh),  .01 * np.zeros_like(Why)
mbh, mby = .01 * np.ones_like(bh), .01 * np.zeros_like(by) # memory variables for RMSProp
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
  loss, dWxh, dWhh,  dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.99 + loss * 0.01
  if n % 100 == 0: 
      print 'iter %d, position in data %d, loss: %f , nb hidden neurons %d, sum-abs norms: %f' % (n, p, smooth_loss, hidden_size, sum(abs(normz))), # print progress
      print normz.T
  if n % 1000 == 0: 
      with open("output.txt", "a") as myf:
        msg = "%d %d %f  %d %f" % (n, p, smooth_loss, hidden_size, sum(abs(normz))) # print progress
        myf.write(msg+"\n")
 

  # perform parameter update with Adagrad
  
  for param, dparam, mem in zip([Wxh, Whh,  Why, bh, by], 
                                [dWxh, dWhh,  dWhy, dbh, dby], 
                                [mWxh, mWhh,  mWhy, mbh, mby]):
    # mem += dparam * dparam   # Adagrad
    mem += .01 * (dparam * dparam - mem)  # RMSProp
    RMSdelta = -learning_rate * dparam / np.sqrt(mem + 1e-8) # RMSProp update
    np.clip(RMSdelta, -g['MAXDW'], g['MAXDW'], out = RMSdelta)  # Clipping the weight modifications
    param += RMSdelta
 
  # Note that 1-norm penalty on weights is applied even for fized-size! If you want to have no penalty, set COEFFWPEN to 0 (but this will decrease performance).
  Why -= g['COEFFWPEN'] * np.sign(Why)
  Whh -= g['COEFFWPEN'] * np.sign(Whh)
  
  # Computing the L1-norm of outgoing weights for each neuron.
  # The norm of lateral weights is scaled by the number of neurons and multiplied by 4, so it should remain roughly similar to the norm of feedforward weights as the network changes size (there are 4 output neurons)
  normz = .5 * (np.sum(np.abs(Why), axis = 0) + 4.0 * np.sum(np.abs(Whh), axis = 0) / hidden_size)

  
  if g['ADDDEL']:


      # Neuron addition / deletion
      # Deletable neurons are those whose outgoing weights fall below a certain threshold in L1-norm.
      # We want to delete excess below-threshold neurons, keeping only NBMARGIN below-threshold neuron at any time; or add one new neuron if no below-threshold neuron remains. (Both with a certain probability)

      ages += 1

      #normz[normz < g['MINMULTIP']] = g['MINMULTIP']  # outgoing weight norms are clipped from below
      

      # Which neurons are above threshold ('selected' for preservation) ?
      sel = abs(normz) > g['DELETIONTHRESHOLD']#[0] # | (ages < 500)
      
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
        normz = normz[sel]
        Why = Why[:, sel]
        bh = bh[sel]
        hprev = hprev[sel]
        ages = ages[sel]
        
        mWxh = mWxh[sel, :]
        mWhh = mWhh[sel,:][:, sel]
        mWhy = mWhy[:, sel]
        mbh = mbh[sel]
        

      # Addition of new neurons, if appropriate:
      if hidden_size < MAX_HIDDEN_SIZE -1:
          if ( (sum((abs(normz) > g['DELETIONTHRESHOLD'])) > hidden_size - g['NBMARGIN']) & (np.random.rand() < g['PROBAADD']))  \
            | (np.random.rand() < 1e-4):

              Whh = np.append(Whh, np.random.randn(1, hidden_size)*0.01, axis=0)
              Wxh = np.append(Wxh, np.random.randn(1, vocab_size)*0.01, axis=0)

            # The (absolute values of) outgoing weights of the added neuron must sum to g['DELETIONTHRESHOLD']
              newWhy = np.random.randn(vocab_size,1)
              newWhy = .5 * g['DELETIONTHRESHOLD'] * newWhy / (1e-8 + np.sum(abs(newWhy)))
              Why = np.append(Why, newWhy, axis=1)
              
              newWhh = np.random.randn(hidden_size+1, 1)
              newWhh = .5 * hidden_size * g['DELETIONTHRESHOLD'] * newWhh / (1e-8 + 4.0 * np.sum(abs(newWhh)))
              #newWhh *= .01
              Whh = np.append(Whh, newWhh, axis=1)

              bh = np.append(bh, np.zeros((1,1)), axis=0)
              hprev = np.append(hprev, np.zeros((1,1)), axis=0)
              #normz = np.append(normz,  g['DELETIONTHRESHOLD'] )  
              ages = np.append(ages, 0)

              mWhh = np.append(mWhh, .01 * np.ones((1, hidden_size)), axis=0)
              mWhh = np.append(mWhh, .01 * np.ones((hidden_size+1, 1)), axis=1)
              mWxh = np.append(mWxh, .01 * np.ones((1, vocab_size)), axis=0)
              mWhy = np.append(mWhy, .01 * np.ones((vocab_size,1)), axis=1)
              mbh = np.append(mbh, .01 * np.ones((1,1)), axis=0)

              hidden_size += 1
              print "Adding Neuron"



  p += seq_length # move data pointer
  n += 1 # iteration counter 
  if (n == int(g['NBSTEPS'] / 3)) & (g['EXPTYPE'] == 'EASYHARDEASY'):
      data = open(g['DIR'] + '/inputhard.txt', 'r').read() # should be simple plain text file
      p = 0
  if (n == int(g['NBSTEPS'] / 2)) & (g['EXPTYPE'] == 'HARDEASY'):
      data = open(g['DIR'] + '/inputeasy.txt', 'r').read() # should be simple plain text file
      p = 0
  if (n == int(2 * g['NBSTEPS'] / 3)) & (g['EXPTYPE'] == 'EASYHARDEASY'):
      data = open(g['DIR'] + '/inputeasy.txt', 'r').read() # should be simple plain text file
      p = 0
  if n > g['NBSTEPS']:
      print "Done!"
      sys.exit(0)


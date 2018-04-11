import numpy as np
import nltk
class RNN ():
  def __init__(self , data):
    self.data = data.split()
    chars = list(set(self.data))
    data_size, self.vocab_size = len(self.data), len(chars)
    print('data has %d chars, %d unique' % (data_size, self.vocab_size))
    self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
    self.ix_to_char = {i: ch for i, ch in enumerate(chars)}
    self.hidden_size = 100
    self.seq_length = 25
    self.learning_rate = 1e-1
    self.Wxh = np.random.rand(self.hidden_size, self.vocab_size)*0.01   # input to hidden
    self.Whh = np.random.rand(self.hidden_size, self.hidden_size)*0.01  # hidden to hidden
    self.Why = np.random.rand(self.vocab_size, self.hidden_size) *0.01  # hidden to output
    self.bh = np.zeros((self.hidden_size, 1))
    self.by = np.zeros((self.vocab_size, 1))
    self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad

  def embedding(self,input):
    self.embedded_word = {}
    self.loadGloveModel()
    input = list(set(input))
    input = nltk.word_tokenize(input)
    for word in input:
      if (len(self.glove.get(word, "none")) == len(self.glove['a'])):
        self.embedded_word[word] = self.glove[word]
      else:
        self.embedded_word[word] = np.random.rand((len(self.glove['a'])))

  def loadGloveModel(self):
    print("Loading Glove Model")
    f = open("glove.6B.50d.txt", 'r')
    self.glove = {}
    for line in f:
      splitLine = line.split()
      self.glove[splitLine[0]] = np.array(splitLine[1:], np.float32)
    print("Done.", len(self.glove), " words loaded!")

  def train(self,inputs , targets , hprev):
    loss , self.xs, self.hs, self.ys, self.ps = self.feed_forward(inputs,targets,hprev)

    dWxh, dWhh, dWhy, dbh, dby = self.back_propagation(inputs,targets)

    self.adgard(dWxh, dWhh, dWhy, dbh, dby)

    return loss,self.hs[len(inputs) - 1]

  def feed_forward(self,inputs,targets,hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    for t in range(len(inputs)):
      xs[t] = np.zeros((self.vocab_size, 1))
      print(np.shape(xs[t]))
      xs[t][inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)
      ys[t] = np.dot(self.Why, hs[t]) + self.by
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
      loss += -np.log(ps[t][targets[t], 0])
    return loss,xs, hs, ys, ps

  def back_propagation(self,inputs,targets):
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(self.hs[0])
    for t in reversed(range(len(inputs))):
      dy = np.copy(self.ps[t])
      dy[targets[t]] -= 1  # backprop into y
      dWhy += np.dot(dy, self.hs[t].T)
      print(np.shape(dby) , np.shape(dy))
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
      dhraw = (1 - self.hs[t] * self.hs[t]) * dh  # backprop through tanh nonlinearity
      dbh += dhraw  # derivative of hidden bias
      dWxh += np.dot(dhraw, self.xs[t].T)  # derivative of input to hidden layer weight
      dWhh += np.dot(dhraw, self.hs[t - 1].T)  # derivative of hidden layer to hidden layer weight
      dhnext = np.dot(self.Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return dWxh, dWhh, dWhy, dbh, dby

  def adgard(self,dWxh, dWhh, dWhy, dbh, dby):

    for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):

      mem += dparam * dparam
      param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

  def test(self,hperv, seed_ix, n):
    h = np.copy(hperv)
    x = np.zeros((self.vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      x = np.zeros((self.vocab_size, 1))
      x[ix] = 1
      ixes.append(ix)
    txt = ' '.join(self.ix_to_char[ix] for ix in ixes)
    print('----\n %s \n----' % (txt,))

  def fit(self):
    n, p = 0, 0
    smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # loss at iteration 0
    while n <= 100000:
      if p + self.seq_length + 1 >= len(self.data) or n == 0:
        hprev = np.zeros((self.hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data

      inputs = [self.char_to_ix[ch] for ch in self.data[p:p + self.seq_length]]
      targets = [self.char_to_ix[ch] for ch in self.data[p + 1:p + self.seq_length + 1]]

      loss , hprev = self.train(inputs, targets, hprev)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001

      if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))
        print(self.ix_to_char[inputs[0]])
        self.test(hprev, inputs[0], 50)

      p += self.seq_length  # move data pointer
      n += 1  # iteration counter

data = open('way_down', 'r').read()
rnn = RNN(data)
rnn.fit()
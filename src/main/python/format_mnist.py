import cPickle, gzip, numpy

f = gzip.open('../../../data/mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x, y = train_set
x = x * 255

mini = []

i = 0
N = 1000

while (i < N):
  ti = numpy.insert(x[i], 0, y[i])
  mini.append(ti)
  i += 1

a = numpy.array(mini)
a = numpy.array(a, dtype='int')

numpy.savetxt("../../../data/mnist/train" + str(N) + ".txt", a, fmt='%i')

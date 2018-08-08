"""
  code for MNN-H2 2d. Non-translation invariant case: Mixture of LC and Conv
  reference:
  Y Fan, J Feliu-Faba, L Lin, L Ying, L Zepeda-Núnez, A multiscale neural network based on
  hierarchical nested bases, arXiv preprint arXiv:1808.02376

  written by Yuwei Fan (ywfan@stanford.edu)
"""
# ------------------ keras ----------------
from keras.models import Model
# layers
from keras.layers import Input, Conv2D, LocallyConnected2D, ZeroPadding2D, Flatten, Lambda
from keras.layers import Add, Reshape

from keras import backend as K
# from keras import regularizers, optimizers
# from keras.engine.topology import Layer
# from keras.constraints import non_neg
# from keras.utils import np_utils
# from keras.utils import plot_model
from keras.callbacks import LambdaCallback, ReduceLROnPlateau

import os
import timeit
import argparse
import h5py
import numpy as np
import math, random

K.set_floatx('float32')

parser = argparse.ArgumentParser(description='RTE - MNN-H2')
parser.add_argument('--epoch', type=int, default=6000, metavar='N',
                    help='input number of epochs for training (default: %(default)s)')
parser.add_argument('--input-prefix', type=str, default='rte2dv1g2', metavar='N',
                    help='prefix of input data filename (default: %(default)s)')
parser.add_argument('--alpha', type=int, default=6, metavar='N',
                    help='number of channels for training (default: %(default)s)')
parser.add_argument('--k-grid', type=int, default=5, metavar='N',
                    help='number of grids (L+1, N=2^L*m) (default: %(default)s)')
parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
                    help='number layer of CNNs (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: %(default)s)')
parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                    help='batch size (default: #train samples/50)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: %(default)s)')
parser.add_argument('--output-suffix', type=str, default='None', metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--percent', type=float, default=0.5, metavar='precent',
                    help='percentage of number of total data(default: %(default)s)')

args = parser.parse_args()
# setup: parameters
N_epochs = args.epoch
alpha = args.alpha
k_multigrid = args.k_grid
N_cnn = args.n_cnn
lr = args.lr

Nsamples = 6000

best_err_train = 1e-2
best_err_test = 1e-2
best_err_T_train = 10
best_err_T_test = 10
best_err_train_max = 10
best_err_test_max = 10

# preparation for output
data_path = 'data/'
log_path = 'logs2d/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
outputfilename = log_path + 't2dH2L' + str(k_multigrid) + 'Nc' + str(N_cnn) + 'Al' + str(alpha);
if(args.output_suffix == 'None'):
    outputfilename += str(os.getpid()) + '.txt'
else:
    outputfilename += args.output_suffix + '.txt'
os = open(outputfilename, "w+")

def output(obj):
    print(obj)
    os.write(str(obj)+'\n')
def outputnewline():
    os.write('\n')
    os.flush()

filenameIpt = data_path + args.input_prefix + '.h5'

print('Reading data...')
fInput = h5py.File(filenameIpt,'r')
InputArray = fInput['Input'][:,:,0:Nsamples]
OutputArray = fInput['Output'][:,:,0:Nsamples]
mua = fInput['sa'][0]
print('Reading data finished')

InputArray = np.transpose(InputArray, (2,1,0))
OutputArray = np.transpose(OutputArray, (2,1,0))
print(InputArray.shape)

assert InputArray.shape[0] == Nsamples
Nx = InputArray.shape[1]
Ny = InputArray.shape[2]

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output("(Nx, Ny)                = (%d, %d)" % (Nx, Ny))
output("Nsamples                = %d" % Nsamples)
outputnewline()

assert OutputArray.shape[0] == Nsamples
assert OutputArray.shape[1] == Nx
assert OutputArray.shape[2] == Ny

n_input = (Nx, Ny)
n_output = (Nx, Ny)

# train data
n_train = int(Nsamples * args.percent)
n_train = min(n_train, 20000)
n_test = Nsamples - n_train
n_test = min(n_train, n_test)
n_test = max(n_test, 5000)

if args.batch_size == 0:
    BATCH_SIZE = n_train // 50
else:
    BATCH_SIZE = args.batch_size

# pre-treat the data
mean_out = np.mean(OutputArray[0:n_train, :, :])
InputArray += mua
OutputArray -= mean_out
output("mean of output is %.6f" % mean_out)

X_train = InputArray[0:n_train, :, :]
Y_train = OutputArray[0:n_train, :, :]
X_test  = InputArray[(Nsamples-n_test):Nsamples, :, :]
Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :, :]

output("[n_input, n_output] = [(%d,%d),  (%d,%d)]" % (n_input[0], n_input[1], n_output[0], n_output[1]))
output("[n_train, n_test]   = [%d, %d]" % (n_train, n_test))

X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1],  X_test.shape[2],  1])

# parameters
# Nx = 2^L *m, L = k_multigrid-1
m = Nx // (2**(k_multigrid - 1))
output('m = %d' % m)

# functions
def matrix2tensor(x, w):
    ns = x.shape[0]
    nx = int(x.shape[1])
    ny = int(x.shape[2])
    nw = int(x.shape[3])
    assert nw == 1
    assert nx%w == 0
    assert ny%w == 0
    y = K.reshape(x, (-1, nx//w, w, nx//w, w))
    z = K.permute_dimensions(y, (0,1,3,2,4))
    return K.reshape(z, (-1, nx//w, ny//w, w**2))

def tensor2matrix(x, w):
    ns = x.shape[0]
    nx = int(x.shape[1])
    ny = int(x.shape[2])
    w2 = int(x.shape[3])
    assert w2 == w**2
    y = K.reshape(x, (-1, nx, ny, w, w))
    z = K.permute_dimensions(y, (0, 1, 3, 2, 4))
    return K.reshape(z, (-1, nx*w, ny*w))

def reshape_interpolation(x):
    ns = x.shape[0]
    nx = int(x.shape[1])
    ny = int(x.shape[2])
    n_chan = int(x.shape[3])//4
    y = K.reshape(x, (-1, nx, ny, n_chan, 2, 2))
    z = K.permute_dimensions(y, (0, 1, 4, 2, 5, 3))
    return K.reshape(z, (-1, 2*nx, 2*ny, n_chan))

# test
def test_data(X, Y, string):
    Yhat = model.predict(X, max(500, BATCH_SIZE))
    dY = Yhat - Y
    errs = np.linalg.norm(dY, axis=(1,2)) / np.linalg.norm(Y+mean_out, axis=(1,2))
    output("max/ave error of %s data:\t %.1e %.1e" % (string, np.amax(errs), np.mean(errs)))
    return errs

flag = True
def checkresult(epoch, step):
    global best_err_train, best_err_test, best_err_train_max, best_err_test_max, flag, best_err_T_train, best_err_T_test
    t1 = timeit.default_timer()
    if((epoch+1)%step == 0):
        err_train = test_data(X_train, Y_train, 'train')
        err_test  = test_data(X_test, Y_test, 'test')
        if(best_err_train > np.mean(err_train)):
            best_err_train = np.mean(err_train)
            best_err_test = np.mean(err_test)
            best_err_train_max = np.amax(err_train)
            best_err_test_max = np.amax(err_test)
            best_err_T_train = np.var(err_train)
            best_err_T_test = np.var(err_test)
        t2 = timeit.default_timer()
        if(flag):
          output("runtime of checkresult = %.2f secs" % (t2-t1))
          flag = False
        output('best train and test error = %.1e, %.1e,\t fit time    = %.1f secs' % (best_err_train, best_err_test, (t2 - start)))
        output('best train and test error var, max = %.1e, %.1e, %.1e, %.1e' % (best_err_T_train, best_err_train_max, best_err_T_test, best_err_test_max))
        outputnewline()

def outputvec(vec, string):
    os.write(string+'\n')
    for i in range(0, vec.shape[0]):
        os.write("%.6e\n" % vec[i])

# u^l = UMV^Tv: v --> Vv --> MVv --> ul = UMVv
# uad = Aad v: uad = Aadv
# Us = [ul, l=2,...,L, uad]
n_b_ad = 1 # see the paper arXiv:1807.01883
n_b_2 = 2
n_b_l = 3
Ipt = Input(shape=(n_input[0], n_input[1], 1))
Vv_list = []

L = k_multigrid - 1
Vv = LocallyConnected2D(alpha, (m, m), strides=(m, m), activation='linear')(Ipt)
Vv_list.insert(0, Vv)
for ll in range(L-1, 1, -1):
    Vv = LocallyConnected2D(alpha, (2, 2), strides=(2, 2), activation='linear')(Vv)
    Vv_list.insert(0, Vv)

MVv_list = []
MVv = Vv_list[0]
for i in range(0, N_cnn):
    MVv = ZeroPadding2D((n_b_2, n_b_2))(MVv)
    MVv = Conv2D(alpha, (2*n_b_2+1, 2*n_b_2+1), activation='relu')(MVv)
MVv_list.append(MVv)
for k in range(1, len(Vv_list)):
    MVv = Vv_list[k]
    for i in range(0, N_cnn):
        MVv = ZeroPadding2D((n_b_l, n_b_l))(MVv)
        MVv = Conv2D(alpha, (2*n_b_l+1, 2*n_b_l+1), activation='relu')(MVv)
    MVv_list.append(MVv)

for ll in range(2, L):
    if ll == 2:
        chi = MVv_list[ll-2]
    else:
        chi = Add()([chi, MVv_list[ll-2]])
    chi = LocallyConnected2D(4*alpha, (1, 1), activation='linear')(chi)
    chi = Lambda(lambda x: reshape_interpolation(x))(chi)
chi = Add()([chi, MVv_list[L-2]])
chi = LocallyConnected2D(m**2, (1, 1), activation='linear')(chi)
chi = Lambda(lambda x: tensor2matrix(x, m))(chi)

uad = Lambda(lambda x: matrix2tensor(x, m))(Ipt)
for i in range(0, N_cnn-1):
    uad = ZeroPadding2D((n_b_ad, n_b_ad))(uad)
    uad = Conv2D(m**2, (2*n_b_ad+1, 2*n_b_ad+1), activation='relu')(uad)

uad = ZeroPadding2D((n_b_ad, n_b_ad))(uad)
uad = Conv2D(m**2, (2*n_b_ad+1, 2*n_b_ad+1), activation='linear')(uad)
# uad = LocallyConnected2D(m, (2*n_b_ad+1, 2*n_b_ad+1), activation='linear')(uad)
uad = Lambda(lambda x: tensor2matrix(x, m))(uad)

Opt = Add()([chi, uad])

# model
model = Model(inputs=Ipt, outputs=Opt)
# plot_model(model, to_file='mnnH2.png', show_shapes=True)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

start = timeit.default_timer()
RelativeErrorCallback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: checkresult(epoch, 10))
#ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
#        verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
model.optimizer.lr = (lr)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=N_epochs, verbose=args.verbose,
        callbacks=[RelativeErrorCallback])

checkresult(1,1)
err_train = test_data(X_train, Y_train, 'train')
err_test  = test_data(X_test, Y_test, 'test')
outputvec(err_train, 'Error for train data')
outputvec(err_test,  'Error for test data')

os.close()

log_os = open('trainresultMix2d.txt', "a")
log_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha, k_multigrid, N_cnn))
log_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
log_os.write('%.3e\t%.3e\t' % (best_err_train, best_err_test))
log_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' % (best_err_train_max, best_err_test_max, best_err_T_train, best_err_T_test))
log_os.write('\n')
log_os.close()

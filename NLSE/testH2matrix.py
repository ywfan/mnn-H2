"""
  code for MNN-H2.
  reference:
  Y Fan, J Feliu-Faba, L Lin, L Ying, L Zepeda-Núnez, A multiscale neural network based on
  hierarchical nested bases, arXiv preprint arXiv:1808.02376

  written by Yuwei Fan (ywfan@stanford.edu)
"""
# ------------------ keras ----------------
from keras.models import Model
# layers
from keras.layers import Input, Conv1D, Flatten, Lambda
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

parser = argparse.ArgumentParser(description='NLSE - MNN-H2')
parser.add_argument('--epoch', type=int, default=4000, metavar='N',
                    help='input number of epochs for training (default: %(default)s)')
parser.add_argument('--input-prefix', type=str, default='nlse2v2', metavar='N',
                    help='prefix of input data filename (default: %(default)s)')
parser.add_argument('--alpha', type=int, default=6, metavar='N',
                    help='number of channels for training (default: %(default)s)')
parser.add_argument('--k-grid', type=int, default=7, metavar='N',
                    help='number of grids (L+1, N=2^L*m) (default: %(default)s)')
parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
                    help='number layer of CNNs (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: %(default)s)')
parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                    help='batch size (default: #train samples/100)')
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

best_err_train = 1e-2
best_err_test = 1e-2
best_err_T_train = 10
best_err_T_test = 10
best_err_train_max = 10
best_err_test_max = 10

# preparation for output
data_path = 'data/'
log_path = 'logs/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
outputfilename = log_path + 'tH2L' + str(k_multigrid) + 'Nc' + str(N_cnn);
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

filenameIpt = data_path + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_path + 'Output_' + args.input_prefix + '.h5'

# import data: size of data: Nsamples * Nx
fInput = h5py.File(filenameIpt, 'r')
InputArray = fInput['Input'][:]

fOutput = h5py.File(filenameOpt, 'r')
OutputArray = fOutput['Output'][:]

Nsamples = InputArray.shape[0]
Nx = InputArray.shape[1]

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("Nx                      = %d" % Nx)
output("Nsamples                = %d" % Nsamples)
outputnewline()

assert OutputArray.shape[0] == Nsamples
assert OutputArray.shape[1] == Nx

n_input = Nx
n_output = Nx

# train data
n_train = int(Nsamples * args.percent)
n_train = min(n_train, 20000)
n_test = Nsamples - n_train
n_test = min(n_train, n_test)
n_test = max(n_test, 5000)

if args.batch_size == 0:
    BATCH_SIZE = n_train // 100
else:
    BATCH_SIZE = args.batch_size

# pre-treat the data
mean_out = np.mean(OutputArray[0:n_train, :])
mean_in  = np.mean(InputArray[0:n_train, :])
output("mean of input / output is %.6f\t %.6f" % (mean_in, mean_out))
InputArray /= mean_in * 2
InputArray -= 0.5
OutputArray -= mean_out

X_train = InputArray[0:n_train, :]   #equal to 0:(n_train-1) in matlab
Y_train = OutputArray[0:n_train, :]
X_test  = InputArray[(Nsamples-n_test):Nsamples, :]
Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :]

output("[n_input, n_output] = [%d, %d]" % (n_input, n_output))
output("[n_train, n_test] = [%d, %d]" % (n_train, n_test))

X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1], 1])

# parameters
# Nx = 2^L *m, L = k_multigrid-1
m = Nx // (2**(k_multigrid - 1))
output('m = %d' % m)

# functions
def padding(x, size):
    return K.concatenate([x[:,x.shape[1]-size//2:x.shape[1],:], x, x[:,0:(size-size//2-1),:]], axis=1)

# test
def test_data(X, Y, string):
    Yhat = model.predict(X, batch_size=max(BATCH_SIZE,1000))
    errs = np.linalg.norm(Y - Yhat, axis=1) / np.linalg.norm(Y+mean_out, axis=1)
    output("ave/max error of %s data:\t %.1e %.1e" % (string, np.mean(errs), np.amax(errs)))
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
Ipt = Input(shape=(n_input,  1)) # Ipt = v
Vv_list = []

L = k_multigrid - 1
Vv = Conv1D(alpha, m, strides=m, activation='linear')(Ipt)
Vv_list.insert(0, Vv)
for ll in range(L-1, 1, -1):
    Vv = Conv1D(alpha, 2, strides=2, activation='linear')(Vv)
    Vv_list.insert(0, Vv)

MVv_list = []
MVv = Vv_list[0]
for i in range(0, N_cnn):
    MVv = Lambda(lambda x: padding(x, 2*n_b_2+1))(MVv)
    MVv = Conv1D(alpha, 2*n_b_2+1, activation='relu')(MVv)
MVv_list.append(MVv)
for k in range(1, len(Vv_list)):
    MVv = Vv_list[k]
    for i in range(0, N_cnn):
        MVv = Lambda(lambda x: padding(x, 2*n_b_l+1))(MVv)
        MVv = Conv1D(alpha, 2*n_b_l+1, activation='relu')(MVv)
    MVv_list.append(MVv)

for ll in range(2, L):
    if ll == 2:
        chi = MVv_list[ll-2]
    else:
        chi = Add()([chi, MVv_list[ll-2]])
    chi = Conv1D(2*alpha, 1, activation='linear')(chi)
    chi = Reshape((2**(ll+1), alpha))(chi)
chi = Add()([chi, MVv_list[L-2]])
chi = Conv1D(m, 1, activation='linear')(chi)
chi = Flatten()(chi)

uad = Reshape((Nx//m, m))(Ipt)
for i in range(0, N_cnn-1):
    uad = Lambda(lambda x: padding(x, 2*n_b_ad+1))(uad)
    uad = Conv1D(m, 2*n_b_ad+1, activation='relu')(uad)

uad = Lambda(lambda x: padding(x, 2*n_b_ad+1))(uad)
uad = Conv1D(m, 2*n_b_ad+1, activation='linear')(uad)
uad = Flatten()(uad)

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

log_os = open('trainresultH2.txt', "a")
log_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha, k_multigrid, N_cnn))
log_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
log_os.write('%.3e\t%.3e\t' % (best_err_train, best_err_test))
log_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' % (best_err_train_max, best_err_test_max, best_err_T_train, best_err_T_test))
log_os.write('\n')
log_os.close()

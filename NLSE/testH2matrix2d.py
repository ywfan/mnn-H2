"""
  MNN-H 2d
"""
# ------------------ keras ----------------
from keras.models import Sequential, Model
# layers
from keras.layers import Input, Activation, Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization, Add, multiply, dot, Reshape, SeparableConv2D, add
from keras.layers import Lambda

from keras import backend as K
from keras import regularizers, optimizers
from keras.engine.topology import Layer
from keras.constraints import non_neg
from keras.utils import np_utils
#from keras.utils import plot_model
from keras.callbacks import LambdaCallback, ReduceLROnPlateau

import tensorflow as tf

K.set_floatx('float32')

import os
import timeit
import argparse
import h5py
import numpy as np
import random
#np.random.seed(123)  # for reproducibility
#random.seed(123)
import math

parser = argparse.ArgumentParser(description='NLSE - MNN-H 2d')
parser.add_argument('--epoch', type=int, default=200, metavar='N',
                    help='input number of epochs for training (default: 200)')
parser.add_argument('--input-prefix', type=str, default='nlse2d2', metavar='N',
                    help='prefix of input data filename (default: nlse2d2)')
parser.add_argument('--alpha', type=int, default=6, metavar='N',
                    help='input number of channels for training (default: 6)')
parser.add_argument('--k-grid', type=int, default=5, metavar='N',
                    help='input number of grids (default: 5)')
parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
                    help='input number layer of CNNs (default: 5)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                    help='batch size (default: n_train/100)')
parser.add_argument('--inter-size', type=int, default=1, metavar='N',
                    help='number of points in interpolation(default: 1)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: 2)')
parser.add_argument('--output-suffix', type=str, default='None', metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--percent', type=float, default=2/3, metavar='precent',
                    help='percentage of number of total data(default: 2/3)')
args = parser.parse_args()
# setup: parameters
N_epochs = args.epoch
alpha = args.alpha
k_multigrid = args.k_grid
N_cnn = args.n_cnn
lr = args.lr
inter_size = args.inter_size #interpolate size

Nsamples = 300

best_err_train = 100
best_err_test = 100
best_err_train_ave = 100
best_err_test_ave = 100
best_err_train_max = 100
best_err_test_max = 100

# preparation for output
#data_direction = '/scratch/users/ywfan/NLSE/data2d/'
data_direction = 'data2d/'
log_direction = 'logs2d/'
outputfilename = log_direction + 't2dHL' + str(k_multigrid) + 'Nc' + str(N_cnn) + 'Al' + str(alpha);
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

filenameIpt = data_direction + 'Input_'  + args.input_prefix + '.h5'
filenameOpt = data_direction + 'Output_' + args.input_prefix + '.h5'

print('Reading data...')
fInput = h5py.File(filenameIpt,'r')
InputArray = fInput['Input'][:,:,0:Nsamples]

fOutput = h5py.File(filenameOpt,'r')
OutputArray = fOutput['Output'][:,:,0:Nsamples]
print('Reading data finished')

InputArray = np.transpose(InputArray, (2,1,0))
OutputArray = np.transpose(OutputArray, (2,1,0))
print(InputArray.shape)

assert InputArray.shape[0] == Nsamples
Nx = InputArray.shape[1]
Ny = InputArray.shape[2]

'''
for i in range(0,Nx):
  for j in range(0,Ny):
    print(InputArray[0,i,j], end=' ')
  print()
'''

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output('Output data filename    = %s' % filenameOpt)
output("(Nx, Ny)                = (%d, %d)" % (Nx, Ny))
output("Nsamples                = %d" % Nsamples)
outputnewline()

assert OutputArray.shape[0] == Nsamples
assert OutputArray.shape[1] == Nx
assert OutputArray.shape[2] == Ny

# pre-treat the data
InputArray /= 40
InputArray -= 0.5
OutputArray -= 1

n_input = (Nx, Ny)
n_output = (Nx, Ny)

# train data
n_train = int(Nsamples * args.percent)
n_train = min(n_train, 30000)
n_test = Nsamples - n_train
n_test = min(n_test, max(n_train, 5000))
if args.batch_size == 0:
    BATCH_SIZE = n_train // 100
else:
    BATCH_SIZE = args.batch_size

X_train = InputArray[0:n_train, :, :] #equal to 0:(n_train-1) in matlab
Y_train = OutputArray[0:n_train, :, :]
X_test  = InputArray[n_train:(n_train+n_test), :, :] #equal to n_train:(Nsamples-1) or n_train:end
Y_test  = OutputArray[n_train:(n_train+n_test), :, :]

output("[n_input, n_output] = [(%d,%d),  (%d,%d)]" % (n_input[0], n_input[1], n_output[0], n_output[1]))
output("[n_train, n_test]   = [%d, %d]" % (n_train, n_test))

X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1],  X_test.shape[2],  1])

# parameters
w_size = Nx // (2**(k_multigrid - 1))
w_size = 2 * ((w_size+1)//2) - 1 # if w_size is even, set it as w_size-1
output('w_size = %d' % w_size)

# functions
#channels last, i.e. x.shape = [batch_size, nx, ny, n_channels]
def padding2d(x, size_x, size_y):
    wx = size_x // 2
    wy = size_y // 2
    nx = x.shape[1]
    ny = x.shape[2]
    # x direction
    y = K.concatenate([x[:,nx-wx:nx,:,:], x, x[:,0:wx,:,:]], axis=1)
    # y direction
    z = K.concatenate([y[:,:, ny-wy:ny,:], y, y[:,:,0:wy,:]], axis=2)
    return z

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
    Yhat = model.predict(X)
    dY = Yhat - Y   #np.ndarray.flatten(Yhat)-np.ndarray.flatten(Y_test);
    #err_t = np.linalg.norm(dY)/np.linalg.norm(Y);
    #output("Total relative Error on the %s data is %.2e" % (string, err_t))
    errs = np.zeros((X.shape[0]));
    errs_res = np.zeros((X.shape[0]));
    for i in range(0, X.shape[0]):
        errs_res[i] = np.linalg.norm(dY[i,:,:]) / np.linalg.norm(Y[i,:,:])
        errs[i] = np.linalg.norm(dY[i,:,:]) / np.linalg.norm(Y[i,:,:]+1)
    output("max/ave error of %s data:\t %.1e %.1e\t %.1e (res) %.1e (res)"
            % (string, np.amax(errs), np.mean(errs), np.amax(errs_res), np.mean(errs_res)))
    return errs

flag = True
def checkresult(epoch, step):
    global best_err_train, best_err_test, best_err_train_max, best_err_test_max, best_err_train_ave, best_err_test_ave, flag
    t1 = timeit.default_timer()
    if((epoch+1)%step == 0):
        err_train = test_data(X_train, Y_train, 'train')
        err_test  = test_data(X_test, Y_test, 'test')
        betr = np.mean(err_train)
        bete = np.mean(err_test)
        betrm = np.amax(err_train)
        betem = np.amax(err_test)
        if(best_err_train+best_err_test > betr + bete):
            best_err_train = betr
            best_err_test = bete
        if ( (best_err_train_ave+best_err_test_ave+(best_err_train_max+best_err_test_max)/5) > (betr+bete+(betrm+betem)/5) ):
            best_err_train_ave = betr
            best_err_train_max = betrm
            best_err_test_ave = bete
            best_err_test_max = betem
        t2 = timeit.default_timer()
        if(flag):
          output("runtime of checkresult = %.2f secs" % (t2-t1))
          flag = False
        output('best train and test error = %.1e, %.1e,\t fit time    = %.1f secs' % (best_err_train, best_err_test, (t2 - start)))
        output('best train and test error ave/max = %.1e, %.1e, %.1e, %.1e' % (best_err_train_ave, best_err_train_max, best_err_test_ave, best_err_test_max))
        outputnewline()

def outputvec(vec, string):
    os.write(string+'\n')
    for i in range(0, vec.shape[0]):
        os.write("%.6e\n" % vec[i])

Ipt = Input(shape=(n_input[0], n_input[1], 1))
Vvs = []

L = k_multigrid - 1
Vv = Conv2D(alpha, (w_size, w_size), strides=(w_size,w_size), activation='linear')(Ipt)
Vvs.insert(0,Vv)
for ll in range(L-1, 1, -1):
    Vv = Conv2D(alpha, (2,2), strides=(2,2), activation='linear')(Vv)
    Vvs.insert(0,Vv)

MVvs = []
MVv = Vvs[0]
for i in range(0,N_cnn):
    MVv = Lambda(lambda x: padding2d(x, 5,5))(MVv)
    MVv = Conv2D(alpha, (5,5), activation='relu')(MVv)
MVvs.append(MVv)
for k in range(1, len(Vvs)):
    MVv = Vvs[k]
    for i in range(0,N_cnn):
        MVv = Lambda(lambda x: padding2d(x, 7,7))(MVv)
        MVv = Conv2D(alpha, (7,7), activation='relu')(MVv)
    MVvs.append(MVv)

for ll in range(2, L):
    if ll == 2:
        chi = MVvs[ll-2]
    else:
        chi = Add()([chi, MVvs[ll-2]])
    chi = Conv2D(4*alpha, (1,1), activation='linear')(chi)
    chi = Lambda(lambda x: reshape_interpolation(x))(chi)
chi = Add()([chi, MVvs[L-2]])
chi = Conv2D(w_size**2, (1,1), activation='linear')(chi)
chi = Lambda(lambda x: tensor2matrix(x, w_size))(chi)


# adjacent
#Layer = Reshape((Nx//w_size, Ny//w_size, w_size**2))(Ipt);
uad = Lambda(lambda x: matrix2tensor(x, w_size))(Ipt)
for i in range(0, N_cnn-1):
    uad = Lambda(lambda x: padding2d(x, 3, 3))(uad)
    uad = Conv2D(w_size**2, (3, 3), activation='relu')(uad)

uad = Lambda(lambda x: padding2d(x, 3, 3))(uad)
uad = Conv2D(w_size**2, (3, 3), activation='linear')(uad)
#Layer = Reshape((Nx, Ny))(Layer)
uad = Lambda(lambda x: tensor2matrix(x, w_size))(uad )

Opt = Add()([chi,uad])

# model
model = Model(inputs=Ipt, outputs=Opt)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

start = timeit.default_timer()
RelativeErrorCallback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: checkresult(epoch, 10))
ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
        verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
model.optimizer.lr = (lr)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=N_epochs, verbose=args.verbose,
        callbacks=[RelativeErrorCallback])

checkresult(1,1)
err_train = test_data(X_train, Y_train, 'train')
err_test  = test_data(X_test, Y_test, 'test')
outputvec(err_train, 'Error for train data')
outputvec(err_test,  'Error for test data')

os.close()

log_os = open('trainresult2d.txt', "a")
log_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha, k_multigrid, N_cnn))
log_os.write('%d\t%d\t' % (BATCH_SIZE, inter_size))
log_os.write('%d\t%d\t%d\t' % (n_train, n_test, model.count_params()))
log_os.write('%.3e\t%.3e\t' % (best_err_train, best_err_test))
log_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' % (best_err_train_ave, best_err_train_max, best_err_test_ave, best_err_test_max))
log_os.write('\n')
log_os.close()

# coding=utf-8
# vim: sw=4 et tw=100
"""
code for MNN-H2 2d. Non-translation invariant case: Mixture of LC and Conv
reference:
    Y Fan, J Feliu-Fabà, L Lin, L Ying, L Zepeda-Núñez, A multiscale neural network based on
    hierarchical nested bases, arXiv preprint arXiv:1808.02376

  written by Yuwei Fan (ywfan@stanford.edu)
"""
# ----------------- import keras tools ----------------------
from keras.models import Model
from keras.layers import Input, Conv2D, LocallyConnected2D, ZeroPadding2D, Add, Lambda

from keras import backend as K
from keras.callbacks import Callback

# ---------------- import python packages --------------------
import os
import timeit
import argparse
import h5py
import numpy as np
import math

# ---- define input parameters and set their default values ---
parser = argparse.ArgumentParser(description='RTE - MNN-H2 2d')
parser.add_argument('--epoch', type=int, default=4000, metavar='N',
                    help='input number of epochs for training (default: %(default)s)')
parser.add_argument('--input-prefix', type=str, default='rte2dv1g2', metavar='N',
                    help='prefix of input data filename (default: %(default)s)')
parser.add_argument('--alpha', type=int, default=6, metavar='N',
                    help='number of channels for training (default: %(default)s)')
parser.add_argument('--k-grid', type=int, default=5, metavar='N',
                    help='number of grids (L+1, N=2^L*m) (default: %(default)s)')
parser.add_argument('--n-cnn', type=int, default=5, metavar='N',
                    help='number of CNN layers (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: %(default)s)')
parser.add_argument('--batch-size', type=int, default=0, metavar='N',
                    help='batch size (default: #train samples/100)')
parser.add_argument('--verbose', type=int, default=2, metavar='N',
                    help='verbose (default: %(default)s)')
parser.add_argument('--output-suffix', type=str, default='None', metavar='N',
                    help='suffix output filename(default: )')
parser.add_argument('--percent', type=float, default=2./3., metavar='precent',
                    help='percentage of number of total data(default: %(default)s)')
parser.add_argument('--initialvalue', type=str, default='', metavar='filename',
                    help='filename storing the weights of the model (default: '')')
args = parser.parse_args()
# === setup: parameters
N_epochs = args.epoch
alpha = args.alpha
k_multigrid = args.k_grid
N_cnn = args.n_cnn
lr = args.lr

Nsamples = 30000  # 300 is for test. Set it as 30000 in real train

# ---------------- code for logs ---------------
data_path = 'data/'
log_path = 'logs2d/'
if not os.path.exists(log_path):
    os.mkdir(log_path)

outputfilename = log_path + 't2dH2MixL' + str(k_multigrid) + 'Nc' + str(N_cnn) + 'Al' + str(alpha)
if(args.output_suffix == 'None'):
    outputfilename += str(os.getpid())
else:
    outputfilename += args.output_suffix

modelfilename = outputfilename + '.h5'
outputfilename += '.txt'
log_os = open(outputfilename, "w+")

def output(obj):
    print(obj)
    log_os.write(str(obj)+'\n')

def outputnewline():
    log_os.write('\n')
    log_os.flush()

# ---------- prepare the training and test data sets ----------
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
[Nsamples, Nx, Ny] = InputArray.shape
assert OutputArray.shape == (Nsamples, Nx, Ny)

output(args)
outputnewline()
output('Input data filename     = %s' % filenameIpt)
output("(Nx, Ny)                = (%d, %d)" % (Nx, Ny))
output("Nsamples                = %d" % Nsamples)
outputnewline()

# === training and test data
n_train = int(Nsamples * args.percent)
n_test  = min(max(n_train, 5000), Nsamples - n_train)

if args.batch_size == 0:
    BATCH_SIZE = n_train // 50
else:
    BATCH_SIZE = args.batch_size

# === pre-treat the data
mean_out = np.mean(OutputArray[0:n_train, :, :])
InputArray += mua
OutputArray -= mean_out
output("mean of output is %.6f" % mean_out)

X_train = InputArray[0:n_train, :, :]
Y_train = OutputArray[0:n_train, :, :]
X_test  = InputArray[(Nsamples-n_test):Nsamples, :, :]
Y_test  = OutputArray[(Nsamples-n_test):Nsamples, :, :]

X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
X_test  = np.reshape(X_test,  [X_test.shape[0],  X_test.shape[1],  X_test.shape[2],  1])

# --------------- prepare parameters for MNN ------------------
n_input = (Nx, Ny)
n_output = (Nx, Ny)
output("[n_input, n_output] = [(%d,%d),  (%d,%d)]" %
       (n_input[0], n_input[1], n_output[0], n_output[1]))
output("[n_train, n_test]   = [%d, %d]" % (n_train, n_test))

# parameters: Nx = m*2^L, L = k_multigrid-1
m = Nx // (2**(k_multigrid - 1))
output('m = %d' % m)

# ----------------- functions used in MNN --------------------
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

# === calculate the relative error of the training/test data sets
def test_data(X, Y):
    Yhat = model.predict(X, max(500, BATCH_SIZE))
    errs = np.linalg.norm(Y - Yhat, axis=(1,2)) / np.linalg.norm(Y+mean_out, axis=(1,2))
    return errs

class SaveBestModel(Callback):
    """Save the best model
    # Arguments
        filename: string to save the model file.
        verbose: verbosity mode, 0 or 1.
        period: Interval (number of epochs) between checkpoints.
    """
    def __init__(self, filename, verbose=1, period=1):
        super(SaveBestModel, self).__init__()
        self.filename               = filename
        self.verbose                = verbose
        self.period                 = period
        self.best_epoch             = 0
        self.epochs_since_last_save = 0
        self.best_err_train         = 1
        self.best_err_test          = 1
        self.best_err_train_max     = 1
        self.best_err_test_max      = 1
        self.best_err_var_train     = 1
        self.best_err_var_test      = 1

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.compare_with_best_model(epoch)

    def compare_with_best_model(self, epoch):
        t1 = timeit.default_timer()
        err_train = test_data(X_train, Y_train)
        err_test  = test_data(X_test, Y_test)
        if(self.best_err_train + self.best_err_test > np.mean(err_train) + np.mean(err_test)):
            self.best_epoch         = epoch + 1
            self.best_err_train     = np.mean(err_train)
            self.best_err_test      = np.mean(err_test)
            self.best_err_train_max = np.amax(err_train)
            self.best_err_test_max  = np.amax(err_test)
            self.best_err_var_train = np.var(err_train)
            self.best_err_var_test  = np.var(err_test)
            # save weights of the model
            self.model.save_weights(self.filename, overwrite=True)

        if self.verbose:
            t2 = timeit.default_timer()
            output("Epoch %d:\t runtime of prediction = %.2f secs" % ((epoch + 1), (t2 - t1)))
            output("ave/max error of train/test data:\t %.1e %.1e \t %.1e %.1e " %
                   (np.mean(err_train), np.mean(err_test),
                    np.amax(err_train), np.amax(err_test)))
            output('best train/test error = %.2e, %.2e\t at epoch = %d,\t fit time = %.1f secs'
                   % (self.best_err_train, self.best_err_test, self.best_epoch, (t2 - start)))
            output('best train/test error: var / max = %.1e, %.1e, %.1e, %.1e'
                   % (self.best_err_var_train, self.best_err_train_max,
                      self.best_err_var_test, self.best_err_test_max))


# ------- LCR, LCK, CNNK, LCI layers, see arXiv:1808.02376 for details ------------
def LCR2D(X, Nin, al_in, Nout, al_out, act_fun):
    w = (Nin[0] // Nout[0], Nin[1] // Nout[1])
    return LocallyConnected2D(al_out, w, strides=w, activation=act_fun)(X)

def CNNK2D(X, Nin, al_in, al_out, w, act_fun):
    tmp = ZeroPadding2D((w[0]//2, w[1]//2))(X)
    return Conv2D(al_out, w, activation=act_fun)(tmp)

def LCK2D(X, Nin, al_in, al_out, w, act_fun):
    tmp = ZeroPadding2D((w[0]//2, w[1]//2))(X)
    return LocallyConnected2D(al_out, w, activation=act_fun)(tmp)

def LCI2D(X, Nin, al_in, al_out, act_fun):
    return LocallyConnected2D(al_out, 1, activation=act_fun)(X)


# ---------- architecture of MNN-H2 -------------------
# read it with the Algorithm 4 of arXiv:1807.01883
# u^l = UMV^Tv: v --> Vv --> MVv --> ul = UMVv
# uad = Aad v
L = k_multigrid - 1
# === Definition of matrix band sizes, see Property 1 of arXiv:1807.01883
n_b_2  = 2
n_b_l  = 3
n_b_ad = 1

Ipt = Input(shape=(n_input[0], n_input[1], 1))
# === adjacent part
uad = Lambda(lambda x: matrix2tensor(x, m))(Ipt)
for i in range(0, N_cnn-1):
    uad = CNNK2D(uad, (2**L, 2**L), m**2, m**2, (2*n_b_ad+1, 2*n_b_ad+1), 'relu')

uad = CNNK2D(uad, (2**L, 2**L), m**2, m**2, (2*n_b_ad+1, 2*n_b_ad+1), 'linear')
uad = Lambda(lambda x: tensor2matrix(x, m))(uad )

# === far field part
Vv_list = []
Vv = LCR2D(Ipt, n_input, 1, (2**L,2**L), alpha, 'linear')
Vv_list.insert(0, Vv)
for ll in range(L-1, 1, -1):
    Vv = LCR2D(Vv, (2**(ll+1),2**(ll+1)), alpha, (2**ll,2**ll), alpha, 'linear')
    Vv_list.insert(0, Vv)

MVv_list = []
for ll in range(2, L+1):
    MVv = Vv_list[ll-2]
    if ll == 2:
        w = (2*n_b_2 + 1, 2*n_b_2 + 1)
    else:
        w = (2*n_b_l + 1, 2*n_b_l + 1)

    for k in range(0, N_cnn):
        MVv = CNNK2D(MVv, (2**ll,2**ll), alpha, alpha, w, 'relu')

    MVv_list.append(MVv)

for ll in range(2, L):
    if ll == 2:
        chi = MVv_list[ll-2]
    else:
        chi = Add()([chi, MVv_list[ll-2]])

    chi = LCI2D(chi, (2**ll, 2**ll), alpha, 4*alpha, 'linear')
    chi = Lambda(lambda x: reshape_interpolation(x))(chi)

chi = Add()([chi, MVv_list[L-2]])
chi = LCI2D(chi, (2**L, 2**L), alpha, m**2, 'linear')
chi = Lambda(lambda x: tensor2matrix(x, m))(chi)

# === addition of far field and adjacent part
Opt = Add()([chi, uad])

# === model
model = Model(inputs=Ipt, outputs=Opt)
# plot_model(model, to_file='mnnH2.png', show_shapes=True)
model.compile(loss='mean_squared_error', optimizer='Nadam')
model.optimizer.schedule_decay = (0.004)
output('number of params      = %d' % model.count_params())
outputnewline()
model.summary()

start = timeit.default_timer()
model.optimizer.lr = (lr)

# if args.initialvalue is given, read weights from the file
if len(args.initialvalue) > 3:
    model.load_weights(args.initialvalue, by_name=False)
    model.optimizer.lr = (lr)
    output('initial the network by %s\n' % args.initialvalue)
    err_train = test_data(X_train, Y_train)
    err_test  = test_data(X_test, Y_test)
    output("ave/max error of train/test data:\t %.1e %.1e \t %.1e %.1e " %
           (np.mean(err_train), np.mean(err_test),
            np.amax(err_train), np.amax(err_test)))

save_best_model = SaveBestModel(modelfilename, period=10)
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=N_epochs, verbose=args.verbose,
          callbacks=[save_best_model])
log_os.close()

# === save summary of results
tr_os = open('trainresult2dH2Mix.txt', "a")
tr_os.write('%s\t%d\t%d\t%d\t' % (args.input_prefix, alpha, k_multigrid, N_cnn))
tr_os.write('%d\t%d\t%d\t%d\t' % (BATCH_SIZE, n_train, n_test, model.count_params()))
tr_os.write('%.3e\t%.3e\t' % (save_best_model.best_err_train, save_best_model.best_err_test))
tr_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' % (save_best_model.best_err_train_max,
                                          save_best_model.best_err_test_max,
                                          save_best_model.best_err_var_train,
                                          save_best_model.best_err_var_test))
tr_os.write('\n')
tr_os.close()

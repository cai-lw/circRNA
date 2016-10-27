import sys
import os
from argparse import ArgumentParser, ArgumentError
if sys.version_info < (3,):
    from ConfigParser import ConfigParser
else:
    from configparser import ConfigParser
from sklearn.metrics import *
from keras.models import Sequential
from keras.layers import *
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from datagen import *

parser = ArgumentParser()
parser.add_argument('-v', type=int, default=0, help='Index of the group used as verification group. Should be 0-9.')
parser.add_argument('--debug', action='store_true', help='Use [debug] configuration, instead of [run]')
parser.add_argument('--alu', action='store_true', help='Use Alu information to (hopefully) increase accuracy.')
args = parser.parse_args()
if args.v < 0 or args.v > 9:
    raise ArgumentError("VERIFICATION_GROUP must be between 0-9")

cf = ConfigParser()
cf.read('config.ini')
sec = 'debug' if args.debug else 'run'
TRAIN_MAXLEN = cf.getint(sec, 'train_maxlen')
VAL_MAXLEN = cf.getint(sec, 'val_maxlen')
BATCH_SIZE = cf.getint(sec, 'batch_size')
LEARNING_RATE = cf.getfloat(sec, 'learning_rate')
HIDDEN_NODES = cf.getint(sec, 'hidden_nodes')
SAMPLE_PER_EPOCH = cf.getint(sec, 'batch_per_epoch') * BATCH_SIZE
N_EPOCH = cf.getint(sec, 'n_epoch')
N_VAL_SAMPLES = cf.getint(sec, 'n_val_samples')
SAMPLE_PER_GROUP = cf.getint(sec, 'sample_per_group')
if SAMPLE_PER_GROUP == 0:
    SAMPLE_PER_GROUP = None
POS_RATIO = cf.getfloat(sec, 'pos_ratio')
if POS_RATIO == 0:
    POS_RATIO = None

try:
    os.mkdir(str(args.v))
except FileExistsError:
    pass
oldStdout = sys.stdout
file = open(str(args.v) + '/log', 'w')
sys.stdout = file

# model description
model = Sequential()
model.add(Masking(input_shape=(None, 5 if args.alu else 4)))
# Switch `consume_less` from `mem` to `gpu` if you have sufficient GPU memory
model.add(LSTM(HIDDEN_NODES, return_sequences=False, consume_less='gpu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=LEARNING_RATE), metrics=['accuracy'])
model.fit_generator(data_gen(filter(lambda x: x != args.v, range(10)), sample_per_iter=BATCH_SIZE,
    maxlen=TRAIN_MAXLEN, use_alu=args.alu, sample_per_group=SAMPLE_PER_GROUP, pos_ratio=POS_RATIO),
    samples_per_epoch=SAMPLE_PER_EPOCH, nb_epoch=N_EPOCH, callbacks=[
    ModelCheckpoint(filepath = str(args.v) + '/model', monitor='val_loss', mode='auto')])

def aupr(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)

# evaluation
xy = xy_gen()
y_pred = model.predict_generator(xy.x_gen([args.v], BATCH_SIZE, VAL_MAXLEN, args.alu), val_samples=N_VAL_SAMPLES)
y_true = xy.y_array(N_VAL_SAMPLES)
y_bin_pred = y_pred > 0.5
print('Accuracy: %.4f' % accuracy_score(y_true, y_bin_pred))
print('AUC: %.4f' % roc_auc_score(y_true, y_pred))
print('AUPR: %.4f' % aupr(y_true, y_pred))
print('F1 score: %.4f' % f1_score(y_true, y_bin_pred))
sys.stdout = oldStdout

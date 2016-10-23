from argparse import ArgumentParser, ArgumentError
from sklearn.metrics import *
from keras.models import Sequential
from keras.layers import *
from keras.layers.recurrent import LSTM
from datagen import *

parser = ArgumentParser()
parser.add_argument('-v', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--alu', action='store_true')
args = parser.parse_args()
if args.v < 0 or args.v > 9:
    raise ArgumentError("VERIFICATION_GROUP must be between 0-9")

SAMPLE_PER_GROUP = 1000
if args.debug:
    BATCH_SIZE = 10
    SAMPLE_PER_EPOCH = BATCH_SIZE
    N_EPOCH = 1
    N_VAL_SAMPLES = 10
else:
    # Set parameters for actual training here
    BATCH_SIZE = 20
    # Note: Log is printed every epoch.
    # Training ends after SAMPLE_PER_EPOCH * N_EPOCH samples are processed.
    # SAMPLE_PER_EPOCH / BATCH_SIZE must be an integer
    SAMPLE_PER_EPOCH = SAMPLE_PER_GROUP
    N_EPOCH = 9
    # N_VAL_SAMPLES / BATCH_SIZE must be an integer
    N_VAL_SAMPLES = 1000


# model description
model = Sequential()
model.add(Masking(input_shape=(MAXLEN, 5 if args.alu else 4)))
# 32 hidden nodes should work, but you can try larger number on powerful computers.
# Switch `consume_less` from `mem` to `gpu` if you have sufficient GPU memory
model.add(LSTM(32, return_sequences=False, consume_less='mem'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(data_gen(filter(lambda x: x != args.v, range(10)), BATCH_SIZE, SAMPLE_PER_GROUP, args.alu),
    samples_per_epoch=SAMPLE_PER_EPOCH, nb_epoch=N_EPOCH, verbose=2)

def aupr(y_true, y_pred):
    p, r, _ = precision_recall_curve(y_true, y_pred)
    return auc(r, p)

# evaluation
xy = xy_gen()
y_pred = model.predict_generator(xy.x_gen([args.v], BATCH_SIZE, SAMPLE_PER_GROUP, args.alu), val_samples=N_VAL_SAMPLES)
y_true = xy.y_array(N_VAL_SAMPLES)
y_bin_pred = y_pred > 0.5
print('Accuracy: %.4f' % accuracy_score(y_true, y_bin_pred))
print('AUC: %.4f' % roc_auc_score(y_true, y_pred))
print('AUPR: %.4f' % aupr(y_true, y_pred))
print('F1 score: %.4f' % f1_score(y_true, y_bin_pred))

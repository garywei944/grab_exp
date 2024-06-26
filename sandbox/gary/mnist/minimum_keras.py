import os

# os.environ['KERAS_BACKEND'] = 'theano'

from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout
from keras.layers import Activation, Flatten, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, ActivityRegularization
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

img_rows, img_cols = 28, 28
num_classes = 10
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()


# ... paste text from model.txt here
def dw_block(sh_l, prev_l):
    l = sh_l(prev_l)
    l = Activation(activation="relu")(l)
    l = InstanceNormalization(axis=None)(l)
    l = Dropout(rate=0.1)(l)
    return l


inputs = Input((28, 28, 1), dtype=np.float32)
l = inputs

l = Conv2D(8, (3, 3), input_shape=input_shape)(l)
l = Activation(activation="relu")(l)
l = InstanceNormalization(axis=None)(l)
l = MaxPooling2D((2, 2))(l)
l = Dropout(rate=0.1)(l)

l = SeparableConv2D(26, (3, 3), depth_multiplier=1)(l)
l = Activation(activation="relu")(l)
l = InstanceNormalization(axis=None)(l)
l = Dropout(rate=0.1)(l)

sh_l = SeparableConv2D(26, (3, 3), depth_multiplier=1, padding="same")

for n in range(3):
    l = dw_block(sh_l, l)

l = GlobalAveragePooling2D()(l)

l = Dense(16, activation="relu")(l)
l = InstanceNormalization(axis=None)(l)
l = Dropout(rate=0.1)(l)
l = Dense(10, activation="softmax")(l)

model = Model(inputs, l)

score = model.validate(x_test, y_test, verbose=1)
print("\n")
print("Test score: ", score[0])
print("Test accuracy: ", score[1])

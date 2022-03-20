from keras import backend as K
from keras.engine.training import Model as KerasModel
from keras.layers import Input, Dense, Activation, Flatten, Dropout, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img, transform_matrix_offset_center, \
    flip_axis, random_channel_shift
from keras.callbacks import Callback, EarlyStopping
from keras.models import load_model
from AVJPointDetection.data_loader import *


LR = 0.001
EPOCHS = 20001
BATCH_SIZE = 192

def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=3, mode=fill_mode, cval=cval) for x_channel
                      in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index + 1)
    return x


class LRDecay(Callback):
    def __init__(self, start=0.001, stop=0.0001, max_epochs=200):
        super(LRDecay, self).__init__()
        self.start, self.stop = start, stop
        self.ls = np.linspace(self.start, self.stop, max_epochs)

    def on_epoch_begin(self, epoch, logs={}):
        new_value = self.ls[epoch]
        K.set_value(self.model.optimizer.lr, new_value)


class CheckpointCallback(Callback):
    def __init__(self, start_index, save_periodic=True, period=500):
        super(CheckpointCallback, self).__init__()
        self.start_index = start_index
        self.save_periodic = save_periodic
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if self.save_periodic:
            if (self.start_index + epoch) % self.period == 0:
                fname = os.path.join('saved_{}.model'.format(self.start_index + epoch))
                self.model.save(fname)


def build_model():
    if K.image_dim_ordering() == 'th':
        inp = Input(shape=(1, INPUT_HEIGHT, INPUT_WIDTH), name='input')
    elif K.image_dim_ordering() == 'tf':
        inp = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 1), name='input')

    x = Convolution2D(32, 3, 3, border_mode='same')(inp)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.1)(x)

    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Convolution2D(128, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    x = Dense(1000)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(500)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(OUTPUTS)(x)
    outp = Activation('relu')(x)

    learning_method = Adam(lr=LR)
    model = KerasModel(inp, outp)
    model.compile(loss='mean_squared_error', optimizer=learning_method)

    print(model.summary())

    return model


def train_model(data_dir, data_csv, save_model_fname, prev_fname=None, start_index=0):
    _, imgs, pts1, pts2 = load_data(data_dir, data_csv)
    model = build_model()
    if prev_fname is not None:
        model.load_weights(prev_fname)

    y = zip(pts1, pts2)
    train_iter = create_generator().flow_from_imglist(imgs, y, target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                                                      batch_size=BATCH_SIZE, shuffle=True)
    model.fit_generator(train_iter,
                        samples_per_epoch=len(imgs),
                        nb_epoch=EPOCHS,
                        verbose=1,
                        callbacks=[CheckpointCallback(start_index), LRDecay(LR, LR / 100, EPOCHS)])

    model.save(save_model_fname)


def test_model(model_fname, test_dir, test_csv):
    _, imgs, pts1, pts2 = load_data(test_dir, test_csv)
    X, y = prepare_data(imgs, pts1, pts2)

    model = load_model(model_fname)
    print(model.evaluate(X, y))


if __name__ == '__main__':
    import time
    start = time.time()

    path_model_checkpoint = 'trained_models/saved_10000.model'  #if no model is saved, set as None
    path_model_output = 'trained_models/saved_final.model'
    path_CMRimages = "../datasets/CMRimages/testingset"
    path_groundtruth = "../datasets/Groundtruth"
    path_results = "../datasets/Groundtruth/S2_2_2ch.csv"

    train_model(path_CMRimages, path_groundtruth, path_model_output, path_model_checkpoint, start_index=0)

    end = time.time()
    print('execution time (model training): ', (end - start))

import os
import numpy as np
import skimage.io as skio
import skimage.transform as skt
from pandas.io.parsers import read_csv

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img, transform_matrix_offset_center, flip_axis, random_channel_shift  # ,  apply_transform

INPUT_HEIGHT = 128
INPUT_WIDTH = 128
OUTPUTS = 4

def load_data(data_dir, data_csv, load_pts=True):
    df = read_csv(data_csv)  # load pandas dataframe
    img_ids = df['ID']
    hR = 96
    xCnt = 128
    yCnt = 128
    imgs = []
    for img_name in img_ids:
        img = skio.imread('%s/%s.png' % (data_dir, img_name), as_gray=True)
        height, width = np.shape(img)[0:2]
        img = skt.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
        imgs.append(img)
    fScale = INPUT_HEIGHT * 1.0 / height
    if load_pts:
        # pts are not normalized
        x1 = df['X1'].values * fScale
        y1 = df['Y1'].values * fScale
        x2 = df['X2'].values * fScale
        y2 = df['Y2'].values * fScale

        pts1 = np.array(zip(x1, y1))
        pts2 = np.array(zip(x2, y2))

    print('Num of images: {}'.format(len(imgs)))

    if load_pts:
        return img_ids, imgs, pts1, pts2
    else:
        return img_ids, imgs

def create_generator():
    datagen = MyGenerator(rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
                          width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                          height_shift_range=0.05)  # randomly shift images vertically (fraction of total height)
    return datagen



class MyGenerator(ImageDataGenerator):
    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.2,
                 height_shift_range=0.2):

        super(MyGenerator, self).__init__(rotation_range=rotation_range,
                                          width_shift_range=width_shift_range,
                                          height_shift_range=height_shift_range,
                                          fill_mode='constant',
                                          cval=0.,
                                          dim_ordering='th')

    # override to return params
    def my_random_transform(self, x):
        # import pdb; pdb.set_trace()
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_axis - 1
        img_col_index = self.col_axis - 1
        img_channel_index = self.channel_axis - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        mat_rotation = np.array([[np.cos(theta), np.sin(theta), 0],
                                 [-np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        # tx is height, ty is width
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        mat_translation = np.array([[1, 0, -tx],
                                    [0, 1, -ty],
                                    [0, 0, 1]])
        if self.shear_range:
            raise RuntimeError('not implemented')
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            raise RuntimeError('not implemented')
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
        mat_transform = np.dot(mat_translation, mat_rotation)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        mat_transform = transform_matrix_offset_center(mat_transform, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)

        return x, mat_transform, tx, ty, theta

    def flow_from_imglist(self, X, y=None,
                          target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                          batch_size=BATCH_SIZE, shuffle=True, seed=None,
                          save_to_dir=None, save_prefix='', save_format='jpeg'):
        return ImgListIterator(X, y, self,
                               target_size=target_size,
                               batch_size=batch_size, shuffle=shuffle, seed=seed,
                               dim_ordering=self.dim_ordering,
                               save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class ImgListIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 target_size=(INPUT_HEIGHT, INPUT_WIDTH),
                 batch_size=BATCH_SIZE, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        # import pdb; pdb.set_trace()
        if y is not None and len(X) != len(y):
            raise Exception('X (images) and y (labels) '
                            'should have the same length. '
                            'Found: X : %s, y : %s' % (len(X), len(y)))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering = dim_ordering
        self.X = X  # list of images
        self.y = y  # list of tuples of points
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImgListIterator, self).__init__(len(X), batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        # default to th ordering here
        batch_x = np.zeros((current_batch_size, 1,) + self.target_size)
        batch_y = np.zeros((current_batch_size,) + (OUTPUTS,))
        # build batch of image data
        for i, j in enumerate(index_array):
            height, width = self.X[j].shape
            x = skt.resize(self.X[j], self.target_size)
            x = np.expand_dims(x, axis=0)
            x, transform_matrix, tx, ty, theta = self.image_data_generator.my_random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.y is not None:
                x1, y1 = self.y[j][0]
                x2, y2 = self.y[j][1]
                # offset_x = width / 2
                # offset_y = height / 2
                x1 = x1
                x2 = x2
                y1 = y1
                y2 = y2
                mat = np.array([[y1, y2], [x1, x2], [1, 1]])
                mat = np.dot(transform_matrix, mat)
                batch_y[i, 0] = mat[1, 0]
                batch_y[i, 1] = mat[0, 0]
                batch_y[i, 2] = mat[1, 1]
                batch_y[i, 3] = mat[0, 1]

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                gray_img = batch_x[i, 0, :, :]
                img = array_to_img(gray_img, dim_ordering='th', scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if 1 == 1:  # self.dim_ordering == 'tf':
            batch_tmp = np.zeros((current_batch_size,) + self.target_size + (1,))
            for i in range(current_batch_size):
                batch_tmp[i] = np.transpose(batch_x[i, 0:1, :, :], (1, 2, 0))

            batch_x = batch_tmp

        if self.y is None:
            return batch_x
        else:
            return batch_x, batch_y

def prepare_data(imgs, pts1=None, pts2=None):
    X = np.zeros((len(imgs), 1, INPUT_HEIGHT, INPUT_WIDTH))
    y = np.zeros((len(imgs), OUTPUTS))

    for i, img in enumerate(imgs):
        # height, width = img.shape
        height, width = 1, 1  # down sample to 128 128
        X[i] = skt.resize(img, (INPUT_HEIGHT, INPUT_WIDTH))
        if pts1 is not None and pts2 is not None:
            y[i, 0] = pts1[i][0] / width
            y[i, 1] = pts1[i][1] / height
            y[i, 2] = pts2[i][0] / width
            y[i, 3] = pts2[i][1] / height

    if pts1 is not None and pts2 is not None:
        return X, y
    else:
        return X
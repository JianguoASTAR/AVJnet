import matplotlib
import matplotlib.pyplot as plt
from AVJPointDetection.data_loader import *

INPUT_HEIGHT = 128
INPUT_WIDTH = 128
PATH_OUTPUTIMAGE = "../datasets/OutputImages"

#AVJ point detection
def predict(model_fname, data_dir, data_csv, out_csv, bShowFlag=False):
    img_ids, imgs, pts1, pts2 = load_data(data_dir, data_csv)
    X = prepare_data(imgs)

    X = np.transpose(X, (0, 2, 3, 1))

    model = load_model(model_fname)
    p = model.predict(X)

    fr = 512.0 / INPUT_HEIGHT

    df = DataFrame({'ID': img_ids,
                    'X1': (p[:, 0] * fr),
                    'Y1': (p[:, 1] * fr),
                    'X2': (p[:, 2] * fr),
                    'Y2': (p[:, 3] * fr)})

    df.to_csv(out_csv, index=False)

    # display(data_dir, out_csv)
    arrX1 = p[:, 0] - pts1[:, 0]
    arrY1 = p[:, 1] - pts1[:, 1]
    arrX2 = p[:, 2] - pts2[:, 0]
    arrY2 = p[:, 3] - pts2[:, 1]

    print(np.mean(abs(arrX1)), np.mean(abs(arrY1)), np.mean(abs(arrX2)), np.mean(abs(arrY2)))
    print(np.min(abs(arrX1)), np.min(abs(arrY1)), np.min(abs(arrX2)), np.min(abs(arrY2)))
    print(np.max(abs(arrX1)), np.max(abs(arrY1)), np.max(abs(arrX2)), np.max(abs(arrY2)))

    for i in range(len(img_ids)):
        pt1 = [p[i, 0], p[i, 1]]
        pt2 = [p[i, 2], p[i, 3]]
        if bShowFlag:
            show_img(imgs[i], pt1, pt2, pts1[i], pts2[i], img_ids[i])


def prepare_data(imgs, pts1=None, pts2=None):
    X = np.zeros((len(imgs), 1, INPUT_HEIGHT, INPUT_WIDTH))
    y = np.zeros((len(imgs), OUTPUTS))

    for i, img in enumerate(imgs):
        height, width = 1, 1
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

#display image with ground groudtruth
def display(data_dir, data_csv):
    img_ids, imgs, pts1, pts2 = load_data(data_dir, data_csv)

    for i, img_name in enumerate(img_ids):
        img = imgs[i]

        print (img_name)
        if (i % 1 == 0):
            if data_csv == 'pred.csv':
                show_img(img, pts1[i] * 2, pts2[i] * 2)
            else:
                show_img(img, pts1[i], pts2[i])


def show_img(img, pts1, pts2, gt1, gt2, title='Image'):
    x1, y1 = pts1
    x2, y2 = pts2

    gx1, gy1 = gt1
    gx2, gy2 = gt2

    dif_p1 = (abs(x1 - gx1) + abs(y1 - gy1)) / 2.0
    dif_p2 = (abs(x2 - gx2) + abs(y2 - gy2)) / 2.0

    plt.ioff()
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
    plt.scatter([x1, x2], [y1, y2], s=5, c='r')
    plt.scatter([gx1, gx2], [gy1, gy2], s=5, c='g')
    plt.title(title + '_' + str(dif_p1)[:5] + '_' + str(dif_p2)[:5])

    img_name = os.path.join(PATH_OUTPUTIMAGE, title + '.tif')
    plt.savefig(img_name)


if __name__ == '__main__':
    import time
    start = time.time()

    path_model = 'trained_models/saved_final.model'
    path_CMRimages = "../datasets/CMRimages/testingset/s2_2ch"
    path_groundtruth = "../datasets/Groundtruth/S2_2_2ch.csv"
    path_results = "../datasets/DetectionResults/S2_2_2ch.csv"

    predict(path_model, path_CMRimages , path_groundtruth, path_results, True)

    end = time.time()
    print ('execution time (model prediction): ', (end - start))

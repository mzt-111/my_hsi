import numpy as np
import math
import torch
import spectral
import scipy.io as scio
import matplotlib.pyplot as plt

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches_GCN(X, Y, L, mini_batch_size, seed):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    shuffled_L1 = L[permutation, :].reshape(L.shape[0], L.shape[1])
    shuffled_L = shuffled_L1[:, permutation].reshape(L.shape[0], L.shape[1])
    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size: k * mini_batch_size + mini_batch_size,
                       k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, Y, L)
    mini_batches.append(mini_batch)

    return mini_batches

def draw_map():
    gt = scio.loadmat('./Datasets/Botswana/Botswana_gt.mat')
    gt = gt['Botswana_gt']
    plt.figure()
    spectral.imshow(classes=gt.astype(int), figsize=(10,50))
    plt.show()

if __name__ == '__main__':
    draw_map()
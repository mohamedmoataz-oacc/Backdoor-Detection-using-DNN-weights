from read_onnx import get_onnx_weights, extract_layered_weights
from feature_extraction.pca import apply_pca
from random_projection import rp_network
import numpy as np


if __name__ == '__main__':
    model_path = "E:\\Desktop\\Final project\\code\\cifar10_resnet18_trojannn.onnx"

    try:
        rp_weights = np.load('cifar10_resnet18_trojannn.npy')
        print("Weights loaded from file")
    except:
        weights = get_onnx_weights(model_path)
        layers_weights = extract_layered_weights(weights)

        rp_weights = rp_network(layers_weights)
        print(rp_weights.shape)
        np.save('cifar10_resnet18_trojannn.npy', rp_weights)

    apply_pca(rp_weights)

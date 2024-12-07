from read_onnx import get_onnx_weights, extract_layered_weights
from feature_extraction.pca import apply_pca
from random_projection import rp_network
import numpy as np


if __name__ == '__main__':
    model_path = input("Enter model path: ")

    rp_components = 2000
    try:
        rp_weights = np.load(f'cifar10_resnet18_trojannn_{rp_components}.npy')
        print("Weights loaded from file")
    except:
        weights = get_onnx_weights(model_path)
        layers_weights = extract_layered_weights(weights)

        rp_weights = rp_network(layers_weights)
        print(rp_weights.shape)
        np.save(f'cifar10_resnet18_trojannn_{rp_components}.npy', rp_weights)

    pca_weights = apply_pca(rp_weights)
    print(pca_weights.shape)

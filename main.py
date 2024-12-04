from read_onnx import get_onnx_weights, extract_layered_weights
from random_projection import rp_network


if __name__ == '__main__':
    model_path = "E:\\Desktop\\Final project\\code\\cifar10_resnet18_trojannn.onnx"
    weights = get_onnx_weights(model_path)
    layers_weights = extract_layered_weights(weights)

    rp_weights = rp_network(layers_weights)
    print(rp_weights.shape)
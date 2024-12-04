import onnx
from onnx import numpy_helper


def get_onnx_weights(model_path):
    # onnx_model is an in-memory ModelProto
    onnx_model = onnx.load(model_path)

    INTIALIZERS = onnx_model.graph.initializer
    onnx_weights = dict()
    for initializer in INTIALIZERS:
        w = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = w

    return onnx_weights

def extract_layered_weights(onnx_weights):
    layered_weights = []
    for k, v in onnx_weights.items():
        if len(v.shape) != 1: layered_weights.append(v)
    return layered_weights

if __name__ == '__main__':
    model_path = "E:\\Desktop\\Final project\\code\\cifar10_resnet18_trojannn.onnx"
    onnx_weights = get_onnx_weights(model_path)
    for k, v in onnx_weights.items():
        print(k, v.shape)

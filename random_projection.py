import numpy as np
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection


def rp(weights: np.ndarray, n_components):
    # Flatten the weights
    weights_flattened = weights.reshape(1, -1)

    # Define the random projection
    grp = SparseRandomProjection(n_components=n_components)

    # Apply the random projection to the weights
    projected_weights = grp.fit_transform(weights_flattened)

    # print("Original shape:", weights_flattened.shape)
    # print("Projected shape:", projected_weights.shape)

    return projected_weights

def rp_network(layered_weights: np.ndarray, n_components=2000):
    projected_weights = []
    for weights_tensor in tqdm(layered_weights):
        projected_weights.append(rp(weights_tensor, n_components))
        # print('-----------------------')

    return np.vstack(projected_weights)

if __name__ == "__main__":
    # Example weights of a DNN layer (randomly generated for illustration)
    weights = [
        np.random.rand(64, 3, 7, 7),
        np.random.rand(128, 128, 3, 3),
        np.random.rand(128, 64, 1, 1),
        np.random.rand(128, 128, 3, 3),
        np.random.rand(128, 128, 3, 3),
        np.random.rand(256, 128, 3, 3),
        np.random.rand(512, 512, 3, 3),
        np.random.rand(1000, 512),
    ]

    projected_weights = []
    for weight in weights:
        projected_weights.append(rp(weight))
        print('-----------------------')

    weights_tensor = np.vstack(projected_weights)
    print(weights_tensor.shape)

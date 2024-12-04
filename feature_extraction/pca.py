from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



def apply_pca(data, n_components=None):
    if n_components:
        pca, principal_components = _apply_pca(data, n_components)
    else:
        threshold = 0.90  # For 90% explained variance
        n_components = 1
        reached = 0

        while reached < threshold:
            n_components += 1
            pca, principal_components = _apply_pca(data, n_components)
            reached = sum(pca.explained_variance_ratio_)

    print("New shape:", principal_components.shape)
    print("Cumilative variance ratio:", np.cumsum(pca.explained_variance_ratio_))
    return principal_components

def _apply_pca(data, n_components):
    df = pd.DataFrame(data).T
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    principal_df = pd.DataFrame(data=principal_components).T
    principal_components = principal_df.to_numpy()
    
    return pca, principal_components

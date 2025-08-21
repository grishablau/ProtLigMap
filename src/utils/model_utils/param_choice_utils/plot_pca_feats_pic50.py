import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_input_feats_pca_colored_by_pic50(input_feats, pic50, title="PCA of Input Features colored by pIC50"):
    """
    input_feats: torch.Tensor or np.array of shape (n_points, n_features)
    pic50: torch.Tensor or np.array of shape (n_points,)
    """
    # Convert to numpy if torch.Tensor
    if 'torch' in str(type(input_feats)):
        input_feats = input_feats.cpu().numpy()
    if 'torch' in str(type(pic50)):
        pic50 = pic50.cpu().numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    input_2d = pca.fit_transform(input_feats)

    # Plot
    plt.figure(figsize=(8,6))
    sc = plt.scatter(input_2d[:,0], input_2d[:,1], c=pic50, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(sc, label='pIC50')
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(title)
    plt.tight_layout()
    plt.show()

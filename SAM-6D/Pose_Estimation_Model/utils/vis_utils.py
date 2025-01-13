import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from sklearn.decomposition import PCA

def features_to_colors(features):
    """
    Convert high-dimensional feature matrix to RGB colors, considering cosine similarity.
    
    Args:
        features (numpy.ndarray): Feature matrix of shape (N, D).
        
    Returns:
        colors (numpy.ndarray): RGB colors of shape (N, 3).
    """
    # Normalize features to unit length for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6  # Avoid division by zero
    features = features / norms  # Normalize to unit length
    
    # Reduce to 3 dimensions if necessary
    if features.shape[1] > 3:
        pca = PCA(n_components=3)
        features = pca.fit_transform(features)
    
    # Normalize reduced features to [0, 1]
    features = (features - features.min(axis=0)) / (features.ptp(axis=0) + 1e-6)
    
    # Map directly to RGB
    colors = features
    return colors

def visualize_points_3d(tem_pts, points_name, num_frames=360, **kwargs):
    output_video_path=f'{points_name}_visualization.mp4'
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of points
    ax.scatter(tem_pts[:, 0], tem_pts[:, 1], tem_pts[:, 2], **kwargs)

    # Hide grid and axes
    ax.grid(False)
    ax.axis('off')

    # Configure axes limits for better visibility
    max_extent = np.max(np.abs(tem_pts))
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Rotate and save each frame
    from matplotlib.animation import FuncAnimation

    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save as a video
    anim.save(output_video_path, fps=30, writer='ffmpeg')
    print(f"Visualization saved to {output_video_path}")
    plt.close()
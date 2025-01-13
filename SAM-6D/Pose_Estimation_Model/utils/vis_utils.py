import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

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

    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,

    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save as a video
    anim.save(output_video_path, fps=30, writer='ffmpeg')
    print(f"Visualization saved to {output_video_path}")
    plt.close()



def visualize_points_3d_two_sets(pts1, pts2, points_name, 
                                 color1='r', color2='g', 
                                 num_frames=360, **kwargs):
    """
    Create a 3D rotating scatter plot for two sets of points.
    
    Parameters:
    -----------
    pts1 : np.ndarray
        First set of 3D points with shape (N, 3).
    pts2 : np.ndarray
        Second set of 3D points with shape (M, 3).
    points_name : str
        Name (or prefix) for the output video file.
    color1 : str, optional
        Color for the first set of points (default 'r').
    color2 : str, optional
        Color for the second set of points (default 'b').
    num_frames : int, optional
        Number of frames to rotate through (default 360).
    **kwargs : dict
        Additional keyword arguments passed to `plt.scatter()`.
    """
    output_video_path = f'{points_name}_visualization.mp4'
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for both sets of points
    ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], c=color1, **kwargs)
    ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c=color2, **kwargs)

    # Hide grid and axes
    ax.grid(False)
    ax.axis('off')

    # Determine axis limits from both sets of points for consistency
    all_points = np.vstack([pts1, pts2])
    max_extent = np.max(np.abs(all_points))

    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Define update function for rotation
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save animation as video
    anim.save(output_video_path, fps=30, writer='ffmpeg')
    print(f"Visualization saved to {output_video_path}")
    plt.close()
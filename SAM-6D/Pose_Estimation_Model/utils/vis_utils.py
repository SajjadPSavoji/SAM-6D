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


def visualize_two_sets_3d(
    points1, 
    points2, 
    vis_name,
    points1_name='Set 1', 
    points2_name='Set 2', 
    color1='red', 
    color2='green', 
    num_frames=360, 
    output_video_path='two_sets_visualization.mp4',
    **kwargs,
):
    """
    Visualize two sets of 3D points in a single animated plot.

    Parameters:
    - points1 (np.ndarray): First set of points, shape (N, 3).
    - points2 (np.ndarray): Second set of points, shape (M, 3).
    - points1_name (str): Label for the first point set.
    - points2_name (str): Label for the second point set.
    - color1 (str): Color for the first point set.
    - color2 (str): Color for the second point set.
    - num_frames (int): Number of frames in the animation.
    - output_video_path (str): Path to save the output video.
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plots for both point sets
    scatter1 = ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], 
                          c=color1, **kwargs)
    scatter2 = ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], 
                          c=color2, **kwargs)

    # Hide grid and axes for a cleaner look
    ax.grid(False)
    ax.axis('off')

    # Determine the combined maximum extent for all points
    all_points = np.vstack((points1, points2))
    max_extent = np.max(np.abs(all_points)) * 1.1  # Slightly larger for padding
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Function to update the view for each frame
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    # Save the animation as a video file
    anim.save(f"{vis_name}_{output_video_path}", fps=30, writer='ffmpeg')
    plt.close(fig)  # Close the figure to free memory
    print(f"Visualization saved to {vis_name}_{output_video_path}")
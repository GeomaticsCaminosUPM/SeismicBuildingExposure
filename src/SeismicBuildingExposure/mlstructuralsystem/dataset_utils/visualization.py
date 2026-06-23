"""
Plotting helpers for inspecting model inputs and outputs.

Not used by the production pipeline (preprocess/load/save_test); intended
for ad-hoc inspection in notebooks or exploratory scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import distinctipy


def plot_2D_cluster(
        X_2D: np.ndarray,
        y: np.ndarray,
        label_encoder: LabelEncoder,
        title: str = "2D visualization of processed features",
        xlabel: str = "Component 1",
        ylabel: str = "Component 2"
    ) -> None:
    """
    Creates a 2D scatter plot to visualize clustered or dimensionally-reduced data.

    Each point is colored according to its class label, which is useful for assessing
    class separability after dimensionality reduction techniques like PCA or UMAP.

    Args:
        X_2D (np.ndarray): Data with two dimensions, shape (n_samples, 2).
        y (np.ndarray): Integer-encoded labels for each sample.
        label_encoder (LabelEncoder): The fitted encoder used to map integer labels
                                      back to class names for the legend.
        title (str, optional): The title of the plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
    """
    plt.figure(figsize=(10, 8))
    # Define a color palette for the classes

    colors = colors = distinctipy.get_colors(len(label_encoder.classes_))

    # Plot points for each class separately to create a legend
    for i, class_name in enumerate(label_encoder.classes_):
        # Find the indices for the current class
        idx = (y == i)
        plt.scatter(X_2D[idx, 0], X_2D[idx, 1], color=colors[i], label=f'{class_name}', alpha=0.7, s=20)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, normalize: bool = False) -> None:
    """
    Plots a confusion matrix for evaluating classification performance.

    Uses a seaborn heatmap for a clear visual representation. The matrix can be
    normalized to show recall rates for each class.

    Args:
        y_pred (np.ndarray): The predicted labels from a model.
        y_true (np.ndarray): The ground truth labels.
        normalize (bool): If True, the confusion matrix is normalized by the number
                          of true instances for each class (row-wise normalization).
                          Defaults to False.
    """
    # Automatically infer the sorted list of class names from the data
    class_names = np.unique(np.concatenate([y_true, y_pred]))

    # Compute the confusion matrix using scikit-learn
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Format for printing (raw counts or normalized floats)
    fmt = 'd'
    title = 'Confusion Matrix (Counts)'

    if normalize:
        # Normalize each row (true class) so it sums to 1
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Confusion Matrix'

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.tight_layout()
    plt.show()
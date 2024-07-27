# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:12:36 2024

@author: Salma
"""

import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from skimage.feature import hog
from skimage import color, io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Function to load images from a folder and preprocess them
def load_images_and_labels_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []
    for label, subfolder in enumerate(['dogs', 'cats']):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(subfolder_path, filename)
                img = io.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize the image
                    images.append(img)
                    labels.append(label)  # 0 for dogs, 1 for cats
    return np.array(images), np.array(labels)

# Extract HOG features from images
def extract_hog_features(images, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False):
    hog_features = []
    for img in images:
        # Convert to grayscale
        img_gray = color.rgb2gray(img)
        if visualize:
            features, _ = hog(img_gray, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, visualize=visualize)
        else:
            features = hog(img_gray, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, visualize=visualize)
        hog_features.append(features)
    return np.array(hog_features)

# Load training and testing data
train_folder = '../training_set'
test_folder = '../test_set'

train_images, train_labels = load_images_and_labels_from_folder(train_folder)
test_images, test_labels = load_images_and_labels_from_folder(test_folder)

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing images shape: {test_images.shape}")
print(f"Testing labels shape: {test_labels.shape}")

# Extract HOG features
train_hog_features = extract_hog_features(train_images)
test_hog_features = extract_hog_features(test_images)

# Standardize the HOG features
scaler_standard = StandardScaler()
train_hog_features_standard_scaled = scaler_standard.fit_transform(train_hog_features)
test_hog_features_standard_scaled = scaler_standard.transform(test_hog_features)

# Optionally, apply MinMaxScaler
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
train_hog_features_minmax_scaled = scaler_minmax.fit_transform(train_hog_features)
test_hog_features_minmax_scaled = scaler_minmax.transform(test_hog_features)

# Apply PCA for feature extraction
pca = PCA(n_components=50)
train_hog_features_pca = pca.fit_transform(train_hog_features_standard_scaled)
test_hog_features_pca = pca.transform(test_hog_features_standard_scaled)

print(f"Training PCA shape: {train_hog_features_pca.shape}")
print(f"Testing PCA shape: {test_hog_features_pca.shape}")

# Determine the optimal number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
    kmeans.fit(train_hog_features_pca)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()

# Choose the optimal number of clusters based on the elbow graph
optimal_clusters = 2  # Set this to the number of clusters identified by the elbow method

# Apply KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10, random_state=0)
train_clusters = kmeans.fit_predict(train_hog_features_pca)
test_clusters = kmeans.predict(test_hog_features_pca)

# Print evaluation metrics
print("Ground truth clusters:\n", test_labels)
print("Estimated clusters:\n", test_clusters)
print("Confusion matrix:\n", confusion_matrix(test_labels, test_clusters))
print("Classification report:\n", classification_report(test_labels, test_clusters))

# Relabel clusters
relabel_clusters = [1 if cluster == 0 else 0 for cluster in test_clusters]

print("Relabel clusters:\n", relabel_clusters)
print("Confusion matrix (relabelled):\n", confusion_matrix(test_labels, relabel_clusters))
print("Classification report (relabelled):\n", classification_report(test_labels, relabel_clusters))

# Function to plot PCA components
def plot_pca_components(images_pca, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(images_pca[:, 0], images_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

# Plot PCA components of training data
plot_pca_components(train_hog_features_pca, train_clusters, 'KMeans Clustering Results (Training Data)')

# Function to plot images with cluster labels
def plot_images(images, labels, title, image_size=(64, 64)):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape(image_size + (3,)))
            ax.set_title(f'Cluster {labels[i]}')
        ax.axis('off')
    plt.show()

# Plot images with predicted clusters
plot_images(test_images, test_clusters, 'KMeans Clustering Results (Test Data)')

# Perform PCA to reduce dimensionality to 2D for visualization
pca_2d = PCA(n_components=2)
pca_result = pca_2d.fit_transform(test_hog_features)  # Use HOG features for PCA visualization

# Visualize the clustering results
colormap = np.array(['green', 'red'])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Classification K-means')
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colormap[relabel_clusters], label='Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Ground Truth')
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colormap[test_labels], label='Ground Truth')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.show()

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set image parameters
img_height = 128
img_width = 128
channels = 3
num_classes = 2  # Adjusted to match the number of classes in lab dictionary

# Dictionary to label the images
lab = {'dogs': 0, 'cats': 1}

def load_images_from_folder(folder_path, label_dict):
    data = []
    labels = []
    print(f"Loading images from folder: {folder_path}")
    for folder in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder)
        if not os.path.isdir(folder_path_full):
            print(f"Skipping {folder_path_full} as it is not a directory.")
            continue
        print(f"Processing folder: {folder}")
        for filename in os.listdir(folder_path_full):
            file_path = os.path.join(folder_path_full, filename)
            if os.path.isfile(file_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        img = load_img(file_path, target_size=(img_height, img_width))
                        img_array = img_to_array(img)
                        data.append(img_array.flatten())  # Flatten the image array
                        labels.append(label_dict[folder])
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
                else:
                    print(f"Skipping non-image file {file_path}.")
    print(f"Loaded {len(data)} images from {folder_path}")
    return np.array(data) / 255.0, np.array(labels)

# Load training data
dataset_path1 = 'Data/Train'
x_train, y_train = load_images_from_folder(dataset_path1, lab)

# Load testing data
dataset_path2 = 'Data/Test'
x_test, y_test = load_images_from_folder(dataset_path2, lab)

# Check if data is loaded correctly
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Testing labels shape: {y_test.shape}")

# Ensure that the data is not empty
if x_train.size == 0 or y_train.size == 0 or x_test.size == 0 or y_test.size == 0:
    raise ValueError("One of the data arrays is empty. Please check your data loading process.")

# Debug output
print(f"Sample image data shape: {x_train[0].shape}")
print(f"Sample image label: {y_train[0]}")

# Define and train Decision Tree Classifier
dt_model = DecisionTreeClassifier()  # You can adjust parameters as needed
dt_model.fit(x_train, y_train)

# Predict and evaluate Decision Tree Classifier
y_pred_dt = dt_model.predict(x_test)

print("----Decision Tree Classifier evaluation:-------")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("Classification report:")
print(classification_report(y_test, y_pred_dt))


# Define and train K-Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust `n_neighbors`
knn_model.fit(x_train, y_train)

# Predict and evaluate K-Neighbors Classifier
y_pred_knn = knn_model.predict(x_test)
print("----K-Neighbors Classifier evaluation:-------")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification report:")
print(classification_report(y_test, y_pred_knn))

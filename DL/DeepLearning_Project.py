import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
img_height = 128
img_width = 128
num_classes = 2
lab = {'cats': 0, 'dogs': 1}

# Load training data
train_data = []
train_labels = []
train_base_path = 'C:/Users/t460/Downloads/archive/training_set/training_set'

for folder in os.listdir(train_base_path):
    folder_path = os.path.join(train_base_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                train_data.append(img_array)
                train_labels.append(lab[folder])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

x_train = np.array(train_data) / 255.0  # Normalize pixel values
y_train = np.array(train_labels)

# Load test data
test_data = []
test_labels = []
test_base_path = 'C:/Users/t460/Downloads/archive/test_set/test_set'

for folder in os.listdir(test_base_path):
    folder_path = os.path.join(test_base_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                test_data.append(img_array)
                test_labels.append(lab[folder])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

x_test = np.array(test_data) / 255.0  # Normalize pixel values
y_test = np.array(test_labels)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")



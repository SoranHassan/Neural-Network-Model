import logging
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logging.info("Start Processing")

# Define the characters present in the captcha, including the padding character '_'
characters = '345689ABCDEFHJKMNPRSTUVWXY_'
char_to_num = {char: idx for idx, char in enumerate(characters)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Function to remove noise from the image
def remove_noise(image):
    return cv2.medianBlur(image, 3)

# Path to captcha images
captcha_dir = './dataset'

# Lists to store images and labels
images = []
labels = []

# Read images and extract labels from file names
for filename in os.listdir(captcha_dir):
    if filename.endswith('.jpg'):
        label = filename.split('_')[1].split('.')[0]
        image = cv2.imread(os.path.join(captcha_dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))  # Changed dimensions
        image = remove_noise(image)
        image = image / 255.0
        images.append(image)
        
        # One-hot encode the label
        label_encoded = [char_to_num[char] for char in label.ljust(6, '_')]  # Pad with '_' to length 6
        labels.append(label_encoded)

# Convert lists to numpy arrays
images = np.array(images).reshape(-1, 32, 128, 1)  # Changed dimensions
labels = np.array(labels)

# One-hot encode the labels
labels = np.array([tf.keras.utils.to_categorical(label, num_classes=len(characters)) for label in labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the neural network model
input_shape = (32, 128, 1)  # Changed dimensions
inputs = tf.keras.layers.Input(shape=input_shape)

x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)  # Added Batch Normalization
x = tf.keras.layers.Dropout(0.2)(x)  # Adjusted Dropout
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)  # Added Batch Normalization
x = tf.keras.layers.Dropout(0.2)(x)  # Adjusted Dropout
x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)  # Added Convolutional layer
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)  # Added Batch Normalization
x = tf.keras.layers.Dropout(0.2)(x)  # Adjusted Dropout
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)  # Increased neurons
x = tf.keras.layers.BatchNormalization()(x)  # Added Batch Normalization
x = tf.keras.layers.Dropout(0.2)(x)  # Adjusted Dropout

outputs = [tf.keras.layers.Dense(len(characters), activation='softmax')(x) for _ in range(6)]

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

metrics = ['accuracy'] * 6  # Create a list of 'accuracy' metrics for each output
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=metrics)

# Split the labels into 6 separate outputs
y_train_split = [y_train[:, i, :] for i in range(6)]
y_test_split = [y_test[:, i, :] for i in range(6)]

# Train the model
model.fit(X_train, y_train_split, epochs=50, validation_data=(X_test, y_test_split))

# Save the model
model.save('captcha_model.keras')

# Evaluate the model on the test set
test_loss, *test_acc = model.evaluate(X_test, y_test_split)

# Calculate overall accuracy
correct_predictions = 0
total_predictions = len(X_test)

predictions = model.predict(X_test)
predicted_labels = np.array([np.argmax(pred, axis=-1) for pred in predictions]).T

for i in range(total_predictions):
    true_label = ''.join([num_to_char[np.argmax(y_test_split[j][i])] for j in range(6)]).replace('_', '')
    predicted_label = ''.join([num_to_char[pred] for pred in predicted_labels[i]]).replace('_', '')
    if true_label == predicted_label:
        correct_predictions += 1
        # Mark the correctly predicted characters on the image
        image = (X_test[i] * 255).astype(np.uint8).reshape(32, 128)  # Changed dimensions
        plt.imshow(image, cmap='gray')
        for j, char in enumerate(predicted_label):
            plt.text(j * 20, 30, char, color='red', fontsize=12)  # Adjusted text positioning
        plt.savefig(f'./predicted/correct_prediction_{i}.png')
        plt.close()

overall_accuracy = correct_predictions / total_predictions

print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')
print(f'Correct Predictions: {correct_predictions}/{total_predictions}')

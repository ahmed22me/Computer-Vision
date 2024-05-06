import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from traitlets import validate
import sklearn.model_selection
import numpy as np
import tensorflow as tf
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image


start_path = "E:/cv/test"

image_paths = []
labels = []

for root, dirs, files in os.walk(start_path):
    for dir_name in dirs:
        label = int(dir_name) - 1
        dir_path = os.path.join(root, dir_name)
        temp_images = os.listdir(dir_path)

        # Append image paths and labels
        image_paths.extend([os.path.join(dir_path, img) for img in temp_images])
        labels.extend([label] * len(temp_images))

# Create DataFrame
test_df = pd.DataFrame({
    'images': image_paths,
    'labels': labels
})

test_df["mat_images"] = test_df['images'].apply(lambda x: cv2.resize(cv2.imread(x), (224, 224)))

test_df.head()

print(len(test_df["mat_images"]))

first_image = test_df["mat_images"].iloc[0]
plt.imshow(first_image, cmap='gray')
plt.title("First Image")
plt.show()

test_df=test_df.sample(frac=1, random_state=2).reset_index(drop=True)

x_test=test_df["mat_images"]/255
y_test=test_df["labels"]



y_test=to_categorical(y_test,20)

x_test = np.array([np.array(x, dtype=np.float32) for x in x_test], dtype=object)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

y_test = np.array(y_test)

# Decode one-hot encoded labels
y_test = np.argmax(y_test, axis=1)

model = load_model('E:/cv/deep learning model/vision94e.h5')
predictions = model.predict(x_test)
print(predictions)
true_labels = y_test
predicted_labels = np.argmax(predictions, axis=1)




print("True Labels:", true_labels)
print("Predicted Labels:", predicted_labels)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Accuracy: {accuracy * 100:.2f}%")

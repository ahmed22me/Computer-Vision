import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images(folder_path):
    images = []
    labels = []
    for class_label in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img = img.convert('RGB')  # Ensure RGB format
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                images.append(img_array)
                labels.append(class_label)
    return np.vstack(images), np.array(labels)

def test_deep_learning(model_path, test_folder):

    model = load_model(model_path)

    X_test, y_test = load_images(test_folder)

    X_test = X_test / 255.0

    y_pred = model.predict(X_test)

    y_pred_labels = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")


test_deep_learning('E:/cv/deep learning model/vision94e.h5', 'E:/cv/test')
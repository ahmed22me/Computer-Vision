# import os
# import cv2
# import numpy as np
# import random
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
#
# # Set paths for training and validation data
# training_dir = '/kaggle/input/reco-part2/Product Recoginition/Training Data'
# validation_dir = '/kaggle/input/reco-part2/Product Recoginition/Validation Data'
#
# from tensorflow.keras.preprocessing import image as keras_image
#
# def read_image(index, base_dir, target_size=(224, 224)):
#     path = os.path.join(base_dir, index[0], index[1])
#
#     # Print the generated path for debugging
#     print(f"Loading image: {path}")
#
#     try:
#         # Use Keras' image.load_img to read the image
#         img = keras_image.load_img(path, target_size=target_size)
#         img_array = keras_image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         return img_array
#     except Exception as e:
#         print(f"Error loading image: {path} - {e}")
#         return None
#
#
#
# def get_batch(triplet_list, base_dir, batch_size=256, preprocess=True):
#     batch_steps = len(triplet_list) // batch_size
#
#     for i in range(batch_steps + 1):
#         anchor = []
#         positive = []
#         negative = []
#
#         j = i * batch_size
#         while j < (i + 1) * batch_size and j < len(triplet_list):
#             a, p, n = triplet_list[j]
#             anchor_img = read_image(a, base_dir)
#             positive_img = read_image(p, base_dir)
#             negative_img = read_image(n, base_dir)
#
#             if anchor_img is not None and positive_img is not None and negative_img is not None:
#                 anchor.append(anchor_img)
#                 positive.append(positive_img)
#                 negative.append(negative_img)
#
#             j += 1
#
#         anchor = np.array(anchor)
#         positive = np.array(positive)
#         negative = np.array(negative)
#
#         if preprocess:
#             anchor = preprocess_input(anchor)
#             positive = preprocess_input(positive)
#             negative = preprocess_input(negative)
#
#         yield ([anchor, positive, negative])
#
# def create_triplets(directory):
#     triplets = []
#
#     # Iterate through each class directory
#     for class_folder in os.listdir(directory):
#         class_path = os.path.join(directory, class_folder)
#
#         if os.path.isdir(class_path):
#             # Get a list of image filenames in the current class
#             image_filenames = os.listdir(class_path)
#
#             # Create triplets for each anchor image in the current class
#             for anchor_img_filename in image_filenames:
#                 anchor_img_path = os.path.join(class_path, anchor_img_filename)
#
#                 # Randomly select a positive example from the same class
#                 positive_img_filename = random.choice(image_filenames)
#                 positive_img_path = os.path.join(class_path, positive_img_filename)
#
#                 # Randomly select a negative example from a different class
#                 other_classes = [c for c in os.listdir(directory) if c != class_folder]
#                 negative_class = random.choice(other_classes)
#                 negative_img_filename = random.choice(os.listdir(os.path.join(directory, negative_class)))
#                 negative_img_path = os.path.join(directory, negative_class, negative_img_filename)
#
#                 # Check if all images exist before adding the triplet
#                 if os.path.exists(anchor_img_path) and os.path.exists(positive_img_path) and os.path.exists(negative_img_path):
#                     triplet = (anchor_img_path, positive_img_path, negative_img_path)
#                     triplets.append(triplet)
#
#     return triplets
#
# def create_validation_triplets(directory):
#     triplets = []
#
#     # Iterate through each class directory
#     for class_folder in os.listdir(directory):
#         class_path = os.path.join(directory, class_folder)
#
#         if os.path.isdir(class_path):
#             # Get a list of image filenames in the current class
#             image_filenames = os.listdir(class_path)
#
#             # Create triplets for each anchor image in the current class
#             for anchor_img_filename in image_filenames:
#                 anchor_img_path = os.path.join(class_path, anchor_img_filename)
#
#                 # Randomly select a positive example from the same class
#                 positive_img_filename = random.choice(image_filenames)
#                 positive_img_path = os.path.join(class_path, positive_img_filename)
#
#                 # Randomly select a negative example from a different class
#                 other_classes = [c for c in os.listdir(directory) if c != class_folder]
#                 negative_class = random.choice(other_classes)
#                 negative_img_filename = random.choice(os.listdir(os.path.join(directory, negative_class)))
#                 negative_img_path = os.path.join(directory, negative_class, negative_img_filename)
#
#                 # Check if all images exist before adding the triplet
#                 if os.path.exists(anchor_img_path) and os.path.exists(positive_img_path) and os.path.exists(negative_img_path):
#                     triplet = (anchor_img_path, positive_img_path, negative_img_path)
#                     triplets.append(triplet)
#
#     return triplets
#
# # Generate triplets for training data
# train_triplet = create_triplets(training_dir)
#
# # Training
# num_plots = 3
# f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))
#
# for x in get_batch(train_triplet, training_dir, batch_size=num_plots, preprocess=False):
#     a, p, n = x
#     for i in range(num_plots):
#         axes[i, 0].imshow(a[i])
#         axes[i, 1].imshow(p[i])
#         axes[i, 2].imshow(n[i])
#         i += 1
#     break
#
# # Generate triplets for validation data
# validation_triplet = create_validation_triplets(validation_dir)
#
# # Validation
# correct_predictions = 0
# total_predictions = 0
#
# for x in get_batch(validation_triplet, validation_dir, batch_size=1, preprocess=True):
#     a, p, n = x
#
#     similarities = {}
#     for train_product_folder in training_images:
#         for train_img_array in training_images[train_product_folder]:
#             similarity_score = cosine_similarity(a.reshape(1, -1), train_img_array.reshape(1, -1))
#             similarities[train_product_folder] = similarity_score[0][0]
#
#     most_similar_product = max(similarities, key=similarities.get)
#     print(f'Most similar product: {most_similar_product}')


# -*- coding: utf-8 -*-
"""prefinal-cv2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uTdBW0cZtkBu-Aahl--qCdzpHawvrO58
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import cv2
import time
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

import seaborn as sns
import matplotlib.pyplot as plt

import os

def count_images_in_folders(directory):
    folder_counts = {}

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(folder_path):
            # Count the number of files in the folder
            num_files = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
            folder_counts[folder_name] = num_files

    return folder_counts

train_dir = "E:/cv/archive/Product Recoginition/Training Data"
val_dir = "E:/cv/archive/Product Recoginition/Validation Data"

train_dict = count_images_in_folders(train_dir)
val_dict = count_images_in_folders(val_dir)

print("Train Dictionary:",train_dict)

print("Validation Dictionary:",val_dict)

"""# triples"""

import os
import random

def create_triplets(directory, folder_list, max_files=10, file_extension=".png"):
    triplets = []
    folders = list(folder_list.keys())

    for folder in folders:
        path = os.path.join(directory, folder)
        files = [f for f in os.listdir(path) if f.endswith(file_extension)][:max_files]
        num_files = len(files)

        for i in range(num_files-1):
            for j in range(i+1, num_files):
                anchor = (folder, f"{files[i]}")
                positive = (folder, f"{files[j]}")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_files = [f for f in os.listdir(os.path.join(directory, neg_folder)) if f.endswith(file_extension)]
                neg_file = random.choice(neg_files)
                negative = (neg_folder, f"{neg_file}")

                triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets

# Assuming train_dir, train_dict, val_dir, and val_dict are defined
train_triplet = create_triplets(train_dir, train_dict)
test_triplet = create_triplets(val_dir, val_dict)

print("Number of training triplets:", len(train_triplet))
print("Number of testing triplets:", len(test_triplet))

print("\nExamples of triplets:")
for i in range(5):
    print(train_triplet[i])

"""def read_image2(index):
    path = os.path.join(train_dir, index[0], "web"+str(int(index[1][0])+1)
#                         + index[1][1:]
                       +".png")

# batch generator
"""

################for train
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt


ROOT    = "E:/cv/archive/Product Recoginition/Training Data"
#val_dir = "/kaggle/input/reco-part2/Product Recoginition/Validation Data"


def read_image(index , target_size=(224, 224)):
    path  = os.path.join(ROOT, index[0], index[1])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)  # Resize the image
    return image

def get_batch(triplet_list, batch_size=256, preprocess=True):
    batch_steps = len(triplet_list)//batch_size

    for i in range(batch_steps + 1):
        anchor   = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor_img   = read_image(a)
            positive_img = read_image(p)
            negative_img = read_image(n)

            if anchor_img is not None and positive_img is not None and negative_img is not None:
                anchor.append(anchor_img)
                positive.append(positive_img)
                negative.append(negative_img)

            j += 1

        anchor   = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor   = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])

num_plots = 3

f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))

for x in get_batch(train_triplet, batch_size=num_plots, preprocess=False):
    a, p, n = x
    for i in range(num_plots):
        axes[i, 0].imshow(a[i])
        axes[i, 1].imshow(p[i])
        axes[i, 2].imshow(n[i])
        i += 1
    break

################for test

import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt


root = "E:/cv/archive/Product Recoginition/Validation Data"


def read_image1(index , target_size=(224, 224)):
    path  = os.path.join(root, index[0], index[1])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)  # Resize the image
    return image

def get_batch_test(triplet_list, batch_size=256, preprocess=True):
    batch_steps = len(triplet_list)//batch_size

    for i in range(batch_steps + 1):
        anchor   = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            anchor_img   = read_image1(a)
            positive_img = read_image1(p)
            negative_img = read_image1(n)

            if anchor_img is not None and positive_img is not None and negative_img is not None:
                anchor.append(anchor_img)
                positive.append(positive_img)
                negative.append(negative_img)

            j += 1

        anchor   = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            anchor   = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

        yield ([anchor, positive, negative])

num_plots = 3

f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))

for x in get_batch_test(test_triplet, batch_size=num_plots, preprocess=False):
    a, p, n = x
    for i in range(num_plots):
        axes[i, 0].imshow(a[i])
        axes[i, 1].imshow(p[i])
        axes[i, 2].imshow(n[i])
        i += 1
    break

"""# #create model"""

from tensorflow.keras import backend, layers, metrics

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )

    for i in range(len(pretrained_model.layers)-27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model

class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_siamese_network(input_shape = (224, 224, 3)):
    encoder = get_encoder(input_shape)

    # Input Layers for the images
    anchor_input   = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")

    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)

    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )

    # Creating the Model
    siamese_network = Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
        name = "Siamese_Network"
    )
    return siamese_network

siamese_network = get_siamese_network()
siamese_network.summary()

plot_model(siamese_network, show_shapes=True, show_layer_names=True)

class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]

siamese_model = SiameseModel(siamese_network)

optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)

def test_on_triplets(batch_size = 256):
    pos_scores, neg_scores = [], []

    for data in get_batch_test(test_triplet, batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on test = {accuracy:.5f}")
    return (accuracy, ap_mean, an_mean, ap_stds, an_stds)

import warnings

# Suppress libpng warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.PngImagePlugin")

"""save_all = False
epochs = 30
batch_size = 128

max_acc = 0
train_loss = []
test_metrics = []

for epoch in range(1, epochs+1):
    t = time.time()

    # Training the model on train data
    epoch_loss = []

    for data in get_batch(train_triplet, batch_size=batch_size):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    train_loss.append(epoch_loss)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time()-t)} sec)")
    print(f"Loss on train    = {epoch_loss:.5f}")

    # Testing the model on test data
    metric = test_on_triplets(batch_size=batch_size)
    test_metrics.append(metric)
    accuracy = metric[0]

    # Saving the model weights
    if save_all or accuracy>=max_acc:
        siamese_model.save_weights("siamese_model")
        max_acc = accuracy

# Saving the model after all epochs run
siamese_model.save_weights("siamese_model-final")"""

save_all = False
epochs = 3
batch_size = 128

max_acc = 0
train_loss = []
test_metrics = []

start_time_training = time.time()

for epoch in range(1, epochs+1):
    t = time.time()

    # Training the model on train data
    epoch_loss = []

    for data in get_batch(train_triplet, batch_size=batch_size):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    train_loss.append(epoch_loss)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time()-t)} sec)")
    print(f"Loss on train    = {epoch_loss:.5f}")

    # Testing the model on test data
    start_time_testing = time.time()

    metric = test_on_triplets(batch_size=batch_size)
    test_metrics.append(metric)
    accuracy = metric[0]

    end_time_testing = time.time()
    total_time_testing = end_time_testing - start_time_testing

    # Saving the model weights
    if save_all or accuracy>=max_acc:
        siamese_model.save_weights("siamese_model_weights.h5")
        max_acc = accuracy


end_time_training = time.time()
total_time_training = end_time_training - start_time_training


# Saving the model after all epochs run
siamese_model.save("siamese_model_final", save_format="tf")


print(f"\nTotal Training Time: {total_time_training} seconds")

print(f"Total Testing Time: {total_time_testing} seconds")

def plot_metrics(loss, metrics):
    # Extracting individual metrics from metrics
    accuracy = metrics[:, 0]
    ap_mean  = metrics[:, 1]
    an_mean  = metrics[:, 2]
    ap_stds  = metrics[:, 3]
    an_stds  = metrics[:, 4]

    plt.figure(figsize=(15,5))

    # Plotting the loss over epochs
    plt.subplot(121)
    plt.plot(loss, 'b', label='Loss')
    plt.title('Training loss')
    plt.legend()

    # Plotting the accuracy over epochs
    plt.subplot(122)
    plt.plot(accuracy, 'r', label='Accuracy')
    plt.title('Testing Accuracy')
    plt.legend()

    plt.figure(figsize=(15,5))

    # Comparing the Means over epochs
    plt.subplot(121)
    plt.plot(ap_mean, 'b', label='AP Mean')
    plt.plot(an_mean, 'g', label='AN Mean')
    plt.title('Means Comparision')
    plt.legend()

    # Plotting the accuracy
    ap_75quartile = (ap_mean+ap_stds)
    an_75quartile = (an_mean-an_stds)
    plt.subplot(122)
    plt.plot(ap_75quartile, 'b', label='AP (Mean+SD)')
    plt.plot(an_75quartile, 'g', label='AN (Mean-SD)')
    plt.title('75th Quartile Comparision')
    plt.legend()

test_metrics = np.array(test_metrics)
plot_metrics(train_loss, test_metrics)

def extract_encoder(model):
    encoder = get_encoder((224, 224, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

encoder = extract_encoder(siamese_model)
encoder.save_weights("encoder")
encoder.summary()

def classify_images(face_list1, face_list2, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(face_list1)
    tensor2 = encoder.predict(face_list2)

    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    prediction = np.where(distance<=threshold, 0, 1)
    return prediction

def ModelMetrics(pos_list, neg_list):
    true = np.array([0]*len(pos_list)+[1]*len(neg_list))
    pred = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, pred)}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, pred)

    categories  = ['Similar','Different']
    names = ['True Similar','False Similar', 'False Different','True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


pos_list = np.array([])
neg_list = np.array([])

for data in get_batch_test(test_triplet, batch_size=256):
    a, p, n = data
    pos_list = np.append(pos_list, classify_images(a, p))
    neg_list = np.append(neg_list, classify_images(a, n))
    break

ModelMetrics(pos_list, neg_list)

model = tf.keras.models.load_model("siamese_model_final")

model

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity

# # model = load_model('E:/cv/siames_30/siamese_model_weights.h5')

# # model = tf.keras.models.load_model("siamese_model_final")



# test_dir = '/kaggle/input/teesss/test_reco'
# reference_dir = '/kaggle/input/reco-part2/Product Recoginition/Training Data'
# reference_images = {}

# correct_predictions = 0
# total_predictions = 0
# class_count = 0

# for class_dir in os.listdir(reference_dir):
#     class_path = os.path.join(reference_dir, class_dir)
#     if os.path.isdir(class_path):
#         class_count += 1

#         for img_filename in os.listdir(class_path):
#             reference_image_path = os.path.join(class_path, img_filename)
#             reference_img = image.load_img(reference_image_path, target_size=(224, 224))
#             reference_img_array = image.img_to_array(reference_img)
#             reference_img_array = np.expand_dims(reference_img_array, axis=0)
#             reference_img_array = preprocess_input(reference_img_array)
#             reference_images[img_filename] = (class_dir, reference_img_array)
# print(f'Total number of classes in {reference_dir}: {class_count}')


# class_accuracies = {}

# for class_name in os.listdir(test_dir):
#     class_path = os.path.join(test_dir, class_name)

#     if os.path.isdir(class_path):
#         class_correct_predictions = 0
#         class_total_predictions = 0

#         for filename in os.listdir(class_path):
#             img_path = os.path.join(class_path, filename)

#             if os.path.exists(img_path):
#                 test_img = image.load_img(img_path, target_size=(224, 224))
#                 test_img_array = image.img_to_array(test_img)
#                 test_img_array = np.expand_dims(test_img_array, axis=0)
#                 test_img_array = preprocess_input(test_img_array)

#                 similarities = {}
#                 for ref_filename, (ref_class, reference_img_array) in reference_images.items():
#                     similarity_score = cosine_similarity(test_img_array.reshape(1, -1),
#                                                           reference_img_array.reshape(1, -1))
#                     similarities[ref_class] = (ref_class, similarity_score[0][0])

#                 most_similar_class, _ = max(similarities.items(), key=lambda x: x[1][1])
#                 predicted_class = similarities[most_similar_class][0]

#                 print(
#                     f'Test image {filename} in class {class_name} belongs to class {predicted_class} (Similarity: {similarities[most_similar_class][1]:.4f})')

#                 class_total_predictions += 1
#                 if predicted_class == class_name:
#                     class_correct_predictions += 1

#                 total_predictions += 1
#                 if predicted_class == class_name:
#                     correct_predictions += 1

#             else:
#                 print(f'Test image {filename} in class {class_name} not found.')

#         class_accuracy = class_correct_predictions / class_total_predictions if class_total_predictions > 0 else 0
#         class_accuracies[class_name] = class_accuracy
#         print(f'Class {class_name} Accuracy: {class_accuracy:.2%} ({class_correct_predictions}/{class_total_predictions} correct predictions)')
#         print()


# accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
# print(f'Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)')


# print('\nClass-wise Accuracies:')
# for class_name, class_accuracy in class_accuracies.items():
#     print(f'Class {class_name}: {class_accuracy:.2%}')

# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity


# # Set paths for training and validation data
# training_dir = '/kaggle/input/reco-part2/Product Recoginition/Training Data'
# validation_dir = '/kaggle/input/reco-part2/Product Recoginition/Validation Data'

# # Load training images
# training_images = {}
# for product_folder in os.listdir(training_dir):
#     product_path = os.path.join(training_dir, product_folder)
#     if os.path.isdir(product_path):
#         images = []
#         for img_filename in os.listdir(product_path):
#             img_path = os.path.join(product_path, img_filename)
#             print(f"Loading training image: {img_path}")
#             img = image.load_img(img_path, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)
#             images.append(img_array)
#         training_images[product_folder] = images

# # Load validation images
# validation_images = {}
# for product_folder in os.listdir(validation_dir):
#     product_path = os.path.join(validation_dir, product_folder)
#     if os.path.isdir(product_path):
#         images = []
#         for img_filename in os.listdir(product_path):
#             img_path = os.path.join(product_path, img_filename)
#             print(f"Loading validation image: {img_path}")
#             img = image.load_img(img_path, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)
#             images.append(img_array)
#         validation_images[product_folder] = images

# # Perform one/few-shot learning
# # (You'll need to implement this part based on your chosen approach)

# # Validation
# correct_predictions = 0
# total_predictions = 0

# for product_folder in validation_images:
#     for img_array in validation_images[product_folder]:
#         similarities = {}
#         for train_product_folder in training_images:
#             for train_img_array in training_images[train_product_folder]:
#                 similarity_score = cosine_similarity(img_array.reshape(1, -1),
#                                                       train_img_array.reshape(1, -1))
#                 similarities[train_product_folder] = similarity_score[0][0]

#         most_similar_product = max(similarities, key=similarities.get)
#         if most_similar_product == product_folder:
#             correct_predictions += 1

#         total_predictions += 1

# # Calculate and print validation accuracy
# accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
# print(f'Validation Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)')

# import os
# import cv2
# import numpy as np
# import random
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt

# # Set paths for training and validation data
# training_dir = '/kaggle/input/reco-part2/Product Recoginition/Training Data'
# validation_dir = '/kaggle/input/reco-part2/Product Recoginition/Validation Data'

# from tensorflow.keras.preprocessing import image as keras_image

# def read_image(index, base_dir, target_size=(224, 224)):
#     path = os.path.join(base_dir, index[0], index[1])

#     # Print the generated path for debugging
#     print(f"Loading image: {path}")

#     try:
#         # Use Keras' image.load_img to read the image
#         img = keras_image.load_img(path, target_size=target_size)
#         img_array = keras_image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         return img_array
#     except Exception as e:
#         print(f"Error loading image: {path} - {e}")
#         return None



# def get_batch(triplet_list, base_dir, batch_size=256, preprocess=True):
#     batch_steps = len(triplet_list) // batch_size

#     for i in range(batch_steps + 1):
#         anchor = []
#         positive = []
#         negative = []

#         j = i * batch_size
#         while j < (i + 1) * batch_size and j < len(triplet_list):
#             a, p, n = triplet_list[j]
#             anchor_img = read_image(a, base_dir)
#             positive_img = read_image(p, base_dir)
#             negative_img = read_image(n, base_dir)

#             if anchor_img is not None and positive_img is not None and negative_img is not None:
#                 anchor.append(anchor_img)
#                 positive.append(positive_img)
#                 negative.append(negative_img)

#             j += 1

#         anchor = np.array(anchor)
#         positive = np.array(positive)
#         negative = np.array(negative)

#         if preprocess:
#             anchor = preprocess_input(anchor)
#             positive = preprocess_input(positive)
#             negative = preprocess_input(negative)

#         yield ([anchor, positive, negative])

# def create_triplets(directory):
#     triplets = []

#     # Iterate through each class directory
#     for class_folder in os.listdir(directory):
#         class_path = os.path.join(directory, class_folder)

#         if os.path.isdir(class_path):
#             # Get a list of image filenames in the current class
#             image_filenames = os.listdir(class_path)

#             # Create triplets for each anchor image in the current class
#             for anchor_img_filename in image_filenames:
#                 anchor_img_path = os.path.join(class_path, anchor_img_filename)

#                 # Randomly select a positive example from the same class
#                 positive_img_filename = random.choice(image_filenames)
#                 positive_img_path = os.path.join(class_path, positive_img_filename)

#                 # Randomly select a negative example from a different class
#                 other_classes = [c for c in os.listdir(directory) if c != class_folder]
#                 negative_class = random.choice(other_classes)
#                 negative_img_filename = random.choice(os.listdir(os.path.join(directory, negative_class)))
#                 negative_img_path = os.path.join(directory, negative_class, negative_img_filename)

#                 # Check if all images exist before adding the triplet
#                 if os.path.exists(anchor_img_path) and os.path.exists(positive_img_path) and os.path.exists(negative_img_path):
#                     triplet = (anchor_img_path, positive_img_path, negative_img_path)
#                     triplets.append(triplet)

#     return triplets

# def create_validation_triplets(directory):
#     triplets = []

#     # Iterate through each class directory
#     for class_folder in os.listdir(directory):
#         class_path = os.path.join(directory, class_folder)

#         if os.path.isdir(class_path):
#             # Get a list of image filenames in the current class
#             image_filenames = os.listdir(class_path)

#             # Create triplets for each anchor image in the current class
#             for anchor_img_filename in image_filenames:
#                 anchor_img_path = os.path.join(class_path, anchor_img_filename)

#                 # Randomly select a positive example from the same class
#                 positive_img_filename = random.choice(image_filenames)
#                 positive_img_path = os.path.join(class_path, positive_img_filename)

#                 # Randomly select a negative example from a different class
#                 other_classes = [c for c in os.listdir(directory) if c != class_folder]
#                 negative_class = random.choice(other_classes)
#                 negative_img_filename = random.choice(os.listdir(os.path.join(directory, negative_class)))
#                 negative_img_path = os.path.join(directory, negative_class, negative_img_filename)

#                 # Check if all images exist before adding the triplet
#                 if os.path.exists(anchor_img_path) and os.path.exists(positive_img_path) and os.path.exists(negative_img_path):
#                     triplet = (anchor_img_path, positive_img_path, negative_img_path)
#                     triplets.append(triplet)

#     return triplets

# # Generate triplets for training data
# train_triplet = create_triplets(training_dir)

# # Training
# num_plots = 3
# f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))

# for x in get_batch(train_triplet, training_dir, batch_size=num_plots, preprocess=False):
#     a, p, n = x
#     for i in range(num_plots):
#         axes[i, 0].imshow(a[i])
#         axes[i, 1].imshow(p[i])
#         axes[i, 2].imshow(n[i])
#         i += 1
#     break

# # Generate triplets for validation data
# validation_triplet = create_validation_triplets(validation_dir)

# # Validation
# correct_predictions = 0
# total_predictions = 0

# for x in get_batch(validation_triplet, validation_dir, batch_size=1, preprocess=True):
#     a, p, n = x

#     similarities = {}
#     for train_product_folder in training_images:
#         for train_img_array in training_images[train_product_folder]:
#             similarity_score = cosine_similarity(a.reshape(1, -1), train_img_array.reshape(1, -1))
#             similarities[train_product_folder] = similarity_score[0][0]

#     most_similar_product = max(similarities, key=similarities.get)
#     print(f'Most similar product: {most_similar_product}')

# import os
# import cv2
# import matplotlib.pyplot as plt

# training_dir = '/kaggle/input/reco-part2/Product Recoginition/Training Data'


# def read_and_print_images(directory, num_images=3, target_size=(224, 224)):
#     # Get a list of class folders in the directory
#     class_folders = [class_folder for class_folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, class_folder))]

#     # Loop through class folders
#     for class_folder in class_folders:
#         class_path = os.path.join(directory, class_folder)

#         # Get a list of image filenames in the current class
#         image_filenames = os.listdir(class_path)

#         # Randomly select a few images from the class
#         selected_images = random.sample(image_filenames, num_images)

#         # Display the selected images
#         for img_filename in selected_images:
#             img_path = os.path.join(class_path, img_filename)

#             # Use OpenCV to read and display the image
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, target_size)

#             plt.imshow(img)
#             plt.title(f"Class: {class_folder} - Image: {img_filename}")
#             plt.show()

# # Example usage for training data
# read_and_print_images(training_dir)


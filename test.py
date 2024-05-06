
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from sklearn.metrics.pairwise import cosine_similarity
#
# model = load_model('E:/cv/deep learning model/vision94e.h5')
#
# test_dir = 'E:/cv/test_reco'
# reference_dir = 'E:/cv/Product Recoginition/Training Data'
# reference_images = {}
#
# correct_predictions = 0
# total_predictions = 0
#
# # Load reference images
# for class_dir in os.listdir(reference_dir):
#     class_path = os.path.join(reference_dir, class_dir)
#     if os.path.isdir(class_path):
#         for img_filename in os.listdir(class_path):
#             reference_image_path = os.path.join(class_path, img_filename)
#             reference_img = image.load_img(reference_image_path, target_size=(224, 224))
#             reference_img_array = image.img_to_array(reference_img)
#             reference_img_array = np.expand_dims(reference_img_array, axis=0)
#             reference_img_array = preprocess_input(reference_img_array)
#             reference_images[img_filename] = (class_dir, reference_img_array)
#
# # Class-wise predictions
# class_accuracies = {}
#
# for class_name in os.listdir(test_dir):
#     class_path = os.path.join(test_dir, class_name)
#
#     if os.path.isdir(class_path):
#         class_correct_predictions = 0
#         class_total_predictions = 0
#
#         for filename in os.listdir(class_path):
#             img_path = os.path.join(class_path, filename)
#
#             # Check if the file exists before trying to load it
#             if os.path.exists(img_path):
#                 test_img = image.load_img(img_path, target_size=(224, 224))
#                 test_img_array = image.img_to_array(test_img)
#                 test_img_array = np.expand_dims(test_img_array, axis=0)
#                 test_img_array = preprocess_input(test_img_array)
#
#                 similarities = {}
#                 for ref_filename, (ref_class, reference_img_array) in reference_images.items():
#                     similarity_score = cosine_similarity(test_img_array.reshape(1, -1),
#                                                           reference_img_array.reshape(1, -1))
#                     similarities[ref_class] = (ref_class, similarity_score[0][0])
#
#                 most_similar_class, _ = max(similarities.items(), key=lambda x: x[1][1])
#                 predicted_class = similarities[most_similar_class][0]
#
#                 print(
#                     f'Test image {filename} in class {class_name} belongs to class {predicted_class} (Similarity: {similarities[most_similar_class][1]:.4f})')
#
#                 # Update class-wise accuracy metrics
#                 class_total_predictions += 1
#                 if predicted_class == class_name:
#                     class_correct_predictions += 1
#
#                 # Update overall accuracy metrics
#                 total_predictions += 1
#                 if predicted_class == class_name:
#                     correct_predictions += 1
#
#             else:
#                 print(f'Test image {filename} in class {class_name} not found.')
#
#         # Calculate and print class-wise accuracy
#         class_accuracy = class_correct_predictions / class_total_predictions if class_total_predictions > 0 else 0
#         class_accuracies[class_name] = class_accuracy
#         print(f'Class {class_name} Accuracy: {class_accuracy:.2%} ({class_correct_predictions}/{class_total_predictions} correct predictions)')
#         print()
#
# # Calculate and print overall accuracy
# accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
# print(f'Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)')
#
# # Print class-wise accuracies
# print('\nClass-wise Accuracies:')
# for class_name, class_accuracy in class_accuracies.items():
#     print(f'Class {class_name}: {class_accuracy:.2%}')





import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

model = load_model('E:/cv/siames_30/siamese_model_weights.h5')

test_dir = 'E:/cv/test_reco'
reference_dir = 'E:/cv/Product Recoginition/Training Data'
reference_images = {}

correct_predictions = 0
total_predictions = 0
class_count = 0

for class_dir in os.listdir(reference_dir):
    class_path = os.path.join(reference_dir, class_dir)
    if os.path.isdir(class_path):
        class_count += 1

        for img_filename in os.listdir(class_path):
            reference_image_path = os.path.join(class_path, img_filename)
            reference_img = image.load_img(reference_image_path, target_size=(224, 224))
            reference_img_array = image.img_to_array(reference_img)
            reference_img_array = np.expand_dims(reference_img_array, axis=0)
            reference_img_array = preprocess_input(reference_img_array)
            reference_images[img_filename] = (class_dir, reference_img_array)
print(f'Total number of classes in {reference_dir}: {class_count}')


class_accuracies = {}

for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)

    if os.path.isdir(class_path):
        class_correct_predictions = 0
        class_total_predictions = 0

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)

            if os.path.exists(img_path):
                test_img = image.load_img(img_path, target_size=(224, 224))
                test_img_array = image.img_to_array(test_img)
                test_img_array = np.expand_dims(test_img_array, axis=0)
                test_img_array = preprocess_input(test_img_array)

                similarities = {}
                for ref_filename, (ref_class, reference_img_array) in reference_images.items():
                    similarity_score = cosine_similarity(test_img_array.reshape(1, -1),
                                                          reference_img_array.reshape(1, -1))
                    similarities[ref_class] = (ref_class, similarity_score[0][0])

                most_similar_class, _ = max(similarities.items(), key=lambda x: x[1][1])
                predicted_class = similarities[most_similar_class][0]

                print(
                    f'Test image {filename} in class {class_name} belongs to class {predicted_class} (Similarity: {similarities[most_similar_class][1]:.4f})')

                class_total_predictions += 1
                if predicted_class == class_name:
                    class_correct_predictions += 1

                total_predictions += 1
                if predicted_class == class_name:
                    correct_predictions += 1

            else:
                print(f'Test image {filename} in class {class_name} not found.')

        class_accuracy = class_correct_predictions / class_total_predictions if class_total_predictions > 0 else 0
        class_accuracies[class_name] = class_accuracy
        print(f'Class {class_name} Accuracy: {class_accuracy:.2%} ({class_correct_predictions}/{class_total_predictions} correct predictions)')
        print()


accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f'Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)')


print('\nClass-wise Accuracies:')
for class_name, class_accuracy in class_accuracies.items():
    print(f'Class {class_name}: {class_accuracy:.2%}')

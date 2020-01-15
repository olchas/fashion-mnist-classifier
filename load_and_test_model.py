import json

import cv2
import numpy as np
from keras import metrics
from keras.applications.vgg16 import preprocess_input
from keras.datasets import fashion_mnist
from keras.models import load_model
from keras.preprocessing import image as image_manip
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


def rescale_and_preprocess_images(dataset, resize_dim):
    """Resize image to size required by VGG16 net and apply the same preprocessing as in original net"""
    preproceses_images = []
    for image in dataset:
        image_preprocessed = cv2.resize(image, (resize_dim, resize_dim))
        image_preprocessed = image_manip.img_to_array(image_preprocessed)
        image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
        image_preprocessed = preprocess_input(image_preprocessed)
        preproceses_images.append(image_preprocessed[0])
    return np.array(preproceses_images)


def load_test_data():

    (_, _), (x_test, y_test) = fashion_mnist.load_data()

    # rescale to three channels
    x_test = np.stack((x_test,)*3, axis=-1)

    # vgg16 requires images to be at least 32x32 (max pooling 5 times) -> reshaping images to 56x56 should suffice
    x_test = rescale_and_preprocess_images(x_test, x_test.shape[1] * 2)

    number_of_classes = len(np.unique(y_test))

    y_test = np_utils.to_categorical(y_test, number_of_classes)

    return x_test, y_test


def top_1_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def analyze_results(model, x_test, y_test):

    prob = model.predict(x_test, verbose=1) 
    y_pred = prob.argmax(axis=-1)

    y_test_labels = [np.where(test_case == 1)[0][0] for test_case in y_test]

    cm = confusion_matrix(y_test_labels, y_pred)

    accuracy = cm.diagonal().sum()/cm.sum()

    # accuracy per class
    class_accuracies = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()

    # classes sorted by their accuracy
    class_ranking = np.argsort(class_accuracies)

    class_accuracy_dict = [{'class': int(label), 'accuracy': float(class_accuracies[label])} for label in class_ranking]

    # most commonly made mistakes
    most_common_mistakes = [{'mistakes_count': int(cm[el//10][el%10]), 'expected_class': int(el//10), 'predicted_class': int(el%10)} for el in np.argsort(-cm, axis=None) if el//10 != el%10]

    return accuracy, cm, class_accuracy_dict, most_common_mistakes


if __name__ == "__main__":

    model = load_model('model/vgg16_model.h5', custom_objects={"top_1_accuracy": top_1_accuracy, "top_5_accuracy": top_5_accuracy})

    print('model loaded')
    
    x_test, y_test = load_test_data()

    print('data preprocessed')

    accuracy, cm, class_accuracy_dict, most_common_mistakes = analyze_results(model, x_test, y_test)

    print('accuracy: {}'.format(accuracy))
    print('confuson_matrix:\n {}'.format(cm))
    print('class_accuracy_dict:')
    print(json.dumps(class_accuracy_dict, indent=4, sort_keys=True))
    print('most_common_mistakes:')
    print(json.dumps(most_common_mistakes[:10], indent=4, sort_keys=True))
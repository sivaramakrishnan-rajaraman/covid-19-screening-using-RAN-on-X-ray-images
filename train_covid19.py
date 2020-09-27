
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, DenseNet201, mobilenet, ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from keras import backend as K

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 123

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

DATASET_DIR = "./dataset"
INIT_LR = 1e-4
EPOCHS = 100
BATCH_SIZE = 8
NUM_CLASSES = 2
HEIGHT = 224
WIDTH = 224
CHANNEL = 3


print("[INFO] loading images...")
imagePaths = list(paths.list_images(DATASET_DIR))
data = []
labels = []



def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    data.append(image)
    labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.30, stratify=labels, random_state=42)
(testX, valX, testY, valY) = train_test_split(testX, testY, test_size=0.30, stratify=testY, random_state=42)

# testY = to_categorical(testY)
# trainY = to_categorical(trainY)
# valY = to_categorical(valY)
# print (trainX.shape)
# print (testX.shape)
# print (valX.shape)

from collections import Counter
# print (np.unique(trainY, return_counts=True))
# print (np.unique(testY, return_counts=True))
# print (np.unique(valY, return_counts=True))




# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")



'''
[[25  0]
 [ 3 22]]
acc: 0.9400
sensitivity: 1.0000
specificity: 0.8800
precision: 0.8929
recall: 1.0000


[[11  0]
 [ 2  9]]
acc: 0.9091
sensitivity: 1.0000
specificity: 0.8182
precision: 0.8462
recall: 1.0000
'''
# load the VGG16 network, ensuring the head FC layer sets are left
# VGG16 acc: 0.9375
# acc: 0.9091
# CONV Layer: 13
# RESULTS = './results/csv/covid19_VGG_16.csv'
# PLOT_NAME = "./results/plot_VGG16.png"
# MODEL_NAME = "./results/covid19_VGG16.h5"
# PLOT_TITLE = "VGG 16"
# baseModel = VGG16(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)


'''
[[24  1]
 [ 2 23]]
acc: 0.9400
sensitivity: 0.9600
specificity: 0.9200
precision: 0.9231
recall: 0.9583

[[10  1]
 [ 0 11]]
acc: 0.9545
sensitivity: 0.9091
specificity: 1.0000
precision: 1.0000
recall: 0.9167
'''
# VGG19 acc: 0.9200
# acc: 0.9545
# CONV Layer: 16
INIT_LR = 1e-4
RESULTS = './results/csv/covid19_VGG_19_new.csv'
PLOT_NAME = "./results/plot_VGG19_new.png"
MODEL_NAME = "./results/covid19_VGG19_new.h5"
PLOT_TITLE = "VGG 19"
baseModel = VGG19(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
'''
##########
VGG 19 NEW
###########
[INFO] evaluating network...
              precision    recall  f1-score   support

       covid       0.93      1.00      0.96        25
      normal       1.00      0.92      0.96        25

    accuracy                           0.96        50
   macro avg       0.96      0.96      0.96        50
weighted avg       0.96      0.96      0.96        50

[[25  0]
 [ 2 23]]
acc: 0.9600
sensitivity: 1.0000
specificity: 0.9200
precision: 0.9259
recall: 1.0000
[INFO] evaluating validation...
              precision    recall  f1-score   support

       covid       1.00      1.00      1.00        11
      normal       1.00      1.00      1.00        11

    accuracy                           1.00        22
   macro avg       1.00      1.00      1.00        22
weighted avg       1.00      1.00      1.00        22

[[11  0]
 [ 0 11]]
acc: 1.0000
sensitivity: 1.0000
specificity: 1.0000
precision: 1.0000
recall: 1.0000
[INFO] saving COVID-19 detector model...


'''

# acc: 0.7400
# acc: 0.7727
# CONV Layer: 120
'''
[[23  2]
 [ 7 18]]
acc: 0.8200
sensitivity: 0.9200
specificity: 0.7200
precision: 0.7667
recall: 0.9000

[[9 2]
 [2 9]]
acc: 0.8182
sensitivity: 0.8182
specificity: 0.8182
precision: 0.8182
recall: 0.8182

'''
# INIT_LR = 5e-3
# RESULTS = './results/csv/covid19_DenseNet121.csv'
# PLOT_NAME = "./results/plot_DenseNet121.png"
# MODEL_NAME = "./results/covid19_DenseNet121.h5"
# PLOT_TITLE = "DenseNet121"
# baseModel = DenseNet121(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)


'''
[[13 12]
 [ 0 25]]
acc: 0.7600
sensitivity: 0.5200
specificity: 1.0000
precision: 1.0000
recall: 0.6757

[[ 9  2]
 [ 1 10]]
acc: 0.8636
sensitivity: 0.8182
specificity: 0.9091
precision: 0.9000
recall: 0.8333
'''
# acc: 0.7200
# acc: 0.7727
# CONV Layer: 200
# INIT_LR = 5e-3
# RESULTS = './results/csv/covid19_DenseNet201.csv'
# PLOT_NAME = "./results/plot_DenseNet201.png"
# MODEL_NAME = "./results/covid19_DenseNet201.h5"
# PLOT_TITLE = "DenseNet201"
# baseModel = DenseNet201(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)


'''
[[10 15]
 [ 0 25]]
acc: 0.7000
sensitivity: 0.4000
specificity: 1.0000
precision: 1.0000
recall: 0.6250

[[ 6  5]
 [ 0 11]]
acc: 0.7727
sensitivity: 0.5455
specificity: 1.0000
precision: 1.0000
recall: 0.6875
'''
from tensorflow.keras.applications import InceptionResNetV2
# acc: 0.6800
# acc: 0.6364
# CONV Layer: 244
# RESULTS = './results/csv/covid19_InceptionResNetV2.csv'
# PLOT_NAME = "./results/plot_InceptionResNetV2.png"
# MODEL_NAME = "./results/covid19_InceptionResNetV2.h5"
# PLOT_TITLE = "InceptionResNetV2"
# baseModel = InceptionResNetV2(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)



'''
[[18  7]
 [ 0 25]]
acc: 0.8600
sensitivity: 0.7200
specificity: 1.0000
precision: 1.0000
recall: 0.7812

[[ 6  5]
 [ 1 10]]
acc: 0.7273
sensitivity: 0.5455
specificity: 0.9091
precision: 0.8571
recall: 0.6667
'''
from tensorflow.keras.applications import MobileNetV2
# acc: 0.5600
# acc: 0.5909
# Conv: 52
# RESULTS = './results/csv/covid19_MobileNetV2.csv'
# PLOT_NAME = "./results/plot_MobileNetV2.png"
# MODEL_NAME = "./results/covid19_MobileNetV2.h5"
# PLOT_TITLE = "MobileNetV2"
# baseModel = MobileNetV2(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)



'''
[[14 11]
 [ 0 25]]
acc: 0.7800
sensitivity: 0.5600
specificity: 1.0000
precision: 1.0000
recall: 0.6944


[[ 9  2]
 [ 0 11]]
acc: 0.9091
sensitivity: 0.8182
specificity: 1.0000
precision: 1.0000
recall: 0.8462
'''
from tensorflow.keras.applications import MobileNet
# acc: 0.8800
# acc: 0.9091
# CONV: 32
# RESULTS = './results/csv/covid19_MobileNet.csv'
# PLOT_NAME = "./results/plot_MobileNet.png"
# MODEL_NAME = "./results/covid19_MobileNet.h5"
# PLOT_TITLE = "MobileNet"
# baseModel = MobileNet(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)



'''
[[21  4]
 [ 0 25]]
acc: 0.9200
sensitivity: 0.8400
specificity: 1.0000
precision: 1.0000
recall: 0.8621


[[10  1]
 [ 2  9]]
acc: 0.8636
sensitivity: 0.9091
specificity: 0.8182
precision: 0.8333
recall: 0.9000
'''
from tensorflow.keras.applications import Xception
# acc: 0.9600
# acc: 0.8636
# CONV: 40
# RESULTS = './results/csv/covid19_Xception.csv'
# PLOT_NAME = "./results/plot_Xception.png"
# MODEL_NAME = "./results/covid19_Xception.h5"
# PLOT_TITLE = "Xception"
# baseModel = Xception(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)




'''
[[18  7]
 [ 2 23]]
acc: 0.8200
sensitivity: 0.7200
specificity: 0.9200
precision: 0.9000
recall: 0.7667


[[ 9  2]
 [ 1 10]]
acc: 0.8636
sensitivity: 0.8182
specificity: 0.9091
precision: 0.9000
recall: 0.8333
'''
from tensorflow.keras.applications import NASNetLarge
# acc: 0.8800
# CONV: 296
# RESULTS = './results/csv/covid19_NASNetLarge.csv'
# PLOT_NAME = "./results/plot_NASNetLarge.png"
# MODEL_NAME = "./results/covid19_NASNetLarge.h5"
# PLOT_TITLE = "NASNetLarge"
# baseModel = NASNetLarge(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)



'''
[[18  7]
 [ 8 17]]
acc: 0.7000
sensitivity: 0.7200
specificity: 0.6800
precision: 0.6923
recall: 0.7083


[[9 2]
 [5 6]]
acc: 0.6818
sensitivity: 0.8182
specificity: 0.5455
precision: 0.6429
recall: 0.7500
'''
# from tensorflow.keras.applications import NASNetMobile
# #acc: 0.80
# #acc: 0.7727
# # CONV: 224
# RESULTS = './results/csv/covid19_NASNetMobile.csv'
# PLOT_NAME = "./results/plot_NASNetMobile.png"
# MODEL_NAME = "./results/covid19_NASNetMobile.h5"
# PLOT_TITLE = "NASNetMobile"
# baseModel = NASNetMobile(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNEL)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(64, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)







from keras.utils.vis_utils import plot_model
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


print(len(model.layers))
print(model.layers)
# print(model.summary())




opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=[['accuracy', f1_m, precision_m, recall_m]])


print("[INFO] training...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    epochs=EPOCHS)


print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))



cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
recall = cm[1, 1] / (cm[0, 1] + cm[1, 1])


print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
print("precision: {:.4f}".format(precision))
print("recall: {:.4f}".format(recall))





print("[INFO] evaluating validation...")
predIdxs = model.predict(valX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(valY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

cm = confusion_matrix(valY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
recall = cm[1, 1] / (cm[0, 1] + cm[1, 1])

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
print("precision: {:.4f}".format(precision))
print("recall: {:.4f}".format(recall))


# Save results
import pandas as pd
pd.DataFrame(H.history).to_csv(RESULTS, index=False)


# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(MODEL_NAME)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from imutils import paths
import cv2
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import keras
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from models.models import AttentionResNetCifar10, AttentionResNet56, AttentionResNet92
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd


DATASET_DIR = "./dataset"
# INIT_LR = 5e-6
INIT_LR = 1e-5
EPOCHS = 110
BATCH_SIZE = 8
NUM_CLASSES = 2
HEIGHT = 224
WIDTH = 224
CHANNEL = 3
# PLOT_NAME = "./results/plt_Attention_92.png"
# MODEL_NAME = "./results/covid19_Attention_92.h5"
# RESULTS = './results/csv/covid19_Attention.csv'

PLOT_NAME = "./results/plt_Attention_92_vanilla.png"
MODEL_NAME = "./results/covid19_Attention_92_vanilla.h5"
RESULTS = './results/csv/covid19_Attention_vanilla.csv'

print("[INFO] loading images...")
imagePaths = list(paths.list_images(DATASET_DIR))
data = []
labels = []

'''
[[25  0]
 [ 1 24]]
acc: 0.9800
sensitivity: 1.0000
specificity: 0.9600
precision: 0.9615
recall: 1.0000


[[11  0]
 [ 0 11]]
acc: 1.0000
sensitivity: 1.0000
specificity: 1.0000
precision: 1.0000
recall: 1.0000

'''

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

# initialize the training data augmentation object
train_datagen  = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")

val_datagen = ImageDataGenerator(
    featurewise_std_normalization=True)




# build a model
# model = AttentionResNetModified(shape=(HEIGHT, WIDTH, CHANNEL), n_channels=CHANNEL, n_classes=2)
model = AttentionResNet56(shape=(HEIGHT, WIDTH, CHANNEL), n_channels=CHANNEL, n_classes=2)




print(len(model.layers))
print(model.layers)
print (trainX.shape)
print (testX.shape)
print (valX.shape)



# prepare usefull callbacks
# early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1)
# lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=7, min_lr=10e-9, epsilon=0.01, verbose=1)
# callback = [early_stopper, lr_reducer]

model.compile(keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS), loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])


# define loss, metrics, optimizer
# model.compile(keras.optimizers.Adam(lr=INIT_LR), loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
# RESULTS = './results/csv/covid19_Attention_without_decay.csv'

# fits the model on batches with real-time data augmentation
batch_size = 8
H = model.fit_generator(train_datagen.flow(trainX, trainY, batch_size=batch_size),
                    steps_per_epoch=len(trainX)//batch_size, epochs=EPOCHS,
                    validation_data=val_datagen.flow(testX, testY, batch_size=batch_size),
                    validation_steps=len(testX)//batch_size)
                    # callbacks=callback, initial_epoch=0)

# results = model.evaluate_generator(val_datagen.flow(valX, valY), steps=len(valX)/32, use_multiprocessing=True)
# print (results)





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


# plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()

pd.DataFrame(H.history).to_csv(RESULTS, index=False)

# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Residual Attention Network")
# plt.xlabel("#Epoch")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(PLOT_NAME)

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save(MODEL_NAME)

'''
val_recall_m: 0.8958
[INFO] evaluating network...
              precision    recall  f1-score   support

       covid       0.92      0.88      0.90        25
      normal       0.88      0.92      0.90        25

    accuracy                           0.90        50
   macro avg       0.90      0.90      0.90        50
weighted avg       0.90      0.90      0.90        50

[[22  3]
 [ 2 23]]
acc: 0.9000
sensitivity: 0.8800
specificity: 0.9200
precision: 0.9167
recall: 0.8846
[INFO] evaluating validation...
              precision    recall  f1-score   support

       covid       0.91      0.91      0.91        11
      normal       0.91      0.91      0.91        11

    accuracy                           0.91        22
   macro avg       0.91      0.91      0.91        22
weighted avg       0.91      0.91      0.91        22

[[10  1]
 [ 1 10]]
acc: 0.9091
sensitivity: 0.9091
specificity: 0.9091
precision: 0.9091
recall: 0.9091
[INFO] saving COVID-19 detector model...

'''
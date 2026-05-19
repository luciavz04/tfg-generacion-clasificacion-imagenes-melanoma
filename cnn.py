

import os
import sys
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage import io
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, ReLU, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser(description='Train VGG16 on dermoscopy images')
parser.add_argument('--epochs',     type=int, default=500, help='Number of training epochs (default: 500)')
parser.add_argument('--batch-size', type=int, default=32,  help='Batch size for training (default: 32)')
parser.add_argument('--augment',    action='store_true',   help='Apply data augmentation during training')
args = parser.parse_args()

LABELS       = ['invasivo', 'in_situ'] # 0 = benign, 1 = pathologic
i_LABELS     = [0, 1]
IMAGE_WIDTH  = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH  = 3 # color channels (RGB)

# Define the folder where you saved the processed data
LOAD_DATA_FOLDER = '/home/jpdominguez/projects/TFGLuciaVela/TFG_Processed_Data/'

print(f"Loading processed data from: {LOAD_DATA_FOLDER}")

try:
    # Load training data
    train_images = np.load(os.path.join(LOAD_DATA_FOLDER, 'train_images.npy'))
    train_labels = np.load(os.path.join(LOAD_DATA_FOLDER, 'train_labels.npy'))

    # Load validation data
    valid_images = np.load(os.path.join(LOAD_DATA_FOLDER, 'valid_images.npy'))
    valid_labels = np.load(os.path.join(LOAD_DATA_FOLDER, 'valid_labels.npy'))

    # Load test data
    test_images = np.load(os.path.join(LOAD_DATA_FOLDER, 'test_images.npy'))
    test_labels = np.load(os.path.join(LOAD_DATA_FOLDER, 'test_labels.npy'))

    # Update example counts after loading (important for model definition or metrics)
    training_examples = len(train_images)
    validation_examples = len(valid_images)
    test_examples = len(test_images)

    print("All processed data loaded successfully!")
    print('train_images array has shape:', train_images.shape)
    print('train_labels array has shape:', train_labels.shape)
    print('valid_images array has shape:', valid_images.shape)
    print('valid_labels array has shape:', valid_labels.shape)
    print('test_images array has shape:', test_images.shape)
    print('test_labels array has shape:', test_labels.shape)

except FileNotFoundError:
    print(f"Error: Processed data not found in {LOAD_DATA_FOLDER}. Please ensure you have run the saving step first or check the path.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    sys.exit(1)

image_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=image_shape, classes=2,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model = tf.keras.Sequential()
if args.augment:
    vgg_model.add(tf.keras.layers.RandomFlip('horizontal_and_vertical', input_shape=image_shape))
    vgg_model.add(tf.keras.layers.RandomRotation(0.2))
    vgg_model.add(tf.keras.layers.RandomZoom(0.1))
    vgg_model.add(tf.keras.layers.RandomContrast(0.1))
    print("Data augmentation: ENABLED")
else:
    print("Data augmentation: DISABLED (use --augment to enable)")
vgg_model.add(pretrained_model)
vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dropout(0.25))
vgg_model.add(Dense(2, activation='softmax'))

vgg_model.summary() # summary() print the model structures and the description of the parameters.

# tf.keras.utils.plot_model(vgg_model, show_shapes=True) # plot_model draws the scheme

vgg_model.compile( optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])


CNN_OUTPUT_FOLDER = '/home/jpdominguez/projects/TFGLuciaVela/src/output/cnn_out/'
os.makedirs(CNN_OUTPUT_FOLDER, exist_ok=True)

checkpoint_callback = ModelCheckpoint(os.path.join(CNN_OUTPUT_FOLDER, 'model.keras'),
                                      monitor='val_loss',
                                      mode='min',
                                      save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        restore_best_weights=True)

history = vgg_model.fit(train_images,
                            train_labels,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            callbacks=[checkpoint_callback, early_stopping_callback],
                            validation_data=(valid_images, valid_labels))

plt.figure()
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.legend()
plt.savefig(os.path.join(CNN_OUTPUT_FOLDER, 'loss.png'))
plt.close()

plt.figure()
plt.plot(history.history['binary_accuracy'], label="training accuracy")
plt.plot(history.history['val_binary_accuracy'], label="validation accuracy")
plt.legend()
plt.savefig(os.path.join(CNN_OUTPUT_FOLDER, 'accuracy.png'))
plt.close()

# load the trained and saved model (to avoid redo training for future usages)
vgg_model = load_model(os.path.join(CNN_OUTPUT_FOLDER, 'model.keras'), compile=False)

images = valid_images
labels = valid_labels
predictions = vgg_model.predict(images)
print(predictions.shape)

plt.figure(figsize=(20,8))
for i in range(10):
    plt.subplot(2,5,i+1)
    k = random.randint(0, validation_examples - 1)
    image = images[k]
    true  = np.argmax(labels[k,:])
    pred  = np.argmax(predictions[k,:]) # Take closest integer

    plt.imshow(image)
    plt.xlabel(
        "True {}\nPred. {}".format(
            LABELS[true],
            LABELS[pred]
        )
    )
    plt.xticks([])
    plt.yticks([])

plt.savefig(os.path.join(CNN_OUTPUT_FOLDER, 'predictions_sample.png'))
plt.close()

def metrics_identification(y_true, y_pred):
  # Calculate metrics directly using sklearn functions
  precision = precision_score(y_true, y_pred)
  recall    = recall_score(y_true, y_pred)
  accuracy  = accuracy_score(y_true, y_pred)
  f1        = f1_score(y_true, y_pred)

  # For roc_auc_score, it's generally best to use probabilities if available.
  # However, to maintain the 'logic' of using the same `y_pred` as other metrics,
  # which is currently `predictions_argmax` (binary 0/1), we pass that.
  # If `predictions` (raw probabilities) were to be used for AUC,
  # `roc_auc_score(labels_argmax, predictions[:, 1])` would be more appropriate.
  auc_value = roc_auc_score(y_true, y_pred)

  return precision, recall, accuracy, f1, auc_value

# y_pred: has the probabilities converted into 0 and 1s
predictions_argmax = np.argmax(predictions, axis=1)
labels_argmax = np.argmax(labels, axis=1)

print('Confusion matrix: \n', confusion_matrix(labels_argmax, predictions_argmax))

# Evaluate the precision, recall, accuracy, f1 and AUC
precision, recall, accuracy, f1, auc_value = metrics_identification(labels_argmax, predictions_argmax)

print('precision: \t%.3f' %precision)
print('recall: \t%.3f' %recall)
print('accuracy: \t%.3f' %accuracy)
print('f1-score: \t%.3f' %f1)
print('auc: \t\t%.2f' %auc_value)

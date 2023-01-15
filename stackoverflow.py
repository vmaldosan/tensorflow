###################################################################################################
# 
# This notebook trains a multi-class classifier to predict the tag of a programming question on
# Stack Overflow.
# Steps:
# 1. (once) Download text dataset
# 2. Load raw train, validation (80:20 split) and test datasets
# 3. Standardize, tokenize and vectorize data
# 4. Configure dataset for perfomance: cache and prefetch
# 5. Create neural network
# 6. Use a loss function and an optimizer
# 7. Train the model
# 8. Evaluate the model
# 9. Create a plot of accuracy and loss over time
#
###################################################################################################
 
import matplotlib.pyplot as plt
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses


# print (tf.__version__)

# 1. (once) Download text dataset

# url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

# 2. Load raw train, validation (80:20 split) and test datasets

BATCH_SIZE = 32
SEED = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'so_16k/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=SEED
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'so_16k/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=SEED
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'so_16k/test',
    batch_size=BATCH_SIZE
)

# 3. Standardize, tokenize and vectorize data

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print('Review: ', first_review)
# print('Label: ', raw_train_ds.class_names[first_label])
# print('Vectorized review: ', vectorize_text(first_review, first_label))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# 4. Configure dataset for perfomance: cache and prefetch

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 5. Create neural network

EMBEDDING_DIM = 16

model = tf.keras.Sequential([
    layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)
])
model.summary()

# 6. Use a loss function and an optimizer

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# 7. Train the model

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 8. Evaluate the model

loss, accuracy = model.evaluate(test_ds)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

# 9. Create a plot of accuracy and loss over time
history_dict = history.history
# print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

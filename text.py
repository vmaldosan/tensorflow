###################################################################################################
# 
# This notebook trains a sentiment analysis model to classify 50K movie reviews from Imdb as
# positive or negative. These reviews are split into 20K reviews for training, 5K for validation
# and 25K for testing. Both training and test sets are balanced (equal number of positive and 
# negative reviews).
# Steps:
# 1. (once) Download text dataset
# 2. (once) Remove unnecessary folders
# 3. Load raw train, validation (80:20 split) and test datasets
# 4. Standardize, tokenize and vectorize data
# 5. Configure dataset for perfomance: cache and prefetch
# 6. Create neural network
# 7. Use a loss function and an optimizer
# 8. Train the model
# 9. Evaluate the model
# 10. Create a plot of accuracy and loss over time
#
###################################################################################################
 
import matplotlib.pyplot as plt
import os
import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses


# print (tf.__version__)

# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname('.'), 'aclImdb') # os.path.join(os.path.dirname(dataset), 'aclImdb')

print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
    # print(f.read())

# Create validation split
BATCH_SIZE = 32
SEED = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=SEED
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=SEED
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=BATCH_SIZE
)

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

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

EMBEDDING_DIM = 16

model = tf.keras.Sequential([
    layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)
])
model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0)
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

loss, accuracy = model.evaluate(test_ds)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

# Create a plot of accuracy and loss over time
history_dict = history.history
# print(history_dict.keys())
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
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

import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import os
import glob
import shutil

gpu_list = list(filter(lambda x: x.device_type == 'GPU', tf.config.experimental.list_physical_devices()))
if len(gpu_list) > 0:
    tf.config.experimental.set_memory_growth(gpu_list[0], True)
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
TOTAL_SIZE = 12500
PADDING = 'valid'
CLASS_NAMES = ['cat', 'dog']

CACHE_FOLDER = './cache'
TRAIN_CACHE_FOLDER = CACHE_FOLDER + '/gan_cache_train'
VALID_CACHE_FOLDER = CACHE_FOLDER + '/gan_cache_valid'

WEIGHTS_FOLDER = './weights'
__WEIGHTS_CHECKPOINT_PATH = WEIGHTS_FOLDER + '/weights-e{epoch:04d}-acc{accuracy:.3f}-valacc{val_accuracy:.3f}.h5'
__WEIGHTS_FILE_PATTERN = WEIGHTS_FOLDER + '/weights*.h5'

__train_valid_split_ratio = 0.7
steps_per_epoch = int(TOTAL_SIZE * __train_valid_split_ratio / BATCH_SIZE) - 1
valid_steps_per_epoch = int(TOTAL_SIZE * (1 - __train_valid_split_ratio) / BATCH_SIZE) - 1

def get_label(file_path):
    # convert the path to a list of path components
    file_name = tf.strings.split(file_path, '/')[-1]
    # The second to last is the class-directory
    return tf.case([(tf.strings.regex_full_match(file_name, ".*" + CLASS_NAMES[0] + ".*"), lambda: tf.constant([1, 0])),
                    (tf.strings.regex_full_match(file_name, ".*" + CLASS_NAMES[1] + ".*"), lambda: tf.constant([0, 1])),
                    ])


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    print("process path")
    print(file_path)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle_buffer_size > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def split(dataset: tf.data.Dataset, total_size: int, ratio: float):
    take_test = int(total_size * ratio)
    shuffled_dataset = dataset.shuffle(500)
    return shuffled_dataset.take(take_test), shuffled_dataset.skip(take_test).take(total_size - take_test)


def create_model_vgg():
    model = models.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2))

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding=PADDING, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding=PADDING))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.build((None, IMG_WIDTH, IMG_HEIGHT, 3))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def train_model():
    if os.path.isdir(CACHE_FOLDER):
        shutil.rmtree(CACHE_FOLDER)
        os.mkdir(CACHE_FOLDER)
    else:
        os.mkdir(CACHE_FOLDER)

    os.makedirs(os.path.dirname(WEIGHTS_FOLDER), exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../logs/scalars/1', update_freq='batch')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(__WEIGHTS_CHECKPOINT_PATH, save_weights_only=True, save_best_only=True)
    model = create_model_vgg()
    initial_epoch = 1
    weights_file, latest_epoch = get_latest_weights_file_and_epoch()

    if weights_file and latest_epoch:
        initial_epoch = latest_epoch
        model.load_weights(weights_file)

    labeled_ds = tf.data.Dataset.list_files(str('./data/train/*.jpg')).map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds, validation_ds = split(labeled_ds, TOTAL_SIZE, 0.7)
    train_ds = prepare_for_training(train_ds,
                                    cache=TRAIN_CACHE_FOLDER, shuffle_buffer_size=100)
    validation_ds = prepare_for_training(validation_ds,
                                    cache=VALID_CACHE_FOLDER, shuffle_buffer_size=0)
    history = model.fit(train_ds
        , steps_per_epoch=steps_per_epoch
        , epochs=5000, initial_epoch=initial_epoch
        , validation_data=validation_ds, validation_steps=valid_steps_per_epoch
        , callbacks=[tensorboard_callback, model_checkpoint_callback]
    )

def get_latest_weights_file_and_epoch() -> (str, int):
    weights_files = glob.glob(__WEIGHTS_FILE_PATTERN)
    if len(weights_files) > 0:
        latest_file = max(weights_files, key=os.path.getctime)
        re_epoch_search = re.search('.*e(\d{4}).*', latest_file, re.IGNORECASE)
        if re_epoch_search:
            return latest_file, int(re_epoch_search.group(1))
    return None, None


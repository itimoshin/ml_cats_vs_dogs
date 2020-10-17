import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os, random
import numpy as np
from src.main import decode_img, CLASS_NAMES, WEIGHTS_FOLDER

TEST_DIR = '../data/test1'
MODEL_PATH = WEIGHTS_FOLDER + '/test_weights.h5'

def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    print("process path")
    print(file_path)
    return img


def predict_image(file_path, model_arg):
    img = mpimg.imread(file_path)
    imgplot = plt.imshow(img)
    predictions = np.array(tf.nn.softmax(model_arg.predict(tf.stack([process_path(file_path)]))[0]))
    text = "{} {}%, {}".format(
        CLASS_NAMES[np.argmax(predictions)].capitalize(),
        round(np.max(predictions)*100, 2),
        file_path)
    plt.text(0, 0, text, bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.show()


model = tf.keras.applications.VGG16()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'mse'])

test_images = os.listdir(TEST_DIR)
while True:
    input('Press Enter...')
    predict_image(TEST_DIR + "/" + random.choice(test_images), model)
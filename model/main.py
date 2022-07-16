# python code
# tensorflow, keras, numpy and Ipython are the external libraries used (matplotlib -  # optional to plot the results in graphical manner)

import tensorflow as tf
import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from IPython.display import display

# custom made dataset is saved in root_dir
root_dir = r"../input/newfocusdataset/Focus_dataset"
crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8

# training & validation images are prepared with 8:2 ratio
train_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)
valid_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

# scaling the images from (0, 255) to (0, 1)
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)

# validating the process
for batch in train_ds.take(1):
    for img in batch:
        display(array_to_img(img))

# internal test path sorting
test_path = r"../input/newfocusdataset/Focus_dataset/Focus_testImages"
test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)

# using tensorflow operation to process
# converting rgb to yuv color scheme
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")
def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y
train_ds = train_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
train_ds = train_ds.prefetch(buffer_size=32)
valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
valid_ds = valid_ds.prefetch(buffer_size=32)

# to check the images converted to yuv
for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))

# convolution filters are used (no pooling is being used)
# process activation is set to rectified linear unit activation function
def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }

# filter size are defined 
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(512, 5, **conv_args)(inputs)
    x = layers.Conv2D(512, 3, **conv_args)(x)
    x = layers.Conv2D(256, 5, **conv_args)(x)
    x = layers.Conv2D(256, 3, **conv_args)(x)
    x = layers.Conv2D(128, 5, **conv_args)(x)
    x = layers.Conv2D(128, 3, **conv_args)(x)
    x = layers.Conv2D(64, 5, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)
    return keras.Model(inputs, outputs)

# setting early stopping callbacks to secure model checkpoints and weights
# also helps to stop before overtraining
# psnr calculation added
class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super(ESPCNCallback, self).__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

# storing psnr value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []
    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 10 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")
    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
checkpoint_filepath = "./tmp/checkpoint"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)wres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr
    print(
        "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
    )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    imgplot = plt.imshow(lowres_img)
    plt.title("Low Res")
    plt.show()
    imgplot = plt.imshow(prediction)
    plt.title("Prediction")
    plt.show()
    imgplot = plt.imshow(highres_img)
    plt.title("high Res")
    plt.show()
print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))

model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# model epochs were set to 250 but with our dataset went to 119 epoch before overtraining
# verbose set to 1 gives update each second
epochs = 250
model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=1
)

# model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

# predication is compared in terms of psnr
total_bicubic_psnr = 0.0
total_test_psnr = 0.0

for index, test_img_path in enumerate(test_img_paths[50:60]):
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr
    print(
        "PSNR of low resolution image and high resolution image is %.4f" %        bicubic_psnr
    )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    imgplot = plt.imshow(lowres_img)
    plt.title("Low Res")
    plt.show()
    imgplot = plt.imshow(prediction)
    plt.title("Prediction")
    plt.show()
    imgplot = plt.imshow(highres_img)
    plt.title("high Res")
    plt.show()
print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))

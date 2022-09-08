import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from camera import image_creator
from audio_record import recorder

"""Block of the functions used"""


def load_noise_sample(path):
    """Splits noise into chunks of 16000 each"""
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == sampling_rate:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / sampling_rate)
        sample = tf.split(sample[: slices * sampling_rate], slices)
        return sample
    else:
        print(f"Sampling rate for {path} is incorrect. Ignoring it")
        return None


def audio_dir_to_dataset(audio_paths, labels):
    """Constructs the dataset of audios and labels"""

    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def image_dir_to_dataset(image_paths, labels):
    """Constructs the dataset of image tensors and labels"""

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(lambda x: path_to_image(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.Dataset.zip((image_ds, label_ds))



def path_to_audio(path):
    """Reads and decodes an audio file"""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, sampling_rate)
    return audio


def path_to_image(path):
    image = tf.io.read_file(path)
    image, _ = tf.image.decode_image(image, 3, dtype="int32")
    image = tf.image.resize(image, (150, 150))
    return image

def add_noise(audio, noises=None, scale=0.5):
    """Adds noise to the audio recording"""
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have
        tf_rnd = tf.random.uniform((tf.shape(audio)[0],) 0, noises.shape[0], dtype=tf.int32)
        noise = tf.gather(noises, tf_rnd, axis=0)

        # get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # add the rescaled noise to audio
        audio = audio + noise * prop * scale
    
    return audio


def audio_to_fft(audio):
    """Applies Fast Fourier Transform to the audio recording"""
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def residual_block(x, filters, conv_num=3, activation="relu"):
    """Constructs the block of 1D-convolution layers with residual memory"""

    conv0 = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for layer in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, conv0])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def cnn_block():
    """Load pre-trained Xception model to process the watermelons images"""
    xception_model = keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(150, 150, 3)  
    )
    xception_model.trainable = False
    return xception_model


def build_model(audio_input_shape, image_input_shape=(150, 150, 3)):
    """Builds the model applying to the data"""
    audio_input = keras.layers.Input(shape=audio_input_shape, name="audio_input")
    image_input = keras.layers.Input(shape=image_input_shape)
    
    audio_x = residual_block(audio_input, 16, 2)
    audio_x = residual_block(audio_input, 32, 2)
    audio_x = residual_block(audio_input, 64, 3)
    audio_x = residual_block(audio_input, 128, 3)
    audio_x = residual_block(audio_input, 128, 3)
    
    audio_x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(audio_x)
    audio_x = keras.layers.Flatten()(audio_x)
    audio_x = keras.layers.Dense(256, activation="relu")(audio_x)
    audio_x = keras.layers.Dense(128, activation="relu")(audio_x)

    image_x = cnn_block()(image_input)

    final_x = keras.layers.Add()([audio_x, image_x])
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(final_x)

    return keras.models.Model(inputs=[audio_input, image_input], output=output)



# create the folder for datasets in the current working directory
dataset_root = os.path.join(os.getcwd(), "datasets")

# create subfolders for noise and sounds of watermelon bumping
audio_subdir = "audio"
noise_subdir = "noise"
image_subdir = "images"

dataset_audio_path = os.path.join(dataset_root, audio_subdir)
dataset_noise_path = os.path.join(dataset_root, noise_subdir)
dataset_image_path = os.join(dataset_root, image_subdir)


# check if directories exist, otherwise, create them
for direct in [audio_subdir, noise_subdir, image_subdir]:
    if direct in os.listdir(dataset_root):
        continue
    else:
        os.mkdir(os.path.join(dataset_root, direct))
     


# create directories for training and valdition images
# percentage of validation samples
valid_splt = 0.2

# random seed to use for shuffling the dataset
random_seed = 76

# the sampling rate to use (we will transform all data to this rate)
sampling_rate = 16e3

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
scale = 0.5

# the size of batches of data
batch_size = 128

# epochs to be used during learning the netrwok
epochs = 20


# get the list of all noise files
noise_paths = []
for subdir in os.listdir(dataset_noise_path):
    subdir_path = Path(dataset_noise_path) / subdir
    if os.path.isdir(subdir_path):
        noise_paths += [
            os.path.join(subdir_path, filepath) 
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]

print(f"Found {len(noise_paths)} files belonging to {len(os.listdir(dataset_noise_path))} directories")

command = (
    "for dir in `ls -1 " + dataset_noise_path + "`; do "
    "for file in `ls -1 " + dataset_noise_path + "/$dir/*.wav`; do "
    "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
    "$file | grep sample_rate | cut -f2 -d=`; "
    "if [ $sample_rate -ne 16000 ]; then "
    "ffmpeg -hide_banner -loglevel panic -y "
    "-i $file -ar 16000 temp.wav; "
    "mv temp.wav $file; "
    "fi; done; done"
)

os.system(command)

# DATA PROCESSING
noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[1] // sampling_rate
    )
)

audio_files = []
image_files = []
labels = []
ripe_number, green_number = 0, 0

for audio_filename, image_filename in os.listdir(dataset_audio_path), os.listdir(dataset_image_path):
    if "ripe" in audio_filename and "ripe" in image_filename:
        audio_files.append(audio_filename)
        image_files.append(image_filename)
        labels.append(1)
        ripe_number += 1
    elif "green" in audio_filename and "green" in image_filename:
        audio_files.append(audio_filename)
        image_files.append(image_filename)
        labels.append(0)
        green_number += 1

print(f"There are {green_number} green watermelon recordings \n and {ripe_number} ripe watermelon recordings")

# shuffle data
rng = np.random.RandomState(random_seed)
rng.shuffle(audio_files)
rng = np.random.RandomState(random_seed)
rng.shuffle(labels)
rng = np.random.RandomState(random_seed)
rng.shuffle(image_files)

# split the data into training and validation sets
num_val_samples = int(valid_splt * len(audio_files))
print(f"Use {len(audio_files) - num_val_samples} for training")
train_audio_files = audio_files[:num_val_samples]
train_image_files = image_files[:num_val_samples]
train_labels = labels[:num_val_samples]


print(f"Use {num_val_samples} for validation")
valid_audio_files = audio_files[num_val_samples:]
valid_image_files = image_files[num_val_samples:]
valid_labels = labels[num_val_samples:]


# create datasets for training and for validation
train_audio_ds = audio_dir_to_dataset(train_audio_files, train_labels)
train_audio_ds = train_audio_ds.shuffle(buffer_size=batch_size * 8, seed=random_seed).batch(
    batch_size)

train_image_ds = image_dir_to_dataset(train_image_files, train_labels)
train_image_ds = train_image_ds.shuffle(buffer_size=batch_size * 8, seed=random_seed).batch(
    batch_size)

valid_audio_ds = audio_dir_to_dataset(valid_audio_files, valid_labels)
valid_audio_ds = valid_audio_ds.shuffle(buffer_size=32 * 8, seed=random_seed).batch(
    32)

valid_image_ds = image_dir_to_dataset(valid_image_files, valid_labels)
valid_image_ds = valid_image_ds.shuffle(buffer_size=32 * 8, seed=random_seed).batch(
    32)

# add noise to the training and validation audio
train_audio_ds = train_audio_ds.map(
    lambda x, y: (add_noise(x, noises, scale=scale), y), num_parallel_calls=tf.data.AUTOTUNE,
)
valid_audio_ds = valid_audio_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_audio_ds = valid_audio_ds.prefetch(tf.data.AUTOTUNE)


# build the model
model = build_model(audio_input_shape=(sampling_rate // 2, 1))
print(model.summary())
model.compile(
    optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# add callbacks:
# EarlyStopping to prevent over-fitting
# ModelCheckPoint to keep the model with the best accuracy

model_save_filename = "watermodel.h5"

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)

history = model.fit([train_audio_ds, train_image_ds], epochs=epochs,
    validation_data=[valid_audio_ds, valid_image_ds], 
    callbacks=[early_stopping_cb, model_checkpoint_cb])

print(model.evaluate([valid_audio_ds, valid_image_ds]))


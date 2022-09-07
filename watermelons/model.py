
"""Source code: https://keras.io/examples/audio/speaker_recognition_using_cnn/"""


import tensorflow as tf
from tensorflow import keras

from keras import Model 
from tensorflow.keras.utils import img_to_array

import sounddevice as sd
from scipy.io.wavfile import write


import os
import shutil
import numpy as np

from pathlib import Path
from IPython.display import display, Audio



# create the folder for datasets in the current working directory
dataset_root = os.path.join(os.getcwd(), "datasets")

# create subfolders for noise and sounds of watermelon bumping
audio_subfolder = "audio"
noise_subfolder = "noise"

dataset_audio_path = os.path.join(dataset_root, audio_subfolder)
dataset_noise_path = os.path.join(dataset_root, noise_subfolder)

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

# Split noise into chunks of 16000 each
def load_noise_sample(path):
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

# dataset generation
def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels"""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file"""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, sampling_rate)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    """Adds noise to the audio recording"""
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],) 0, noises.shape[0], dtype=tf.int32)
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


# get the list of audio file paths along with their corresponding labels
# at the moment we use binary classification {"ripe": 1, "green": 0}

audio_labels = dict()
ripe_number, green_number = 0, 0
for filename in os.listdir(dataset_audio_path):
    if "ripe" in filename:
        audio_labels[filename] = 1
        ripe_number += 1
    elif "green" in filename:
        audio_labels[filename] = 0
        green_number += 1

print(f"There are {green_number} green watermelon recordings \n and {ripe_number} ripe watermelon recordings")


# shuffle data
audio_files = audio_labels.keys()
labels = audio_files.values()

rng = np.random.RandomState(random_seed)
rng.shuffle(audio_files)
rng = np.random.RandomState(random_seed)
rng.shuffle(labels)

# split the data into training and validation sets
num_val_samples = int(valid_splt * len(audio_files))
print(f"Use {len(audio_files) - num_val_samples} for training")
train_audio_files = audio_files[:num_val_samples]
train_labels = labels[:num_val_samples]

print(f"Use {num_val_samples} for validation")
valid_audio_files = audio_files[num_val_samples:]
valid_labels = labels[num_val_samples:]


# create two datasets, one for training and the other for the validation
train_ds = paths_and_labels_to_dataset(train_audio_files, train_labels)
train_ds = train_ds.shuffle(buffer_size=batch_size * 8, seed = random_seed).batch(
    batch_size
)


valid_ds = paths_and_labels_to_dataset(valid_audio_files, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=random_seed).batch(32)

# add noise to the training set
train_ds = train_ds.map(
    lambda x, y: (add_noise(x, noises, scale=scale), y), num_parallel_calls=tf.data.AUTOTUNE,
)

# transform audio wave to the frequency domain using "audio_to_fft"
train_ds = train_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)


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


def cnn_block(image_tensor, input_shape):
    """Load pre-trained Xception model to process the watermelons images"""
    xception_model = keras.applications.Xception(
        include_top=False,
        weights="imagenet"
    )
    xception_model.trainable = False


def build_model(input_shape):
    """Builds the model applying to the data"""

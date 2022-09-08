import os
import tensorflow as tf
from camera import image_creator
from audio_record import recorder
from tensorflow import keras
from data_processing import user_audio_path, user_image_path, sampling_rate

# load pretrained model
model = keras.models.load_model("watermodel.h5")

# create audio recording and image 
image_creator(user_image_path)
recorder(user_audio_path)

# process the taken data
image = tf.io.read_file(os.path.join(user_image_path, os.listdir(user_image_path)[-1]))
image, _ = tf.image.decode_image(image, dtype="int32")
image = tf.image.resize(image, (150, 150))
image = image / tf.constant(255, dtype=tf.float32)

audio = tf.io.read_file(os.path.join(user_image_path, os.listdir(user_audio_path)[-1]))
audio, _ = tf.audio.decode_wav(audio, 1, sampling_rate)

# make a prediction
prediction = model.fit([audio, image])
print(prediction)
    


import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input


@tf.keras.utils.register_keras_serializable()
class StackLayer(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.stack(inputs, axis=-1)


@tf.keras.utils.register_keras_serializable()
class SliceLayer(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[..., self.index]

    def get_config(self):
        config = super().get_config()
        config.update({
            "index": self.index
        })
        return config


# urutan kelas harus sama persis seperti saat training
CLASS_NAMES = ['Biru', 'Hijau', 'Ungu']

# load model sekali saat aplikasi pertama jalan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(
    os.path.join(BASE_DIR, 'model_deploy.keras'),
    custom_objects={
        "StackLayer": StackLayer,
        "SliceLayer": SliceLayer
    }
)

def predict_image(image_path):
    # load gambar dan resize ke 224x224
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))

    # ubah gambar jadi array angka
    img_array = tf.keras.utils.img_to_array(img)

    # tambah dimensi batch: (224,224,3) → (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    # preprocessing sesuai ResNet50 (sama persis seperti saat training)
    img_array = preprocess_input(img_array)

    # catat waktu sebelum inferensi dimulai
    start = time.time()

    # jalankan prediksi
    predictions = model.predict(img_array)

    # hitung durasi inferensi dalam detik (3 angka di belakang koma)
    elapsed = round(time.time() - start, 3)

    # ambil index kelas dengan skor tertinggi
    predicted_index = np.argmax(predictions[0])

    # ambil label dan confidence score
    label = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    # buat list skor semua kelas
    probs = [
        {'label': CLASS_NAMES[i], 'score': round(float(predictions[0][i]) * 100, 2)}
        for i in range(len(CLASS_NAMES))
    ]

    # kembalikan label, confidence, probs, dan waktu inferensi
    return label, confidence, probs, elapsed
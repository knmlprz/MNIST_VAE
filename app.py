from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Flask app
app = Flask(__name__)

# VAE decoder
VAE_decoder = tf.keras.models.load_model('decoder.h5')


# Routes and redicrects
@app.route('/')
def index():
    buffered = BytesIO()
    x = request.args.get('x')
    y = request.args.get('y')
    if x and y:
        data = np.array([[16 * (float(x) / 1024) - 7,
                          -9.8 * (float(y) / 663) + 4.4]])
        img_data = VAE_decoder.predict(data)
        img_data = img_data[0, :, :, 0]
        img_data = Image.fromarray(np.uint8(img_data * 255), 'L')
        img_data.save(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = str(img_str)[2:-1]
        return render_template('index.html', vae_img=img_str)

    return render_template('index.html', vae_img='None')



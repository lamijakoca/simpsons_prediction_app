import numpy as np
import os
from flask import Flask, render_template, request
from tensorflow.keras.utils import img_to_array, load_img
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, static_folder='static')

app.config['TEMPLATES_AUTO_RELOAD'] = True

# This is loads the model
saved_model = load_model('./simpsons_model.h5')

# These are all the labels and possible Simpsons characters
labels = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'troy_mcclure', 'waylon_smithers']

# This function is used to predict the image
def predict(img):
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = saved_model.predict(img_batch)
    prediction = labels[np.argmax(prediction)]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    uploaded_file  = request.files['file']

    # Check if the uploads directory exists, create it if necessary
    if not os.path.exists('./static/upload'):
        os.makedirs('./static/upload')

    
    file_path = os.path.join('./static/upload', uploaded_file.filename)
    uploaded_file.save(file_path)

    img = image.load_img(file_path, target_size=(64, 64, 3))
    prediction = predict(img)

    return render_template('upload.html', file_path=file_path, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

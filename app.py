import os
from secrets import token_hex

import numpy as np
from flask import Flask, render_template, request
from keras_preprocessing.image import img_to_array, load_img

from load import init
from utils import rescale

model = init()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 4
app.config["UPLOAD_DIRECTORY"] = "uploads"
app.config["DEBUG"] = os.environ.get("DEBUG")
app.config["ENV"] = os.environ.get("ENV")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.files.get("image")
    image_data_filename = os.path.join(
        app.config["UPLOAD_DIRECTORY"],
        str(token_hex(16)) + os.path.splitext(image_data.filename)[1],
    )
    image_data.save(image_data_filename)

    x = load_img(image_data_filename, target_size=(224, 224))
    x = np.array(img_to_array(x))
    x = rescale(x)

    value = None

    pred = model.predict_proba(x.reshape(1, 224, 224, 3)).flatten()
    print(pred)
    for i in pred:
        if 0 <= i <= 0.2:
            value = "Cat"
        elif 0.99 <= i <= 1:
            value = "Dog"
        else:
            value = "Neither a cat nor a dog"

    return {"message": value}


if __name__ == "__main__":
    app.run()

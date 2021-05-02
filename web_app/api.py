# way to upload image: endpoint
# way to save the image
# function to make predictions on the image
# show the results
import os
from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
from model import ResNet50_model, get_model, face_detector, dog_detector_ResNet50, predict_breed_transfer, predict_breed_for_human
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/feliciatamasovics/Documents/ML/Term2/Capstone/web_app/static"


RESNET50 = ResNet50_model()
model_transfer = get_model()


@app.route("/", methods=["GET","POST"])
def upload_predict():
    if request.method == "GET":
        return render_template('index.html')

    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
                )
            image_file.save(image_location)

            ## handle cases for a human face, dog, and neither
            if face_detector(image_location):
                print("Hi human")
                doggyname, filepath = predict_breed_for_human(image_location, model_transfer)
                print(f'You look like a ... {doggyname, filepath}')
                filepath="alldogs/"+doggyname.replace(' ', '_')+".jpg"
                return render_template("index.html", human_dog_name=doggyname, uploaded_img=image_file.filename, img=filepath)

            elif dog_detector_ResNet50(image_location, RESNET50):
                print("Hi puppy!")
                probs, class_names, top_class = predict_breed_transfer(image_location, model_transfer)
                print("I think you are: ", class_names)
                print(probs)
                img = "alldogs/"+class_names[0].replace(' ', '_')+".jpg"
                return render_template("index.html", dog_breeds=class_names, prediction=probs, uploaded_img=image_file.filename, img=img)

            else:
                print("Could not detect a human or dog")
                photo=True
                return render_template("index.html", neither=photo, uploaded_img=image_file.filename)

            return render_template("index.html")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=12000, debug=True)

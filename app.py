from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.applications import ResNet50
import pickle
import keras.utils as image

app = Flask(__name__)

img_width, img_height = 224, 224  # Default input size for ResNet50

# Initialize the model
model = Sequential()

# ResNet50 model
conv_base = ResNet50(weights='imagenet',
                    include_top=False,
                    input_shape=(img_width, img_height, 3))

# Add the convolutional base to the model
model.add(conv_base)
# Add GlobalAveragePooling2D layer to the model
model.add(GlobalAveragePooling2D())

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

    # Extract features matrix
    features = np.around(model.predict(img_tensor.reshape(1, img_width, img_height, 3)), decimals=2)

    return features[0]  # the [0] is just to make it a vector instead of a matrix

def recommend(user_file, user_in):
    # Load the features list for the selected class
    item_to_load = user_in
    file_to_load = f"./static/Extracted_Features-ResNet50/{item_to_load}.pkl"
    with open(file_to_load, 'rb') as file:
        loaded_list = pickle.load(file)    

    # Extract features for the input image
    img_features = extract_features(user_file)

    # Compare the features vectors to get the closest image
    results = []
    for i in range(len(loaded_list)):
        results.append(np.linalg.norm(img_features - loaded_list[i][1]))  # compute L2 norm between the features vectors, append to results list

    # Get the best image
    best = results.index(min(results))  # index of the image with the least cost
    best_name = loaded_list[best][0]
    best_path = f"./static/Dataset/{user_in}/{best_name}.jpg"

    return best_path

@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/Test')
def Test():
    return render_template('Test.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded image
    selected_image = request.files['image']

    # Get the selected option from the dropdown
    selected_option = request.form['options']

    # Check if an image was uploaded
    if selected_image:
        # Construct the path to save the image inside the 'static/images' folder
        image_path = f"static/images/{selected_image.filename}"

        # Save the uploaded image to the specified path
        selected_image.save(image_path)

        result_path = recommend(image_path, selected_option)
        print(result_path)

        # Redirect to the result page with the recommended image
        return render_template('result.html', result_path=result_path, selected_option=selected_option, image_path = image_path)
    else:
        return "No image uploaded."

if __name__ == '__main__':
    app.run(debug=True)
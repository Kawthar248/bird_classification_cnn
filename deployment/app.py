from flask import Flask, render_template, request
import torch
from PIL import Image
from torchvision.transforms import ToTensor, transforms
import numpy as np 

app = Flask(__name__)

# Load the model
model = torch.load('models/bird_classification_model.pth', map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

@app.route('/')
def index():
    return render_template('index.html', prediction="")

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return render_template('index.html', prediction="No photo uploaded")
    
    # Get the uploaded image
    photo = request.files['photo']
    
    # Check if the file is empty
    if photo.filename == '':
        return render_template('index.html', prediction="No photo uploaded")
   
    # Predict the label of the test_images
    predicted_label = model.predict(photo)

    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    
    
    





from flask import Flask, render_template, request
import torch
from PIL import Image
from torchvision import transforms
import os
from src.model import build_model

app = Flask(__name__)

def get_class_names(data_directory):
    """
    Extracts class names from subdirectories of a given data directory.
    Args:
        data_directory (str): Path to the dataset directory containing subdirectories for each class.
    Returns:
        dict: A dictionary mapping class indices to class names.
    """
    class_names = [name for name in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, name))]
    class_names.sort()  # Optional: sort the directory names if order is important
    return {index: name for index, name in enumerate(class_names)}

class_names = get_class_names('datadata/train')

# Load the model
model = build_model()
model.load_state_dict(torch.load('models/bird_classification_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])


@app.route('/')
def index():
    return render_template('index.html', prediction="")


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'photo' not in request.files:
            return render_template('index.html', prediction="No photo uploaded")
        
        # Get the uploaded image
        photo = request.files['photo']
        
        # Check if the file is empty
        if photo.filename == '':
            return render_template('index.html', prediction="No photo uploaded")
        
        image = Image.open(photo.stream).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict the label of the test_images
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label_index = predicted.item()
            predicted_label_name = class_names.get(predicted_label_index, "Unknown class")
            print("Prediction:", predicted_label_name)
            print("Filename:", photo.filename)

        return render_template('index.html', prediction=predicted_label_name, filename=photo.filename)
    except Exception as e:
        return f"An error occurred: {e}"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080 , debug = True)
    
    
    





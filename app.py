import os
import cv2
import torch
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from utils.video_processing import extract_frames_and_faces

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXTRACTED_FRAMES_FOLDER'] = os.path.join('static', 'extracted_frames')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_FRAMES_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Define CNN Model
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load Model
model = DeepfakeCNN()
model.load_state_dict(torch.load('model/deepfake_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Face Detection Function
def process_image(file_path):
    try:
        img = Image.open(file_path).convert('RGB')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_cv = cv2.imread(file_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Crop faces
        cropped_faces = [img.crop((x, y, x + w, y + h)) for (x, y, w, h) in faces]
        return cropped_faces
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Image Processing
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                faces = process_image(file_path)
                if not faces:
                    return render_template('result.html', result="No face detected.", score=0)
                
                predictions = []
                for face in faces:
                    face_tensor = transform(face).unsqueeze(0)
                    with torch.no_grad():
                        output = model(face_tensor)
                        _, predicted = torch.max(output, 1)
                        predictions.append(predicted.item())
                
                # Average Prediction
                real_count = predictions.count(0)
                fake_count = predictions.count(1)
                avg_result = 'real' if real_count > fake_count else 'fake'
                avg_score = round(real_count / (real_count + fake_count) * 100, 2)
                return render_template('result.html', result=avg_result, score=avg_score)

            # Video Processing
            elif filename.lower().endswith(('mp4', 'avi', 'mov')):
                faces, frame_paths = extract_frames_and_faces(file_path, app.config['EXTRACTED_FRAMES_FOLDER'])
                
                if not faces:
                    return render_template('result.html', result="No face detected.", score=0)
                
                predictions = []
                for face in faces:
                    face_tensor = transform(face).unsqueeze(0)
                    with torch.no_grad():
                        output = model(face_tensor)
                        _, predicted = torch.max(output, 1)
                        predictions.append(predicted.item())
                
                real_count = predictions.count(0)
                fake_count = predictions.count(1)
                avg_result = 'real' if real_count > fake_count else 'fake'
                avg_score = round(real_count / (real_count + fake_count) * 100, 2)
                
                static_frame_paths = [os.path.join('static', 'extracted_frames', os.path.basename(path)).replace("\\", "/") for path in frame_paths]
                
                return render_template('result.html', frame_paths=static_frame_paths, result=avg_result, score=avg_score)
            
            else:
                flash("Unsupported file type.")
                return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/result')
def result():
    return render_template('result.html')

# Run on AWS Port 8080
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # AWS expects 8080
    app.run(host='0.0.0.0', port=port)

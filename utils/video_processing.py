import cv2
import os
from PIL import Image

def extract_frames_and_faces(video_path, output_folder='static/extracted_frames', max_frames=20):

    max_frames = int(max_frames)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    faces = []
    frame_paths = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        #frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for face_num, (x, y, w, h) in enumerate(detected_faces):
            # Extract the face
            face = frame[y:y+h, x:x+w]

           
            face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            faces.append(face_image)

            
            face_filename = f"frame_{frame_count}_face_{face_num}.jpg"
            face_path = os.path.join(output_folder, face_filename)
            face_image.save(face_path)
            frame_paths.append(face_filename)

        frame_count += 1

    cap.release()

    return faces, frame_paths

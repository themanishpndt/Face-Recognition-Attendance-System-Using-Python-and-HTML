import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Function to speak a message
def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Initialize Haar Cascade
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create directory for saving data
if not os.path.exists('data'):
    os.makedirs('data')

# Function to capture new faces
def capture_faces(name):
    video = cv2.VideoCapture(0)
    faces_data = []
    i = 0

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error accessing camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, f"Captured: {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= 100:
            break

    video.release()
    cv2.destroyAllWindows()

    # Save the data
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    # Check if names file exists
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save the faces data
    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

    return "Dataset created successfully!"

# Function to start face recognition and attendance
def start_recognition():
    # Load Data for Recognition
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)

    faces_data = faces_data.reshape(faces_data.shape[0], -1)

    # Initialize KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_data, names)

    # Video Capture for Recognition
    video = cv2.VideoCapture(0)
    imgBackground = cv2.imread(r"C:\Users\MANISH SHARMA\Downloads\bg.png")  # Background image path

    COL_NAMES = ['NAME', 'TIME']

    # Ensure the directory exists
    attendance_dir = r"C:\Users\MANISH SHARMA\OneDrive\Desktop\p3\Attendance"
    os.makedirs(attendance_dir, exist_ok=True)

    # Recognition threshold to classify as "Unknown"
    recognition_threshold = 3000
    # You can adjust this based on your dataset

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error accessing camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        attendance = []
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            
            # Predict face using KNN
            distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
            closest_distance = distances[0][0]
            detected_name = knn.predict(resized_img)[0] if closest_distance < recognition_threshold else "Unknown"
            
            # Timestamp
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            attendance = [detected_name, timestamp]
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(detected_name), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        # Overlay frame on background
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Face Recognition", imgBackground)

        k = cv2.waitKey(1)
        if k == ord('o'):
            speak("Attendance Taken..")
            time.sleep(2)
            
            # Full path for the attendance CSV file
            file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")
            file_exists = os.path.isfile(file_path)

            # Write to CSV
            if file_exists:
                with open(file_path, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open(file_path, "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    return "Recognition complete and attendance taken!"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        name = request.form['name']
        if name:
            result = capture_faces(name)
            return render_template('result.html', result=result)
        else:
            return render_template('capture.html', error="Please enter a name.")
    return render_template('capture.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        result = start_recognition()
        return render_template('result.html', result=result)
    return render_template('recognize.html')

@app.route('/attendance')
def show_attendance():
    attendance_records = []
    date = datetime.now().strftime("%d-%m-%Y")
    file_path = os.path.join(r"C:\Users\MANISH SHARMA\OneDrive\Desktop\p3\Attendance", f"Attendance_{date}.csv")

    if os.path.isfile(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            attendance_records = list(reader)

    return render_template('attendance.html', records=attendance_records)

if __name__ == '__main__':
    app.run(debug=True)
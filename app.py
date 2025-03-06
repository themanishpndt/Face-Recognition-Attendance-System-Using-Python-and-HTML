import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Initialize Flask app
app = Flask(__name__)


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
    # Set optimal camera properties for faster capture
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 30)
    
    faces_data = []
    i = 0
    required_samples = 50  # Reduced from 100 to 50 samples
    
    # Pre-calculate face cascade parameters for speed
    scale_factor = 1.2  # Reduced from 1.3 for faster detection
    min_neighbors = 4  # Reduced from 5, still maintains accuracy
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error accessing camera.")
            break

        # Optimize frame processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce frame size for faster processing
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Optimize face detection
        faces = facedetect.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)  # Minimum face size to detect
        )

        # Scale back the coordinates to original frame size
        faces = [(int(x * 2), int(y * 2), int(w * 2), int(h * 2)) for x, y, w, h in faces]

        for (x, y, w, h) in faces:
            if len(faces_data) < required_samples:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                # Remove the modulo operation for faster capture
                faces_data.append(resized_img)
                
                # Visual feedback
                progress = (len(faces_data) / required_samples) * 100
                cv2.putText(frame, f"Progress: {progress:.0f}%", (50, 50), 
                           cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= required_samples:
            break

    video.release()
    cv2.destroyAllWindows()

    # Save the data
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(required_samples, -1)

    # Check if names file exists
    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * required_samples
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [name] * required_samples
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
    try:
        # Check if data directory exists
        if not os.path.exists('data'):
            print("Error: Data directory not found")
            return "Please register at least one face before starting recognition"

        # Check if required files exist
        if not os.path.exists('data/names.pkl') or not os.path.exists('data/faces_data.pkl'):
            print("Error: Required data files not found")
            return "Please register at least one face before starting recognition"

        # Load Data for Recognition
        try:
            with open('data/names.pkl', 'rb') as f:
                names = pickle.load(f)
            print(f"Successfully loaded names.pkl with {len(names)} names")
            if len(names) == 0:
                print("Error: names list is empty")
                return "No registered users found. Please register at least one face before starting recognition"
        except Exception as e:
            print(f"Error loading names.pkl: {str(e)}")
            return "Error loading face data. Please try registering your face again"

        try:
            with open('data/faces_data.pkl', 'rb') as f:
                faces_data = pickle.load(f)
            print(f"Successfully loaded faces_data.pkl with shape {faces_data.shape}")
            if faces_data.size == 0:
                print("Error: faces_data array is empty")
                return "Error: No face data found"
        except Exception as e:
            print(f"Error loading faces_data.pkl: {str(e)}")
            return "Error: Failed to load faces data"

        try:
            faces_data = faces_data.reshape(faces_data.shape[0], -1)
            print(f"Successfully reshaped faces_data to {faces_data.shape}")
            print(f"Number of face samples: {faces_data.shape[0]}")
            print(f"Number of names: {len(names)}")
            if faces_data.shape[0] != len(names):
                print(f"Error: Mismatch between number of faces ({faces_data.shape[0]}) and names ({len(names)})")
                return "Error: Data mismatch between faces and names"
        except Exception as e:
            print(f"Error reshaping faces_data: {str(e)}")
            return "Error: Failed to process faces data"

        # Initialize KNN Classifier
        try:
            if faces_data.shape[0] < 5:
                print("Error: Not enough samples for KNN classifier (need at least 5)")
                return "Error: Not enough face samples for recognition"
            
            knn = KNeighborsClassifier(n_neighbors=min(5, faces_data.shape[0]))
            knn.fit(faces_data, names)
            print("Successfully trained KNN classifier")
        except Exception as e:
            print(f"Error training KNN classifier: {str(e)}")
            return "Error: Failed to train recognition model"

        # Video Capture for Recognition
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Error: Could not open video capture device")
            return "Error: Camera not accessible"

        imgBackground = cv2.imread(r"C:\Users\MANISH SHARMA\OneDrive\Desktop\p8\static\bg.png")
        if imgBackground is None:
            print("Error: Could not load background image")
            return "Error: Background image not found"

        COL_NAMES = ['NAME', 'DATE', 'TIME']

        attendance_dir = r"C:\Users\MANISH SHARMA\OneDrive\Desktop\p8\Attendance"
        os.makedirs(attendance_dir, exist_ok=True)

        # Recognition threshold to classify as "Unknown"
        recognition_threshold = 3000
        # You can adjust this based on your dataset

        attendance_list = []  # List to store multiple attendance records

        while True:
            ret, frame = video.read()
            if not ret:
                print("Error accessing camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                
                distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
                closest_distance = distances[0][0]
                detected_name = knn.predict(resized_img)[0] if closest_distance < recognition_threshold else "Unknown"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(frame, str(detected_name), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
            # Overlay frame on background
            imgBackground[162:162 + 480, 55:55 + 640] = frame
            cv2.imshow("Face Recognition", imgBackground)

            key = cv2.waitKey(1)
            if key == ord('o'):  # Press 'o' to mark attendance
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y+h, x:x+w, :]
                    resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                    detected_name = knn.predict(resized_img)[0]
                    
                    if detected_name != "Unknown":
                        ts = time.time()
                        record_date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                        attendance_list.append([detected_name, record_date, timestamp])
                        speak(f"Attendance recorded for {detected_name}")
            
            elif key == ord('q'):  # Press 'q' to quit
                break

        video.release()
        cv2.destroyAllWindows()

        # Save attendance after session ends
        if attendance_list:
            date = datetime.now().strftime("%d-%m-%Y")
            file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")
            file_exists = os.path.isfile(file_path)

            with open(file_path, "a" if file_exists else "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(COL_NAMES)
                writer.writerows(attendance_list)
            speak("Attendance has been saved")

        return "Recognition complete and attendance taken!"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "Error: An unexpected error occurred"

# User Management functions
def get_all_users():
    if os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        return list(set(names))  # Return unique names
    return []

def delete_user(username):
    if os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        # Remove all instances of the username
        names = [name for name in names if name != username]
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
        return True
    return False

# Settings functions
def save_settings(settings_dict):
    with open('data/settings.pkl', 'wb') as f:
        pickle.dump(settings_dict, f)

def load_settings():
    if os.path.exists('data/settings.pkl'):
        with open('data/settings.pkl', 'rb') as f:
            return pickle.load(f)
    return {
        'camera_index': 0,
        'required_samples': 50,
        'attendance_threshold': 0.6
    }

# Export functions
def export_attendance_csv():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'attendance_export_{timestamp}.csv'
    export_path = os.path.join('data', filename)
    
    if os.path.exists('data/attendance.csv'):
        with open('data/attendance.csv', 'r') as source:
            with open(export_path, 'w', newline='') as target:
                reader = csv.reader(source)
                writer = csv.writer(target)
                for row in reader:
                    writer.writerow(row)
        return filename
    return None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        name = request.form['name']
        # Check if name contains only letters and spaces
        if name and all(c.isalpha() or c.isspace() for c in name) and not name.isspace():
            result = capture_faces(name)
            return render_template('result.html', result=result)
        else:
            return render_template('capture.html', error="Please enter a valid name using only alphabets and spaces between names.")
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
    file_path = os.path.join(r"C:\Users\MANISH SHARMA\OneDrive\Desktop\p8\Attendance", f"Attendance_{date}.csv")

    if os.path.isfile(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            attendance_records = list(reader)

    return render_template('attendance.html', attendance_data=attendance_records)

@app.route('/manage_users')
def manage_users():
    users = get_all_users()
    return render_template('manage_users.html', users=users)

@app.route('/delete_user/<username>')
def delete_user_route(username):
    if delete_user(username):
        return redirect(url_for('manage_users'))
    return "Error deleting user", 400

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        new_settings = {
            'camera_index': int(request.form.get('camera_index', 0)),
            'required_samples': int(request.form.get('required_samples', 50)),
            'attendance_threshold': float(request.form.get('attendance_threshold', 0.6))
        }
        save_settings(new_settings)
        return redirect(url_for('settings'))
    
    current_settings = load_settings()
    return render_template('settings.html', settings=current_settings)

@app.route('/export_attendance')
def export_attendance():
    filename = export_attendance_csv()
    if filename:
        return send_from_directory('data', filename, as_attachment=True)
    return "No attendance data to export", 400

if __name__ == '__main__':
    app.run(debug=True)
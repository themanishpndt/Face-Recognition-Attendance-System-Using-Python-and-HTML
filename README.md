
# **Attendance Face Recognition System Using Python and HTML**

## **Project Overview**

This is a simple Face Recognition Attendance System that uses Python, Flask, and OpenCV to recognize faces and log attendance. The system captures faces from a webcam feed and compares them with stored known faces. Once a match is found, it logs the name, date, and time of attendance. This project provides a web interface to interact with the system.

### **Key Features:**
- Real-time face recognition using OpenCV and `face_recognition` library.
- Logging of attendance in a CSV file.
- Web interface using Flask to capture and display attendance.
- Modular design with separate templates for different pages.
- Supports multiple faces for recognition and attendance logging.

## **Technologies Used:**
- **Python**: Core programming language for implementing the face recognition logic and Flask backend.
- **Flask**: Web framework used for serving the application.
- **OpenCV**: Library for computer vision tasks such as face detection.
- **face_recognition**: A Python library for facial recognition.
- **HTML, CSS, JavaScript**: Used for creating the web interface.
- **CSV**: To store attendance data.
- **Pickle**: To store and load face encodings and names.

## **Project Structure**

```
attendance-face-recognition/
│
├── app.py                       # Main Python script for the Flask app
├── haarcascade_frontalface_default.xml   # Haar Cascade XML file for face detection
├── attendance.csv               # CSV file to store attendance data (Name, Date, Time)
├── data/
│   ├── faces_data.pkl           # Pickle file to store face encodings for recognition
│   └── names.pkl                # Pickle file to store names corresponding to face encodings
├── static/
│   └── bg.png                   # Background image used for the web pages (static assets)
├── templates/
│   ├── base.html                # Base HTML template to be extended by other pages
│   ├── attendance.html          # Page displaying the attendance details (CSV data)
│   ├── capture.html             # Page for capturing the face for attendance recognition
│   ├── error.html               # Error page for invalid access or issues
│   ├── index.html               # Main page that loads the webcam for face recognition
│   ├── recognize.html           # Page for displaying real-time face recognition status
│   └── result.html              # Page showing the result of face recognition (attendance logged)
└── README.md                    # Project overview and setup instructions
```

## **Installation Instructions**

### **1. Clone the repository:**

```bash
git clone https://github.com/themanishpndt/Face-Recognition-Attendance-System-Using-Python-and-HTML.git
cd Face-Recognition-Attendance-System-Using-Python-and-HTML
```

### **2. Install the required dependencies:**

Ensure you have Python 3.x installed. Then, create a virtual environment and install the required libraries:

```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
```

If you don't have `requirements.txt` file, you can manually install the dependencies:

```bash
pip install flask opencv-python face_recognition numpy
```

### **3. Setup the face data:**

To use face recognition, you need to encode faces for individuals who will be recognized by the system. Use the following code to register new faces:

```python
import face_recognition
import pickle

# Load image and encode face
image = face_recognition.load_image_file("path/to/image.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

# Save face encoding and name to pickle files
with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump([face_encoding], f)

with open('data/names.pkl', 'wb') as f:
    pickle.dump(["Person Name"], f)
```

### **4. Run the Flask Application:**

Once the dependencies are installed and the face data is set up, run the Flask app:

```bash
python app.py
```

The server will start, and you can access the web application by opening the following URL in your browser:

```
http://127.0.0.1:5000
```

---

## **Usage Instructions**

### **1. Home Page:**
- The home page will display the webcam feed and automatically start face recognition.
- If a known face is detected, the attendance will be logged.

### **2. Attendance Page:**
- This page displays a table of attendance records from `attendance.csv`, including the name, date, and time of attendance.

### **3. Capture Face Page:**
- Use this page to manually register new faces by capturing and encoding them.
- Face data is saved to `faces_data.pkl` and corresponding names to `names.pkl`.

### **4. Error Page:**
- If there is an issue, such as no face being detected, this page will display an error message.

### **5. Result Page:**
- After a face is recognized and attendance is logged, this page shows the result of the face recognition.

---

## **Contributing**

If you would like to contribute to this project, feel free to fork the repository, make changes, and create a pull request. Any improvements or bug fixes are welcome.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**

- This project uses the `face_recognition` library for face encoding and recognition.
- The `OpenCV` library is used for face detection and processing.
- Flask is used for building the web interface.

---

This `README.md` file provides a comprehensive overview of the project, how to set it up, and how to use the face recognition-based attendance system.

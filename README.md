# Emotion-Detector-Using-CNN-Flask
This project is a real-time facial emotion recognition system built using Convolutional Neural Networks (CNN) and deployed via a Flask web server. It detects and classifies emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise from webcam input.

🚀 Features
Real-time emotion detection using webcam

Deep learning model trained on a custom dataset

Live prediction displayed on a web browser

Flask-based backend for smooth integration

OpenCV for face detection

📁 Project Structure
php
Copy
Edit
emotion-detector/
│
├── static/
│   └── style.css               # Styling for the web UI
│
├── templates/
│   └── index.html              # HTML UI for the app
│
├── model/
│   └── emotion_model.h5        # Trained CNN model
│
├── dataset/                    # Custom facial emotion image dataset
│
├── app.py                      # Main Flask application
├── detector.py                 # Webcam capture and preprocessing
├── train_model.py              # Script to train the CNN
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
💻 Technologies Used
Python

Flask

OpenCV

TensorFlow / Keras

NumPy, Pandas

Matplotlib (for visualization)

🧠 Emotion Classes
Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

⚙️ How to Run the Project
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/emotion-detector.git
cd emotion-detector
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Train the Model (Optional)
bash
Copy
Edit
python train_model.py
Or use the pre-trained emotion_model.h5

4. Start the Flask App
bash
Copy
Edit
python app.py
5. Open in Browser
Visit http://localhost:5000 to use the live webcam-based emotion detector.

🖼️ Sample Output
Webcam feed with face box

Predicted emotion label displayed in real-time

📊 Model Performance
Achieved high accuracy (~90%) on validation dataset.

Trained on a balanced dataset of 7 emotion categories.

Used data augmentation to improve generalization.

🙌 Credits
Kaggle Facial Expression Dataset

Flask documentation

OpenCV tutorials

📃 License
This project is licensed under the MIT License. Feel free to use and modify it for educational and non-commercial purposes.


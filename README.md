🌿 Potato Leaf Disease Detection 🥔

A Deep Learning based web application that detects potato leaf diseases using CNN and TensorFlow/Keras.

📌 About The Project

This project uses a Convolutional Neural Network (CNN) model to identify diseases in potato leaves from uploaded images.
The application helps farmers and researchers detect plant diseases quickly and accurately through an easy-to-use web interface.

✨ Features

✅ Detects Potato Leaf Diseases
✅ Identifies:

Early Blight
Late Blight
Healthy Leaves

✅ Deep Learning CNN Model
✅ Flask Web Application
✅ Upload Leaf Images for Prediction
✅ Simple & Responsive UI
✅ Fast and Accurate Predictions

🛠️ Technologies Used
Technology	Purpose
Python	Backend Programming
Flask	Web Framework
TensorFlow / Keras	Deep Learning Model
OpenCV	Image Processing
NumPy	Numerical Operations
HTML & CSS	Frontend Design


📂 Project Structure
Potato_Leaf_Disease/
│
├── app.py
├── train_model.py
├── potatoes.h5
├── requirements.txt
│
├── dataset/
├── model/
├── static/
└── templates/
📊 Dataset

Dataset used from Kaggle PlantVillage Dataset:

PlantVillage Dataset

Classes Used
Potato___Early_blight
Potato___Late_blight
Potato___healthy


🚀 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/potato-leaf-disease-detection.git
2️⃣ Navigate To Project Folder
cd potato-leaf-disease-detection
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run The Application
python app.py


🧠 Train The Model

To train the CNN model:

python train_model.py

The trained model will be saved as:

potatoes.h5
📸 Project Screenshots
🏠 Home Page
![Home Page](screenshots/home.png)
🔍 Disease Prediction
![Prediction Result](screenshots/result.png)

<p align="center">
  <h1 align="center"> Emotion Detection</h1>
  <p align="center">
    <strong>Deep learning-powered facial emotion recognition with real-time dashboard</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow" />
    <img src="https://img.shields.io/badge/Flask-3.0+-000?logo=flask&logoColor=white" alt="Flask" />
    <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV" />
  </p>
</p>

---

##  About

**Emotion Detection** is an open-source project that uses a custom **Convolutional Neural Network (CNN)** to recognize human emotions from facial expressions in real time. It was built to explore the intersection of computer vision and affective computing — the field concerned with systems that can interpret and respond to human emotions.

The system is trained on grayscale face images resized to **48×48 pixels** and can classify a face into one of **7 universal emotion categories**:

> 😠 Angry · 🤢 Disgust · 😨 Fearful · 😊 Happy · 😐 Neutral · 😢 Sad · 😲 Surprised

### How It Works

1. **Face Detection** — OpenCV's Haar Cascade Classifier scans each frame or image to locate face regions.
2. **Preprocessing** — Detected faces are cropped, converted to grayscale, and resized to 48×48 pixels.
3. **Emotion Prediction** — The cropped face is passed through a 5-block CNN that produces a probability distribution over the 7 emotion classes via a softmax output layer.
4. **Visualization** — Bounding boxes are drawn around each face with the predicted emotion label and confidence score overlaid.

### Why This Project?

Emotion detection has broad practical applications: from **mental health monitoring** and **driver alertness systems** to **customer sentiment analysis** and **interactive entertainment**. This project provides a clean, end-to-end reference implementation — covering model training, evaluation, and deployment — that is easy to adapt for research or production use.

### Who Is It For?

- Researchers exploring facial affect recognition
- Developers building emotion-aware applications
- Students learning deep learning and computer vision
- Anyone curious about how machines perceive human feelings

---

##  Overview

A CNN-based facial emotion detection system that classifies faces into **7 emotion categories**: Angry, Disgust, Fearful, Happy, Neutral, Sad, and Surprised. Comes with a sleek **web dashboard** for real-time predictions.

###  Key Features

| Feature | Description |
|---------|-------------|
|  **Deep CNN Architecture** | 5 Conv blocks with BatchNorm, Dropout, and 2 Dense layers |
|  **Data Augmentation** | Rotation, flip, zoom, shift, shear to prevent overfitting |
|  **Class Balancing** | Automatic class weights for imbalanced datasets |
|  **Smart Training** | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callbacks |
|  **Label Smoothing** | Reduces overconfidence, improves generalization |
|  **Full Evaluation** | Confusion matrix, classification report, training curves |
|  **Web Dashboard** | Upload images or use webcam for real-time emotion detection |

---

##  Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/emotion-detection.git
cd emotion-detection/Emotion-detection

# Install dependencies
pip install -r requirements.txt
```

###  Train the Model

Run `project.py` in Google Colab or Jupyter Notebook (requires a dataset zip with emotion-labeled face images):

```bash
python project.py
```

This will:
1. Extract and split the dataset (80/20 train/test)
2. Train the CNN with augmentation and callbacks
3. Generate `training_curves.png` and `confusion_matrix.png`
4. Save the model (`emotion_model.json` + `emotion_model.weights.h5`)
5. Print a full classification report

###  Launch the Dashboard

```bash
python dashboard.py
```

Open **http://localhost:5000** in your browser.

---

##  Model Architecture

```
Input (48×48×1 grayscale)
    │
    ├── Conv2D(32) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ├── Conv2D(256) → BatchNorm → MaxPool → Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(1024) → BatchNorm → Dropout(0.5)
    ├── Dense(512) → BatchNorm → Dropout(0.5)
    └── Dense(num_classes, softmax)
```

### Training Optimizations

- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Categorical Crossentropy with label smoothing (0.1)
- **Callbacks:** EarlyStopping (patience=8), ReduceLROnPlateau (factor=0.5), ModelCheckpoint
- **Augmentation:** Rotation(15°), shift(15%), zoom(15%), horizontal flip, shear(15%)

---

##  Dashboard Features

| Panel | Description |
|-------|-------------|
|  **Image Upload** | Drag & drop images for face detection + emotion prediction |
|  **Live Webcam** | Real-time video feed with emotion overlays |
|  **Model Architecture** | Layer-by-layer summary with parameter counts |
|  **Dataset Distribution** | Bar charts showing class balance |
|  **Emotion Map** | Visual grid of all emotion class labels |

Built with a **dark glassmorphism UI** — gradient accents, glass cards, micro-animations, and responsive layout.

---

##  Project Structure

```
Emotion-detection/
├── project.py           # Model training & evaluation
├── dashboard.py         # Flask web dashboard
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

**Generated after training:**
```
├── emotion_model.json           # Model architecture
├── emotion_model.weights.h5     # Trained weights
├── training_history.json        # Epoch-wise metrics
├── training_curves.png          # Accuracy & loss plots
├── confusion_matrix.png         # Confusion matrix heatmap
└── emotion_dict.json            # Emotion class mapping
```

---

##  Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV (Haar Cascade face detection)
- **Web:** Flask
- **Data Science:** NumPy, scikit-learn, Matplotlib, Seaborn

---

##  License

This project is open source and available under the [MIT License](LICENSE).

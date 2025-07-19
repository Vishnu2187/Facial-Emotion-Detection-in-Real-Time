# Facial Emotion Detection using CNN and OpenCV

This project implements a real-time facial emotion recognition system using a Convolutional Neural Network (CNN), trained on the FER-2013 dataset. It includes scripts to train the model and detect emotions in real time using your webcam and OpenCV.

---

## 🧠 Model Summary

- Trained using FER-2013 dataset (grayscale 48x48 face images)
- CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- Predicts one of 7 facial expressions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## 🗂️ Project Structure

```
├── train_and_save_model.py         # Train CNN on FER-2013 and save model
├── emotion_detection.py            # Real-time webcam-based emotion detector
├── fer2013.csv                     # Dataset (must be downloaded)
├── haarcascade_frontalface_default.xml # Haarcascade file (download separately)
├── emotion_model.h5                # Trained model (output from training)
└── README.md
```

---

## ⚙️ Requirements

Install necessary dependencies using pip:

```bash
pip install tensorflow numpy pandas scikit-learn opencv-python
```

---

## 📥 Dataset & Haarcascade

- Download FER-2013: [Kaggle FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- Download Haarcascade: [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

Place both files in the same directory as the scripts.

---

## 🚀 How to Run

### 🧪 1. Train the Model

```bash
python train_and_save_model.py
```
- Trains a CNN model on `fer2013.csv`
- Saves model as `emotion_model.h5`

### 🎥 2. Real-Time Emotion Detection

```bash
python emotion_detection.py
```
- Starts webcam
- Detects and displays emotions in real-time using Haarcascade and trained model

---

## 🧠 Emotion Labels

The model predicts one of the following:

```
['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
```

---

## 📊 Performance

- Accuracy: ~65–70% (can improve with data augmentation or deeper model)
- Live prediction speed: ~20 FPS (depends on system hardware)

---

## 💡 Future Enhancements

- Add emotion logging or analytics
- Deploy as a web app with Flask or Streamlit
- Improve accuracy with data augmentation and deeper architectures

---

## 📬 Contact

For suggestions, improvements, or questions — feel free to fork this repo or open an issue!

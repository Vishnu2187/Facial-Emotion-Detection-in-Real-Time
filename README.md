# Facial Emotion Detection using CNN and OpenCV

This project is a real-time facial emotion recognition system using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset. It detects faces from webcam input and classifies them into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features

- Real-time face detection using Haarcascade.
- Emotion classification using a custom-built CNN.
- Trained on FER-2013 dataset with grayscale images (48x48).
- Achieves around 65% validation accuracy.
- Webcam integration for real-time predictions.

## Dataset

- **FER-2013** dataset: Available as a CSV file (`fer2013.csv`).
- Each row contains:
  - `emotion`: Integer label (0–6)
  - `pixels`: Space-separated string representing 48x48 grayscale image.

## Model Architecture

- 2 Convolutional Layers (ReLU)
- 2 MaxPooling Layers
- Dropout Layers for regularization
- Flatten and Dense Layers
- Softmax output for 7 emotion classes

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib

## How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/facial-emotion-detection.git
   cd facial-emotion-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following files are present:
   - `fer2013.csv`
   - `haarcascade_frontalface_default.xml`

4. Run the project:

   ```bash
   python emotion_detection.py
   ```

5. Press `q` to exit the webcam window.

## Output

- Live camera feed with detected faces and predicted emotion label.
- Works for multiple people simultaneously.

## Accuracy

- ~65% validation accuracy on the FER-2013 dataset.
- Can be improved with better architecture and data augmentation.

## Applications

- Mental health analysis
- User feedback detection
- Smart classrooms
- Customer experience tracking
- Human-computer interaction

## Future Scope

- Integrate pre-trained models like VGG or ResNet
- Add data augmentation
- Streamlit/Flask web interface
- Deploy as mobile or browser-based application

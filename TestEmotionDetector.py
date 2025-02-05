import cv2
import numpy as np
import os
from keras.models import model_from_json

# Dictionary for emotion labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Load the model safely
try:
    model_json_path = r"C:\Users\vanga\Downloads\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main\model\emotion_model.json"
    model_weights_path = r"C:\Users\vanga\Downloads\Emotion_detection_with_CNN\Emotion_detection_with_CNN-main\model\emotion_model.h5"

    if not os.path.exists(model_json_path) or not os.path.exists(model_weights_path):
        raise FileNotFoundError("Error: Model files not found!")

    # Load the model architecture
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    emotion_model = model_from_json(loaded_model_json)

    # Load model weights
    emotion_model.load_weights(model_weights_path)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

# Load OpenCV's pre-trained face detector (more reliable)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if the cascade classifier loaded correctly
if face_detector.empty():
    print("❌ Error: Could not load Haar cascade classifier.")
    exit()

print("🎥 Starting real-time emotion detection...")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handle empty frame issue
    if not ret or frame is None:
        print("❌ Error: Could not capture frame from camera. Exiting...")
        break

    # Resize the frame for better visualization
    frame = cv2.resize(frame, (1280, 720))

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Extract the region of interest
        roi_gray = gray_frame[y:y + h, x:x + w]

        # Preprocess the image for the model
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # Display emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the output frame
    cv2.imshow('Emotion Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🛑 Emotion detection stopped.")

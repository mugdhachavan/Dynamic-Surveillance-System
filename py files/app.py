import cv2
import datetime
import time
from collections import deque
import streamlit as st
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov5s.pt")

# Function to perform object detection on the video frames
def object_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    # Initialize variables for object detection
    detection_thresh = 15
    de = deque([False] * detection_thresh, maxlen=detection_thresh)
    status = False
    initial_time = None
    patience = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLO to detect objects
        results = model(frame)

        # This function will return a boolean variable telling if someone was present or not
        detected, num_people = is_person_present(results)

        # Register the current detection status on our deque object
        de.appendleft(detected)

        # If we have consecutively detected a person 15 times then we are sure that someone is present
        if sum(de) >= detection_thresh and not status:
            status = True
            entry_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            st.write("Person detected, entry time:", entry_time)

        # If status is True but the person is not in the current frame
        if status and not detected:
            # Restart the patience timer only if the person has not been detected for a few frames so we are sure it wasn't a False positive
            if sum(de) > (detection_thresh / 2):
                if initial_time is None:
                    initial_time = time.time()
            elif initial_time is not None:
                # If the patience has run out and the person is still not detected then set the status to False
                if time.time() - initial_time >= patience:
                    exit_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    st.write("Person left, exit time:", exit_time)
                    status = False
                    initial_time = None

        # If significant amount of detections (more than half of detection_thresh) has occurred then we reset the Initial Time.
        elif status and sum(de) > (detection_thresh / 2):
            initial_time = None

        # Display the annotated image with bounding boxes
        annotated_image = frame.copy()
        for obj in results.boxes:
            rect = obj.rect
            label = obj.label
            score = obj.score
            cv2.rectangle(annotated_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.putText(annotated_image, f'{label} {score}', (rect[0], rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        # Resize the image to fit within the expected dimensions for Streamlit
        annotated_image_resized = cv2.resize(annotated_image, (640, 480))

        # Display the annotated video frame
        st.image(annotated_image_resized, caption="Annotated Video", use_column_width=True)

        # Display the number of people detected
        st.write('Number of People:', num_people)

        # Display the room status
        st.write('Room Status:', 'Occupied' if status else 'Empty')

    # Release the video capture object
    cap.release()

# Function to check if a person is present in the frame
def is_person_present(results, confidence_threshold=0.5):
    detected = False
    num_people = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract class and confidence
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Check if the detected object is a person and confidence exceeds threshold
            if cls == 0 and conf >= confidence_threshold:  # Person class index is 0
                detected = True
                num_people += 1  # Increment the number of people detected

    return detected, num_people

def main():
    st.title("Real-time Object Detection on Uploaded Video")

    # Upload the video file
    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video file locally
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        # Perform object detection on the uploaded video
        object_detection("uploaded_video.mp4")

if __name__ == "__main__":
    main()

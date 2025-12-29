import cv2
import datetime
import time
from collections import deque
from ultralytics import YOLO
from twilio.rest import Client

# Load YOLO model
model = YOLO("../YoloWeights/yolov8l.pt")

# Initialize Twilio client
account_sid = 'ACa4497c886eeb4cf9f4095b58dc6326ca'
auth_token = 'd4301320704cfa707b57c87625047a41'
twilio_phone_number = '+13344633907'
client = Client(account_sid, auth_token)

# Function to send SMS using Twilio
# Function to send SMS using Twilio
def send_sms(message, to):
    try:
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=to
        )
        print("SMS sent successfully! SID:", message.sid)
    except Exception as e:
        print("Failed to send SMS:", str(e))


# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def is_person_present(frame, results, confidence_threshold=0.5):
    detected = False
    annotated_image = frame.copy()

    # Iterate through YOLO detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract class and confidence
            cls = int(box.cls[0])
            conf = box.conf[0]

            # Check if the detected object is a person and confidence exceeds threshold
            if classNames[cls] == 'person' and conf >= confidence_threshold:
                detected = True

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, 'Person Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    return detected, annotated_image

# Initialize the video capture from the camera
cap = cv2.VideoCapture(1)

# Initialize these variables for calculating FPS
fps = 0
frame_counter = 0
start_time = time.time()

# Status is True when person is present and False when the person is not present.
status = False

# We don't consider an initial detection unless it's detected 15 times, this gets rid of false positives
detection_thresh = 15

# Initial time for calculating if patience time is up
initial_time = None

# After the person disappears from view, wait at least 7 seconds before making the status False
patience = 7

# We are creating a deque object of length detection_thresh and will store individual detection statuses here
de = deque([False] * detection_thresh, maxlen=detection_thresh)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect objects
    results = model(frame, stream=True)

    # This function will return a boolean variable telling if someone was present or not
    detected, annotated_image = is_person_present(frame, results)

    # Register the current detection status on our deque object
    de.appendleft(detected)

    # If we have consecutively detected a person 15 times then we are sure that someone is present
    if sum(de) >= detection_thresh and not status:
        status = True
        entry_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("Person detected, entry time:", entry_time)

        # Send SMS alert
        message = "BE ALERT..!! \n Person detected at {}".format(entry_time)
        send_sms(message, to='+917223085234')

    # If status is True but the person is not in the current frame
    if status and not detected:
        # Restart the patience timer only if the person has not been detected for a few frames so we are sure it wasn't a False positive
        if sum(de) > (detection_thresh / 2):
            if initial_time is None:
                initial_time = time.time()
        elif initial_time is not None:
            # If the patience has run out and the person is still not detected then set the status to False
            if time.time() - initial_time >= patience:
                exit_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                print("Person left, exit time:", exit_time)
                status = False
                initial_time = None

    # If significant amount of detections (more than half of detection_thresh) has occurred then we reset the Initial Time.
    elif status and sum(de) > (detection_thresh / 2):
        initial_time = None

    # Calculate the Average FPS
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))

    # Display the FPS
    cv2.putText(annotated_image, 'FPS: {:.2f}'.format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the Room Status
    cv2.putText(annotated_image, 'Area Occupied: {}'.format(str(status)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the Date and Time
    current_datetime = datetime.datetime.now().strftime("%A, %d %B %Y %I:%M:%S %p")
    cv2.putText(annotated_image, current_datetime, (10, annotated_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8 , (0, 255, 255), 2)

    # Display the frame with annotations
    cv2.imshow('frame', annotated_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Capture and destroy windows
cap.release()
cv2.destroyAllWindows()

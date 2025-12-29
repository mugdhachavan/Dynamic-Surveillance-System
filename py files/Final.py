import cv2
import datetime
import time
from collections import deque
from ultralytics import YOLO
from twilio.rest import Client

# Load YOLO model
model = YOLO("D:\PycharmProjects\MiniProject\dynamic3.pt")

# Initialize Twilio client
account_sid = 'ACa4497c886eeb4cf9f4095b58dc6326ca'
auth_token = 'd4301320704cfa707b57c87625047a41'
twilio_phone_number = '+13344633907'
client = Client(account_sid, auth_token)

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
classNames = ["person"]

def is_person_present(frame, results, confidence_threshold=0.5):
    detected = False
    annotated_image = frame.copy()
    num_people = 0  # Counter for number of people detected

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
                num_people += 1  # Increment the number of people detected

                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, 'Person Detected', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    return detected, annotated_image, num_people

# Initialize the video capture from the camera
cap = cv2.VideoCapture("D:\PycharmProjects\MiniProject\images\sample_video.mp4")

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
patience = 10

# We are creating a deque object of length detection_thresh and will store individual detection statuses here
de = deque([False] * detection_thresh, maxlen=detection_thresh)

# Define the output path for saving the recorded video
output_path = "/Recordings/"

# Initialize the video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO to detect objects
    results = model(frame, stream=True)

    # This function will return a boolean variable telling if someone was present or not
    detected, annotated_image, num_people = is_person_present(frame, results)

    # Register the current detection status on our deque object
    de.appendleft(detected)

    # If we have consecutively detected a person 15 times then we are sure that someone is present
    if sum(de) >= detection_thresh and not status:
        status = True
        entry_time = datetime.datetime.now().strftime("Date: %d-%m-%Y \n Time: %H-%M-%S")
        print("Person detected, entry time:", entry_time)
        out = cv2.VideoWriter(output_path + 'recorded_video_{}.mp4'.format(entry_time), fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        # Send SMS alert
        message = "BE ALERT..!! \n Person detected at {}".format(entry_time)
        send_sms(message, to='+917223085234')  # Replace the 'to' number with your desired recipient's phone number

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
                print("Person left, exit time:", exit_time)
                status = False
                initial_time = None

                # Release the video writer object only when a new person is detected
                out.release()
                out = None

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

    # Display the Number of People
    cv2.putText(annotated_image, 'Number of People: {}'.format(num_people), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the Date and Time
    current_datetime = datetime.datetime.now().strftime("%A, %d %B %Y %I:%M:%S %p")
    cv2.putText(annotated_image, current_datetime, (10, annotated_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8 , (0, 255, 255), 2)

    # Calculate the remaining patience time
    if initial_time is not None:
        remaining_patience = max(0, patience - (time.time() - initial_time))
    else:
        remaining_patience = patience

    # Display the Patience Timer
    cv2.putText(annotated_image, 'Patience: {:.1f}'.format(remaining_patience), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Write the frame to the video if the room is occupied and recording is active
    if status and out is not None:
        out.write(annotated_image)

    # Show the Frame
    cv2.imshow('frame', annotated_image)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Capture and destroy windows
cap.release()
cv2.destroyAllWindows()

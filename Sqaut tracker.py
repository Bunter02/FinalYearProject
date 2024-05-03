import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load video from file
video_file_path = 'Videos/Deadlift 5.mp4'
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print(f"Error opening video file {video_file_path}")
    exit(1)

# Get the frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the delay in milliseconds
delay = int(1000 / (2 * fps))

# Initialize pose estimation
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the coordinates of the knees
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                if landmark == mp_pose.PoseLandmark.LEFT_KNEE:
                    print(f'Left knee coordinates: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})')
                elif landmark == mp_pose.PoseLandmark.RIGHT_KNEE:
                    print(f'Right knee coordinates: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})')

        # Display the image
        cv2.imshow('MediaPipe Pose', image)

        # Break the loop if 'q' is pressed, otherwise wait for the specified delay
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Load video from file
video_file_path = 'Videos/Bench 3.mp4'
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print(f"Error opening video file {video_file_path}")
    exit(1)


# Get the frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the delay in milliseconds
delay = int(1000 / (2 * fps))


# Create a hands object
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Initialize previous wrist coordinates
    prev_wrist_x, prev_wrist_y, prev_wrist_z = None, None, None

      # Initialize lists to store differences in wrist coordinates
    wrist_diffs_x = []
    wrist_diffs_y = []
    wrist_diffs_z = []

    while True:
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe
        results = hands.process(image)

        # Draw landmarks on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            print("Hand landmarks detected")  # Debugging line
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist coordinates
                wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = wrist_landmark.x * image.shape[1]
                wrist_y = wrist_landmark.y * image.shape[0]
                wrist_z = wrist_landmark.z

                # Print wrist coordinates
                print(f'Wrist coordinates: ({wrist_x}, {wrist_y}, {wrist_z})')

              
               # If previous wrist coordinates are available, calculate the difference and add it to the lists
                if prev_wrist_x is not None:
                    wrist_diffs_x.append(wrist_x - prev_wrist_x)
                    wrist_diffs_y.append(wrist_y - prev_wrist_y)
                    wrist_diffs_z.append(wrist_z - prev_wrist_z)
                # Update previous wrist coordinates
                prev_wrist_x, prev_wrist_y, prev_wrist_z = wrist_x, wrist_y, wrist_z
        else:
            print("No hand landmarks detected")  # Debugging line

        # Display the image
        cv2.imshow('MediaPipe Hands', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Calculate the average differences
avg_diff_x = np.mean(np.diff(wrist_diffs_x))
avg_diff_y = np.mean(np.diff(wrist_diffs_y))

# Print the average differences
print(f'Average X difference: {avg_diff_x}')
print(f'Average Y difference: {avg_diff_y}')

# Compare the averages and print the appropriate message
if abs(avg_diff_x - avg_diff_y) <= 0.1:
    print('Both sides are moving at the same speed.')
elif avg_diff_x > avg_diff_y:
    print('The right side is moving too fast.')
else:
    print('The left side is moving too fast.')
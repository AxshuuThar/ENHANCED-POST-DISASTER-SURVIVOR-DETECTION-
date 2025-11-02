import cv2
import os
from ultralytics import YOLO
from tkinter import Tk, filedialog
import time

# Load the trained YOLOv8 model (ensure the model is in the same directory or provide full path)
model = YOLO(r"C:\Users\ebins\MiniPro\code\runs\detect\train2\weights\best.pt")  # Provide path to your trained model

# Create 'result' directory if it doesn't exist
os.makedirs('result', exist_ok=True)

# Function to generate unique filenames
def get_unique_filename(base_name, extension):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Current time as a unique identifier
    return f"{base_name}_{timestamp}{extension}"

# Function to process images or videos
def process_input(input_path):
    # Check if the input is a video or an image
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # If it's a video, process it frame by frame
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        result_video_path = os.path.join('result', get_unique_filename(base_name, '.avi'))
        out = cv2.VideoWriter(result_video_path, fourcc, 30.0, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run object detection on the frame
            results = model(frame)  # Detect objects

            # Check if results are found and render them
            if results:
                frame = results[0].plot()  # Render the bounding boxes and labels on the frame

            # Write the frame with detections to the output video
            out.write(frame)

            # Display the frame
            cv2.imshow('Result', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved at {result_video_path}")

    elif input_path.lower().endswith(('.jpg', '.png', '.jpeg', '.webp', '.avif')):
        # If it's an image, process it once
        img = cv2.imread(input_path)
        results = model(img)  # Detect objects

        # Check if results are found and render them
        if results:
            img = results[0].plot()  # Render the bounding boxes and labels on the image

        # Generate a unique filename for the output image
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        result_image_path = os.path.join('result', get_unique_filename(base_name, '.jpg'))

        # Save the image
        cv2.imwrite(result_image_path, img)
        print(f"Image saved at {result_image_path}")

        # Display the image with the detected survivors
        cv2.imshow('Result', img)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    else:
        print("Unsupported file format. Please upload an image or video.")

# Function to let the user select a file using a file dialog
def select_file():
    Tk().withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select an Image or Video", 
        filetypes=[
            ("All Supported Files", "*.jpg;*.jpeg;*.png;*.webp;*.avif;*.mp4;*.avi;*.mov;*.mkv"),
            ("Images", "*.jpg;*.jpeg;*.png;*.webp;*.avif"),
            ("Videos", "*.mp4;*.avi;*.mov;*.mkv")
        ]
    )
    
    # Debugging: Print the file path
    print(f"Selected file: {file_path}")
    
    if file_path:
        process_input(file_path)
    else:
        print("No file selected.")

if __name__ == "__main__":
    select_file()

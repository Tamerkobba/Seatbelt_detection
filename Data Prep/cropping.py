import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

def save_cropped_images(video_path, model_path, output_dir, frame_skip=1, resize_dim=None, windshield_class_id=0):
    """
    Process a video and save cropped images from YOLO model detection to a specified directory.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLO model weights file.
        output_dir (str): Directory to save cropped images.
        frame_skip (int): Number of frames to skip between processing.
        resize_dim (tuple): Resize dimensions for the video frames (width, height).
        windshield_class_id (int): Class ID for the windshield detection.
    """
    # Load the YOLO model
    yolo_model = YOLO(model_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    cropped_count = 0

    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (frame_skip + 1) == 0:
                if resize_dim:
                    frame = cv2.resize(frame, resize_dim)

                # Perform YOLO model prediction
                results = yolo_model.predict(source=frame, verbose=False)
                boxes = results[0].boxes

                # Save cropped images
                for box in boxes:
                    if int(box.cls[0]) == windshield_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_image = frame[y1:y2, x1:x2]

                        # Resize the cropped image to 224x224
                        resized_image = cv2.resize(cropped_image, (640, 640))

                        # Save the resized cropped image
                        crop_filename = os.path.join(output_dir, f"croed_{cropped_count:05d}.jpg")
                        cv2.imwrite(crop_filename, resized_image)
                        cropped_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"Total cropped images saved: {cropped_count}")
    print(f"Cropped images saved in directory: {output_dir}")

if __name__ == "__main__":
    input_video = ".mp4"
    yolo_model_weights = "../Models/runs/train/weights/best.pt"  # Update with the correct path
    output_cropped_dir = "cropped_images"  # Directory to save cropped images
    frame_skip = 3
    resize_dimensions = (1280,720)

    save_cropped_images(
        video_path=input_video,
        model_path=yolo_model_weights,
        output_dir=output_cropped_dir,
        frame_skip=frame_skip,
        resize_dim=resize_dimensions,
        windshield_class_id=0
    )

import cv2
import os
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm


# Function to preprocess and classify the cropped windshield image using densenet12118
def classify_with_densenet121(cropped_image, model, device,threshold=0.3):
    """
    Classify the cropped image (windshield) using densenet12118 model.
    Args:
        cropped_image (ndarray): Cropped windshield image.
        model (torch.nn.Module): Pretrained densenet121 model.
        device (torch.device): Device (CPU or GPU) to run the model on.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(cropped_image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Use softmax to get probabilities
        not_wearing_prob = probabilities[0][1].item()  # Probability of 'Not Wearing Seatbelt'

        # Apply threshold
    return 1 if not_wearing_prob >= threshold else 0


import time

def process_video(video_path, model_path, densenet121_model_path, output_path, frame_skip=1, resize_dim=None,
                  windshield_class_id=0):
    """
    Process a video with YOLO object detection and classify seatbelt usage on windshields.
    Args:
        video_path (str): Path to input video.
        model_path (str): Path to YOLO model weights.
        densenet121_model_path (str): Path to densenet121 model weights.
        output_path (str): Path for output video.
        frame_skip (int): Number of frames to skip between processing.
        resize_dim (tuple): Optional (width, height) to resize frames.
        windshield_class_id (int): The class ID for the windshield in the YOLO model.
    """
    try:
        yolo_model = YOLO(model_path)
        densenet121_model = models.densenet121()
        num_features = densenet121_model.classifier.in_features
        densenet121_model.classifier = torch.nn.Linear(num_features, 2)

        model_dict = torch.load(densenet121_model_path, map_location=torch.device('cpu'))
        state_dict = model_dict['state_dict'] if 'state_dict' in model_dict else model_dict
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        densenet121_model.load_state_dict(state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        densenet121_model.to(device)
        densenet121_model.eval()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if resize_dim:
            width, height = resize_dim

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processing_times = []  # List to store processing times for each frame

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % (frame_skip + 1) == 0:
                    start_time = time.time()  # Start timing

                    if resize_dim:
                        frame = cv2.resize(frame, resize_dim)

                    results = yolo_model.predict(source=frame, verbose=False)
                    boxes = results[0].boxes

                    for box in boxes:
                        if int(box.cls[0]) == windshield_class_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            windshield_crop = frame[y1:y2, x1:x2]

                            predicted_class = classify_with_densenet121(windshield_crop, densenet121_model, device)
                            label = "Wearing Seatbelt" if predicted_class == 0 else "Not Wearing Seatbelt"

                            cv2.rectangle(frame, (x1, y1), (x2, y2),
                                          (0, 255, 0) if predicted_class == 0 else (0, 0, 255), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0) if predicted_class == 0 else (0, 0, 255), 2)

                    end_time = time.time()  # End timing
                    frame_time = end_time - start_time
                    processing_times.append(frame_time)
                    out.write(frame)

                frame_count += 1
                pbar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Calculate and print average processing time
        average_time = np.mean(processing_times)
        print(f"\nAverage processing time per frame: {average_time:.4f} seconds")
        print(f"Video processing completed. Output saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        raise


if __name__ == "__main__":
    input_video = "Video Test/1.mp4"
    yolo_model_weights = "Models/runs/train/weights/best.pt" #put global path
    densenet121_model_weights = "Models/DenseNet.pth.tar"
    output_video = "output_video_with_labels_4.mp4"
    frame_skip = 0
    resize_dimensions = (1280, 720)

    process_video(
        video_path=input_video,
        model_path=yolo_model_weights,
        densenet121_model_path=densenet121_model_weights,
        output_path=output_video,
        frame_skip=frame_skip,
        resize_dim=resize_dimensions,
        windshield_class_id=0
    )

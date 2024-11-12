import cv2
import os
from ultralytics import YOLO
import torch
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm


# Function to preprocess and classify the cropped windshield image using ResNet18
def classify_with_resnet(cropped_image, model, device):
    """
    Classify the cropped image (windshield) using ResNet18 model.
    Args:
        cropped_image (ndarray): Cropped windshield image.
        model (torch.nn.Module): Pretrained ResNet model.
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

    _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()


def process_video(video_path, model_path, resnet_model_path, output_path, frame_skip=1, resize_dim=None,
                  windshield_class_id=0):
    """
    Process a video with YOLO object detection and classify seatbelt usage on windshields.
    Args:
        video_path (str): Path to input video.
        model_path (str): Path to YOLO model weights.
        resnet_model_path (str): Path to ResNet model weights.
        output_path (str): Path for output video.
        frame_skip (int): Number of frames to skip between processing.
        resize_dim (tuple): Optional (width, height) to resize frames.
        windshield_class_id (int): The class ID for the windshield in the YOLO model.
    """
    try:
        yolo_model = YOLO(model_path)
        resnet_model = models.resnet18()
        num_features = resnet_model.fc.in_features
        resnet_model.fc = torch.nn.Linear(num_features, 2)

        model_dict = torch.load(resnet_model_path, map_location=torch.device('cpu'))
        state_dict = model_dict['state_dict'] if 'state_dict' in model_dict else model_dict
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        resnet_model.load_state_dict(state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet_model.to(device)
        resnet_model.eval()

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
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % (frame_skip + 1) == 0:
                    if resize_dim:
                        frame = cv2.resize(frame, resize_dim)

                    results = yolo_model.predict(source=frame, verbose=False)
                    boxes = results[0].boxes

                    for box in boxes:
                        if int(box.cls[0]) == windshield_class_id:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            windshield_crop = frame[y1:y2, x1:x2]

                            predicted_class = classify_with_resnet(windshield_crop, resnet_model, device)
                            label = "Wearing Seatbelt" if predicted_class == 1 else "Not Wearing Seatbelt"

                            cv2.rectangle(frame, (x1, y1), (x2, y2),
                                          (0, 255, 0) if predicted_class == 1 else (0, 0, 255), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0) if predicted_class == 1 else (0, 0, 255), 2)

                    out.write(frame)

                frame_count += 1
                pbar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\nVideo processing completed. Output saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()
        raise


if __name__ == "__main__":
    input_video = "Traffic IP Camera video.mp4"
    yolo_model_weights = "runs/detect/train/weights/best.pt"
    resnet_model_weights = "checkpoint.pth.tar"
    output_video = "output_video_with_labels.mp4"
    frame_skip = 1
    resize_dimensions = (1280, 720)

    process_video(
        video_path=input_video,
        model_path=yolo_model_weights,
        resnet_model_path=resnet_model_weights,
        output_path=output_video,
        frame_skip=frame_skip,
        resize_dim=resize_dimensions,
        windshield_class_id=0
    )

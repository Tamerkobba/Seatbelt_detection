from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import torch
from ultralytics import YOLO
from torchvision import models, transforms
import numpy as np
import logging
import base64

# Initialize FastAPI app
app = FastAPI(
    title="Video Frame Processor with WebSocket",
    description="WebSocket API to process video frames using YOLO and DenseNet in real-time.",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------
# CORS Configuration
# --------------------

origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Global Variables
# --------------------

yolo_model = None
densenet_model = None
device = None

# --------------------
# Model Loading
# --------------------

@app.on_event("startup")
def load_models():
    global yolo_model, densenet_model, device

    try:
        logger.info("Loading YOLO model...")
        yolo_model = YOLO("Models/runs/train/weights/best.pt").to("cuda")
        logger.info("YOLO model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        raise e

    try:
        logger.info("Loading DenseNet model...")
        densenet_model = models.densenet121()
        num_features = densenet_model.classifier.in_features
        densenet_model.classifier = torch.nn.Linear(num_features, 2)
        densenet_weights = torch.load("Models/DenseNet.pth.tar", map_location="cuda")
        densenet_model.load_state_dict(densenet_weights["state_dict"])
        densenet_model.eval().to("cuda")
        logger.info("DenseNet model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading DenseNet model: {e}")
        raise e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Models loaded on device: {device}")


# --------------------
# Helper Functions
# --------------------

def classify_with_densenet121(cropped_image, model, device, threshold=0.3):
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(cropped_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            not_wearing_prob = probabilities[0][1].item()
        return 1 if not_wearing_prob >= threshold else 0
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return -1


# --------------------
# WebSocket Endpoint
# --------------------

@app.websocket("/ws/process-frame")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()
            np_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text("Error: Invalid image data.")
                continue

            # YOLO Detection
            results = yolo_model.predict(source=frame, verbose=False)
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])

                # Assuming class_id == 0 corresponds to 'Windshield'
                if class_id == 0:
                    logger.debug(f"Processing bounding box: ({x1}, {y1}, {x2}, {y2})")
                    color = (0, 255, 0)
                    label = "Object"
                    cropped = frame[y1:y2, x1:x2]

                    if cropped.size > 0:
                        # DenseNet Classification
                        classification_result = classify_with_densenet121(cropped, densenet_model, device)
                        if classification_result == -1:
                            label = "Error"
                            color = (0, 255, 255)
                        elif classification_result == 0:
                            label = "Wearing Seatbelt"
                            color = (0, 255, 0)
                        else:
                            label = "Not Wearing Seatbelt"
                            color = (0, 0, 255)

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode processed frame
            success, encoded_img = cv2.imencode('.jpg', frame)
            if not success:
                await websocket.send_text("Error: Failed to encode image.")
                continue

            # Send processed frame back to client
            encoded_bytes = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
            await websocket.send_text(encoded_bytes)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.close()

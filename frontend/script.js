const processedCanvas = document.getElementById("processedCanvas");
const uploadButton = document.getElementById("uploadButton");
const startButton = document.getElementById("startButton");
const videoInput = document.getElementById("videoInput");

const ctx = processedCanvas.getContext("2d");

let processing = false;
let ws = null; // WebSocket connection
const FRAME_SKIP_INTERVAL = 2; // Frame skipping interval
let currentFrame = 0; // Counter to track frames
let videoElement = null; // Temporary video element for processing frames

// Event listener for the upload button
uploadButton.addEventListener("click", () => {
    videoInput.click(); // Trigger hidden file input
});

// Event listener for file selection
videoInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        const url = URL.createObjectURL(file);

        // Create a temporary video element for frame extraction
        videoElement = document.createElement("video");
        videoElement.src = url;
        videoElement.play();

        startButton.disabled = false; // Enable the start button
    }
});

// Event listener for the start button
startButton.addEventListener("click", () => {
    if (videoElement && videoElement.readyState >= 2 && !ws) {
        ws = new WebSocket("ws://127.0.0.1:9000/ws/process-frame"); // Establish WebSocket connection

        ws.onopen = () => {
            console.log("WebSocket connection established.");
            processing = true;
            processFrames();
            startButton.disabled = true;
        };

        ws.onmessage = (event) => {
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height); // Clear the previous frame
                ctx.drawImage(img, 0, 0, processedCanvas.width, processedCanvas.height);
            };
            img.src = `data:image/jpeg;base64,${event.data}`; // Display processed frame
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed.");
            ws = null; // Reset WebSocket connection
        };

        videoElement.addEventListener("ended", stopProcessing); // Stop processing when video ends
    }
});

// Function to process video frames
async function processFrames() {
    while (processing && videoElement && !videoElement.paused && !videoElement.ended) {
        currentFrame++;

        // Skip frames based on FRAME_SKIP_INTERVAL
        if (currentFrame % (FRAME_SKIP_INTERVAL + 1) !== 0) {
            await new Promise((resolve) => setTimeout(resolve, 1000 / 30)); // Wait for ~30 FPS
            continue;
        }

        // Capture the current frame from the video
        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const videoCtx = canvas.getContext("2d");
        videoCtx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        // Convert the frame to a Blob
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));

        if (blob && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(blob); // Send the frame to the WebSocket server
        }

        await new Promise((resolve) => setTimeout(resolve, 1000 / 30)); // Wait for ~30 FPS
    }

    if (videoElement.ended) {
        console.log("Video has ended. Stopping processing.");
        stopProcessing();
    }
}

// Function to stop processing
function stopProcessing() {
    console.log("Stopping processing...");
    processing = false;

    if (ws) {
        ws.close(); // Close WebSocket connection
        ws = null;
    }

    startButton.disabled = false; // Re-enable start button for next video
    videoElement.removeEventListener("ended", stopProcessing); // Remove event listener
    videoElement = null; // Remove the temporary video element
}

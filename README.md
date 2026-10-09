# Microscope Control Station

Full camera control interface for the **Arducam IMX519 16MP** camera on a Raspberry Pi 5, designed for real-time microscopy viewing and capture.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-red) ![Camera](https://img.shields.io/badge/Camera-IMX519%2016MP-green)

## Features

### 🔴 Live Stream
- Real-time MJPEG stream at 30fps via web browser
- **Click-to-focus**: click any point on the stream to autofocus the lens
- Spot sharpness meter with peak tracking

### 📷 Full Camera Control
Every hardware control is exposed in the UI and via REST API:

| Control | Range | Description |
|---|---|---|
| **Focus** (LensPosition) | 0–32 dpt | Manual focus with slider + nudge buttons |
| **Exposure Time** | 1–66,666 µs | Logarithmic slider, or auto via AE |
| **Analogue Gain** (ISO) | 1–16× | Sensor amplification |
| **EV Compensation** | ±8.0 | Quick brightness adjust with AE on |
| **Brightness** | -1.0 to +1.0 | Output brightness offset |
| **Contrast** | 0–4× | Image contrast |
| **Saturation** | 0–4× | Color intensity |
| **Sharpness** (ISP) | 0–8× | Hardware sharpening |
| **White Balance** | 7 presets + manual R/B gains | AWB or manual colour tuning |
| **Noise Reduction** | Off / Fast / HQ / Minimal / ZSL | ISP denoising |
| **HDR Mode** | Off / Single / Multi / Night | High dynamic range |
| **Digital Zoom** | 1–4× | Sensor crop zoom |
| **Resolution** | 480p / 720p / 1080p | Stream resolution switching |

### 📸 Capture & Recording
- **16MP still capture** (4656×3496) at current focus/exposure settings
- **Video recording** to MP4 via ffmpeg
- All captures saved to the microscope directory

### 🖼️ Gallery
- Browse all captured photos and videos in a grid with thumbnails
- Sort by date, size, or name; filter by type
- Lightbox viewer, download, and delete

### 🤖 AI Agent API
Fully controllable over the network (e.g. via Tailscale) for automated microscopy:

```bash
# Discover camera capabilities
curl http://<IP>:1306/api/camera_info

# Read all current settings
curl http://<IP>:1306/api/controls

# Set any control
curl "http://<IP>:1306/api/set_control?name=ExposureTime&value=5000"

# Grab a frame
curl http://<IP>:1306/snapshot.jpg -o frame.jpg

# List captured media
curl http://<IP>:1306/api/media

# Trigger 16MP capture
curl http://<IP>:1306/api/capture_16mp

# Start/stop video recording
curl http://<IP>:1306/api/record_start
curl http://<IP>:1306/api/record_stop
```

## Setup

### Hardware
- Raspberry Pi 5
- [Arducam IMX519 16MP Autofocus Camera](https://www.arducam.com/product/imx519-autofocus-camera-module-for-raspberry-pi/)
- DIY microscope with camera mounted on eyepiece

### Software Requirements

```bash
# Install system dependencies (Raspberry Pi OS)
sudo apt install -y python3-picamera2 python3-libcamera python3-pil python3-numpy ffmpeg

# Install any additional Python packages
pip install -r requirements.txt
```

### Running

```bash
python3 stream_server.py
```

The server starts on **port 1306**. Open `http://<your-pi-ip>:1306` in any browser.

### Auto-start on Boot (optional)

```bash
# Create a systemd service
sudo tee /etc/systemd/system/microscope.service << 'EOF'
[Unit]
Description=Microscope Control Station
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/thomas/microscope/stream_server.py
WorkingDirectory=/home/thomas/microscope
User=thomas
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable microscope
sudo systemctl start microscope
```

## Architecture

```
stream_server.py          # Single-file server (Python + embedded HTML/CSS/JS)
├── Picamera2             # Camera interface (video stream + still capture)
├── LibavMjpegEncoder     # MJPEG encoding at 30fps
├── StreamingOutput       # Thread-safe frame buffer with sharpness analysis
├── ThreadingHTTPServer   # HTTP server on :1306
│   ├── /                 # Web UI (Live Stream + Gallery tabs)
│   ├── /stream.mjpg      # MJPEG stream endpoint
│   ├── /snapshot.jpg      # Single frame grab
│   ├── /api/*             # REST API for all controls
│   └── /media/*           # Media file serving
└── ffmpeg subprocess     # Video recording pipeline
```

## License

MIT

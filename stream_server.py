#!/usr/bin/env python3
"""
Microscope Precision Stream Server — Full Camera Control
Arducam IMX519 16MP via Picamera2 on Raspberry Pi 5
All hardware controls exposed for human and AI-agent use.
"""

import io
import json
import logging
import mimetypes
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote, quote

import numpy as np
from PIL import Image
from picamera2 import Picamera2
from picamera2.encoders import LibavMjpegEncoder
from picamera2.outputs import FileOutput
from libcamera import controls, Transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
mimetypes.add_type("font/woff2", ".woff2")

# ═══════════════════════════════════════════════════════════════
# Streaming Output Buffer
# ═══════════════════════════════════════════════════════════════

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()
        self.frame_count = 0
        self.count = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.sharpness = 0.0
        self.peak_sharpness = 0.0
        self._analyze_counter = 0
        self.spot_x = 0.5
        self.spot_y = 0.5

    def write(self, buf):
        with self.condition:
            self.frame = bytes(buf)
            self.count += 1
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now
            self._analyze_counter += 1
            if self._analyze_counter >= 6:
                self._analyze_counter = 0
                threading.Thread(target=self._calc_sharpness, args=(self.frame, self.spot_x, self.spot_y), daemon=True).start()
            self.condition.notify_all()
        return len(buf)

    def _calc_sharpness(self, buf, sx, sy):
        try:
            im = Image.open(io.BytesIO(buf)).convert("L")
            arr = np.array(im, dtype=np.float32)
            h, w = arr.shape
            cx, cy = int(sx * w), int(sy * h)
            half = 50
            x1, x2 = max(0, cx - half), min(w, cx + half)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            roi = arr[y1:y2, x1:x2]
            if roi.shape[0] > 10 and roi.shape[1] > 10:
                lap = roi[1:-1, 2:] + roi[1:-1, :-2] + roi[2:, 1:-1] + roi[:-2, 1:-1] - 4.0 * roi[1:-1, 1:-1]
                score = float(np.var(lap))
                self.sharpness = round(score, 1)
                if score > self.peak_sharpness:
                    self.peak_sharpness = round(score, 1)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════

output = StreamingOutput()
picam2 = None
camera_lock = threading.RLock()  # RLock: re-entrant so apply_all_controls() can be called while lock is held
current_focus_pos = 1.60
latest_capture_path = None
recording_active = False
recording_proc = None
recording_thread = None
recording_filename = None
current_resolution = (2328, 1748)
current_qp = 10  # MJPEG quality: lower=better image, higher=faster stream (3-31)

MEDIA_DIR = "/home/thomas/microscope"
THUMB_DIR = os.path.join(MEDIA_DIR, ".thumbs")
MEDIA_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov'}

# Tracks user-set control values
control_state = {
    "AeEnable": True,
    "ExposureTime": 20000,
    "AnalogueGain": 1.0,
    "ExposureValue": 0.0,
    "AeMeteringMode": "CentreWeighted",
    "AeExposureMode": "Normal",
    "AwbEnable": True,
    "AwbMode": "Auto",
    "ColourGainRed": 2.0,
    "ColourGainBlue": 2.0,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 1.0,
    "Sharpness": 1.0,
    "NoiseReductionMode": 0,
    "HdrMode": 0,
    "DigitalZoom": 1.0,
}

AWB_MODES = {
    "Auto": controls.AwbModeEnum.Auto,
    "Incandescent": controls.AwbModeEnum.Incandescent,
    "Tungsten": controls.AwbModeEnum.Tungsten,
    "Fluorescent": controls.AwbModeEnum.Fluorescent,
    "Indoor": controls.AwbModeEnum.Indoor,
    "Daylight": controls.AwbModeEnum.Daylight,
    "Cloudy": controls.AwbModeEnum.Cloudy,
}
AE_METERING_MODES = {
    "CentreWeighted": controls.AeMeteringModeEnum.CentreWeighted,
    "Spot": controls.AeMeteringModeEnum.Spot,
    "Matrix": controls.AeMeteringModeEnum.Matrix,
}
AE_EXPOSURE_MODES = {
    "Normal": controls.AeExposureModeEnum.Normal,
    "Short": controls.AeExposureModeEnum.Short,
    "Long": controls.AeExposureModeEnum.Long,
}

FLOAT_CONTROLS = {"AnalogueGain", "ExposureValue", "Brightness", "Contrast", "Saturation", "Sharpness"}
INT_CONTROLS = {"ExposureTime", "NoiseReductionMode", "HdrMode"}
BOOL_CONTROLS = {"AeEnable", "AwbEnable"}
ENUM_CONTROLS = {
    "AwbMode": AWB_MODES,
    "AeMeteringMode": AE_METERING_MODES,
    "AeExposureMode": AE_EXPOSURE_MODES,
}


# ═══════════════════════════════════════════════════════════════
# Camera Control Helpers
# ═══════════════════════════════════════════════════════════════

def set_camera_control(name, value):
    """Set a single camera control by name. Returns result dict."""
    global current_focus_pos
    try:
        if name == "LensPosition":
            current_focus_pos = max(0.0, min(32.0, float(value)))
            ctrl = {"AfMode": controls.AfModeEnum.Manual, "LensPosition": current_focus_pos}
        elif name in BOOL_CONTROLS:
            v = str(value).lower() in ("true", "1", "on")
            control_state[name] = v
            ctrl = {name: v}
        elif name in FLOAT_CONTROLS:
            v = float(value)
            control_state[name] = v
            ctrl = {name: v}
        elif name in INT_CONTROLS:
            v = int(float(value))
            control_state[name] = v
            ctrl = {name: v}
        elif name in ENUM_CONTROLS:
            control_state[name] = value
            enum_val = ENUM_CONTROLS[name].get(value)
            if enum_val is None:
                return {"status": "error", "message": f"Invalid value '{value}' for {name}"}
            ctrl = {name: enum_val}
        elif name == "ColourGainRed":
            v = float(value)
            control_state["ColourGainRed"] = v
            ctrl = {"ColourGains": (v, control_state.get("ColourGainBlue", 2.0))}
        elif name == "ColourGainBlue":
            v = float(value)
            control_state["ColourGainBlue"] = v
            ctrl = {"ColourGains": (control_state.get("ColourGainRed", 2.0), v)}
        elif name == "DigitalZoom":
            v = max(1.0, min(4.0, float(value)))
            control_state["DigitalZoom"] = v
            props = picam2.camera_properties
            full_w, full_h = props.get("PixelArraySize", (4656, 3496))
            crop_w = int(full_w / v)
            crop_h = int(full_h / v)
            x = (full_w - crop_w) // 2
            y = (full_h - crop_h) // 2
            ctrl = {"ScalerCrop": (x, y, crop_w, crop_h)}
        else:
            return {"status": "error", "message": f"Unknown control: {name}"}

        with camera_lock:
            picam2.set_controls(ctrl)
        logging.info("🔧 [Control] %s = %s", name, value)
        return {"status": "ok", "name": name, "value": value}
    except Exception as e:
        logging.error("Control error for %s: %s", name, e)
        return {"status": "error", "message": str(e)}


def apply_all_controls():
    """Re-apply all stored controls (e.g. after resolution change)."""
    ctrls = {"AfMode": controls.AfModeEnum.Manual, "LensPosition": current_focus_pos}
    ctrls["AeEnable"] = control_state["AeEnable"]
    if not control_state["AeEnable"]:
        ctrls["ExposureTime"] = int(control_state["ExposureTime"])
        ctrls["AnalogueGain"] = float(control_state["AnalogueGain"])
    ctrls["ExposureValue"] = float(control_state["ExposureValue"])
    ctrls["Brightness"] = float(control_state["Brightness"])
    ctrls["Contrast"] = float(control_state["Contrast"])
    ctrls["Saturation"] = float(control_state["Saturation"])
    ctrls["Sharpness"] = float(control_state["Sharpness"])
    ctrls["NoiseReductionMode"] = int(control_state["NoiseReductionMode"])
    ctrls["AwbEnable"] = control_state["AwbEnable"]
    if control_state["AwbEnable"]:
        mode = AWB_MODES.get(control_state["AwbMode"], controls.AwbModeEnum.Auto)
        ctrls["AwbMode"] = mode
    else:
        ctrls["ColourGains"] = (control_state["ColourGainRed"], control_state["ColourGainBlue"])
    try:
        with camera_lock:
            picam2.set_controls(ctrls)
    except Exception as e:
        logging.warning("Could not restore all controls: %s", e)


def reset_controls_to_defaults():
    """Reset all controls to sensible defaults."""
    global current_focus_pos
    current_focus_pos = 1.60
    control_state.update({
        "AeEnable": True, "ExposureTime": 20000, "AnalogueGain": 1.0,
        "ExposureValue": 0.0, "AeMeteringMode": "CentreWeighted",
        "AeExposureMode": "Normal", "AwbEnable": True, "AwbMode": "Auto",
        "ColourGainRed": 2.0, "ColourGainBlue": 2.0, "Brightness": 0.0,
        "Contrast": 1.0, "Saturation": 1.0, "Sharpness": 1.0,
        "NoiseReductionMode": 0, "HdrMode": 0, "DigitalZoom": 1.0,
    })
    apply_all_controls()


# ═══════════════════════════════════════════════════════════════
# Autofocus
# ═══════════════════════════════════════════════════════════════

def get_fresh_frame(skip_frames=2):
    with output.condition:
        target = output.count + skip_frames
        while output.count < target:
            if not output.condition.wait(timeout=0.2):
                break
        return output.frame


def compute_roi_sharpness(buf, sx, sy, box_size=100):
    if not buf:
        return 0.0
    try:
        im = Image.open(io.BytesIO(buf)).convert("L")
        arr = np.array(im, dtype=np.float32)
        h, w = arr.shape
        cx, cy = int(sx * w), int(sy * h)
        half = box_size // 2
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        roi = arr[y1:y2, x1:x2]
        if roi.shape[0] > 10 and roi.shape[1] > 10:
            lap = roi[1:-1, 2:] + roi[1:-1, :-2] + roi[2:, 1:-1] + roi[:-2, 1:-1] - 4.0 * roi[1:-1, 1:-1]
            return float(np.var(lap))
    except Exception:
        pass
    return 0.0


def run_spot_autofocus(sx, sy, broad=False):
    global current_focus_pos
    with camera_lock:
        t0 = time.time()
        logging.info("🎯 [Click-to-Focus] Starting at (x=%.3f, y=%.3f)...", sx, sy)

        # Stage 1: Coarse sweep across full range
        if broad:
            coarse_pts = [float(x) for x in range(0, 21)]  # 0,1,2,...20
        else:
            coarse_pts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

        best_pos, best_score = current_focus_pos, -1.0
        scores = {}
        for p in coarse_pts:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(p)})
            f = get_fresh_frame(skip_frames=2)
            s = compute_roi_sharpness(f, sx, sy)
            scores[p] = s
            logging.info("  [AF] %.1f dpt → %.1f", p, s)
            if s > best_score:
                best_score, best_pos = s, p

        # Stage 2: Fine sweep ±1.5 dpt around best
        fine_pts = [round(best_pos + d * 0.3, 2) for d in range(-5, 6)]
        fine_pts = [p for p in fine_pts if 0.0 <= p <= 20.0 and p not in scores]
        for p in fine_pts:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(p)})
            f = get_fresh_frame(skip_frames=2)
            s = compute_roi_sharpness(f, sx, sy)
            scores[p] = s
            logging.info("  [AF fine] %.2f dpt → %.1f", p, s)
            if s > best_score:
                best_score, best_pos = s, p

        best_pos = float(round(best_pos, 2))
        current_focus_pos = best_pos
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": best_pos})
        output.spot_x, output.spot_y = sx, sy
        output.sharpness = round(best_score, 1)
        output.peak_sharpness = round(best_score, 1)

        duration = round(time.time() - t0, 2)
        logging.info("✅ [Autofocus] Best: %.2f dpt | Sharpness: %.1f | %.2fs", best_pos, best_score, duration)
        return {"status": "ok", "best_focus": best_pos, "score": round(best_score, 1), "duration": duration}


# ═══════════════════════════════════════════════════════════════
# Capture & Recording
# ═══════════════════════════════════════════════════════════════

def capture_highres_still():
    global current_focus_pos, latest_capture_path
    with camera_lock:
        t0 = time.time()
        filename = os.path.join(MEDIA_DIR, f"capture_16mp_{int(time.time())}.jpg")
        logging.info("📸 [16MP Still] Switching to full sensor mode...")

        picam2.stop_recording()
        picam2.stop()

        config_still = picam2.create_still_configuration(
            main={"size": (4656, 3496)},
            transform=Transform(hflip=True, vflip=True)
        )
        picam2.configure(config_still)
        picam2.start()
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": current_focus_pos})
        time.sleep(0.3)
        picam2.capture_file(filename)

        # Restore video streaming
        picam2.stop()
        config_video = picam2.create_video_configuration(
            main={"size": current_resolution},
            raw={"size": (2328, 1748)},
            transform=Transform(hflip=True, vflip=True)
        )
        picam2.configure(config_video)
        picam2.start()
        apply_all_controls()
        encoder = LibavMjpegEncoder(framerate=30, qp=current_qp)
        picam2.start_recording(encoder, FileOutput(output))

        duration = round(time.time() - t0, 2)
        file_size_mb = round(os.path.getsize(filename) / (1024 * 1024), 2)
        latest_capture_path = filename
        logging.info("✅ [16MP Still] Saved %s (%s MB) in %.2fs", filename, file_size_mb, duration)
        return {"status": "ok", "file": filename, "name": os.path.basename(filename),
                "size_mb": file_size_mb, "duration": duration}


def start_recording():
    global recording_active, recording_proc, recording_thread, recording_filename
    if recording_active:
        return {"status": "error", "message": "Already recording"}
    recording_filename = os.path.join(MEDIA_DIR, f"video_{int(time.time())}.mp4")
    recording_active = True
    recording_thread = threading.Thread(target=_recording_worker, daemon=True)
    recording_thread.start()
    logging.info("⏺ [Recording] Started → %s", recording_filename)
    return {"status": "ok", "file": recording_filename}


def _recording_worker():
    global recording_active, recording_proc
    try:
        recording_proc = subprocess.Popen([
            'ffmpeg', '-y', '-f', 'image2pipe', '-framerate', '30',
            '-vcodec', 'mjpeg', '-i', 'pipe:0',
            '-vcodec', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', recording_filename
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        while recording_active:
            with output.condition:
                if not output.condition.wait(timeout=0.1):
                    continue
                frame = output.frame
            if frame and recording_proc.poll() is None:
                try:
                    recording_proc.stdin.write(frame)
                except BrokenPipeError:
                    break
    except Exception as e:
        logging.error("Recording error: %s", e)
    finally:
        if recording_proc and recording_proc.poll() is None:
            try:
                recording_proc.stdin.close()
                recording_proc.wait(timeout=15)
            except Exception:
                recording_proc.kill()
        recording_active = False
        logging.info("⏹ [Recording] Finished → %s", recording_filename)


def stop_recording():
    global recording_active
    if not recording_active:
        return {"status": "error", "message": "Not recording"}
    recording_active = False
    if recording_thread:
        recording_thread.join(timeout=15)
    size_mb = 0
    if recording_filename and os.path.exists(recording_filename):
        size_mb = round(os.path.getsize(recording_filename) / (1024 * 1024), 2)
    return {"status": "ok", "file": recording_filename, "name": os.path.basename(recording_filename),
            "size_mb": size_mb}


def change_resolution(width, height):
    global current_resolution
    with camera_lock:
        logging.info("🖥️ [Resolution] Switching to %dx%d...", width, height)
        picam2.stop_recording()
        picam2.stop()
        config = picam2.create_video_configuration(
            main={"size": (width, height)},
            raw={"size": (2328, 1748)},
            transform=Transform(hflip=True, vflip=True)
        )
        picam2.configure(config)
        picam2.start()
        current_resolution = (width, height)
        apply_all_controls()
        encoder = LibavMjpegEncoder(framerate=30, qp=current_qp)
        picam2.start_recording(encoder, FileOutput(output))
    logging.info("✅ [Resolution] Now streaming at %dx%d (qp=%d)", width, height, current_qp)
    return {"status": "ok", "resolution": [width, height], "qp": current_qp}


def change_stream_quality(qp):
    """Change MJPEG quality by restarting encoder with new QP value."""
    global current_qp
    qp = max(2, min(31, qp))
    with camera_lock:
        logging.info("🎨 [Quality] Switching to qp=%d...", qp)
        picam2.stop_recording()
        picam2.stop()
        current_qp = qp
        picam2.start()
        encoder = LibavMjpegEncoder(framerate=30, qp=current_qp)
        picam2.start_recording(encoder, FileOutput(output))
    logging.info("✅ [Quality] Now streaming at qp=%d", current_qp)
    return {"status": "ok", "qp": current_qp}


def switch_stream_mode(width, height, qp):
    """Switch resolution and quality in a single camera cycle."""
    global current_resolution, current_qp
    qp = max(2, min(31, qp))
    with camera_lock:
        logging.info("🔄 [Mode] Switching to %dx%d qp=%d...", width, height, qp)
        picam2.stop_recording()
        picam2.stop()
        current_qp = qp
        config = picam2.create_video_configuration(
            main={"size": (width, height)},
            raw={"size": (2328, 1748)},
            transform=Transform(hflip=True, vflip=True)
        )
        picam2.configure(config)
        picam2.start()
        current_resolution = (width, height)
        apply_all_controls()
        encoder = LibavMjpegEncoder(framerate=30, qp=current_qp)
        picam2.start_recording(encoder, FileOutput(output))
    logging.info("✅ [Mode] Now streaming at %dx%d qp=%d", width, height, current_qp)
    return {"status": "ok", "resolution": [width, height], "qp": current_qp}


# ═══════════════════════════════════════════════════════════════
# Media / Gallery Helpers
# ═══════════════════════════════════════════════════════════════

def list_media_files():
    all_exts = MEDIA_EXTS | VIDEO_EXTS
    files = []
    for f in os.listdir(MEDIA_DIR):
        ext = os.path.splitext(f)[1].lower()
        if ext in all_exts:
            path = os.path.join(MEDIA_DIR, f)
            try:
                stat = os.stat(path)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "mtime": stat.st_mtime,
                    "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
                    "type": "video" if ext in VIDEO_EXTS else "image",
                })
            except OSError:
                pass
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files


def get_or_create_thumbnail(filename):
    os.makedirs(THUMB_DIR, exist_ok=True)
    thumb_name = f"thumb_{filename}.jpg"
    thumb_path = os.path.join(THUMB_DIR, thumb_name)
    src_path = os.path.join(MEDIA_DIR, filename)

    if not os.path.exists(src_path):
        return None

    # Use cached thumbnail if source hasn't changed
    if os.path.exists(thumb_path):
        if os.path.getmtime(src_path) <= os.path.getmtime(thumb_path):
            return thumb_path

    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext in MEDIA_EXTS:
            img = Image.open(src_path)
            img.thumbnail((320, 320))
            img.save(thumb_path, "JPEG", quality=75)
            return thumb_path
        elif ext in VIDEO_EXTS:
            subprocess.run([
                'ffmpeg', '-y', '-i', src_path, '-vframes', '1',
                '-vf', 'scale=320:-1', '-q:v', '5', thumb_path
            ], capture_output=True, timeout=10)
            if os.path.exists(thumb_path):
                return thumb_path
    except Exception as e:
        logging.warning("Thumbnail generation failed for %s: %s", filename, e)
    return None


def delete_media_file(filename):
    """Safely delete a media file and its thumbnail."""
    src = os.path.join(MEDIA_DIR, filename)
    real = os.path.realpath(src)
    if not real.startswith(os.path.realpath(MEDIA_DIR)):
        return {"status": "error", "message": "Invalid path"}
    if not os.path.exists(src):
        return {"status": "error", "message": "File not found"}
    try:
        os.remove(src)
        thumb = os.path.join(THUMB_DIR, f"thumb_{filename}.jpg")
        if os.path.exists(thumb):
            os.remove(thumb)
        logging.info("🗑️ [Delete] %s", filename)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════
# HTML Template
# ═══════════════════════════════════════════════════════════════

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Microscope Control Station</title>
  <style>
    /* ─── FONTS ─── */
    @font-face { font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 400; font-display: swap; src: url(/assets/IBMPlexMono-Regular.woff2) format('woff2'); }
    @font-face { font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 500; font-display: swap; src: url(/assets/IBMPlexMono-Medium.woff2) format('woff2'); }
    @font-face { font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 600; font-display: swap; src: url(/assets/IBMPlexMono-SemiBold.woff2) format('woff2'); }
    @font-face { font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 700; font-display: swap; src: url(/assets/IBMPlexMono-Bold.woff2) format('woff2'); }

    /* ─── RESET ─── */
    *, *::before, *::after { box-sizing: border-box; }
    body { min-width: 320px; margin: 0; }

    /* ─── THEME TOKENS ─── */
    [data-page="microscope"] {
      color-scheme: dark;
      --s-bg: #0a0a0a;
      --s-layer: #141414;
      --s-layer-2: #1e1e1e;
      --s-line: rgba(255,255,255,0.08);
      --s-line-strong: rgba(255,255,255,0.16);
      --s-text: #e5e5e5;
      --s-muted: #999999;
      --s-faint: #777777;
      --s-accent: #6b8aff;
      --s-accent-text: #6b8aff;
      --s-dot: #333333;
      --s-pattern: #1e1e1e;
      --s-green: #22c55e;
      --s-red: #ef4444;

      min-height: 100vh;
      display: flex;
      flex-direction: column;
      font-synthesis: none;
      overflow-x: clip;
      background: var(--s-bg);
      font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: var(--s-text);
      -webkit-font-smoothing: antialiased;
    }

    [data-page="microscope"] h1, [data-page="microscope"] h2, [data-page="microscope"] p { margin: 0; }
    [data-page="microscope"] a { color: var(--s-text); text-decoration: none; }
    [data-page="microscope"] a:hover { text-decoration: none; }

    /* ─── CONTAINER ─── */
    .s-container { width: 100%; max-width: 90rem; margin: 0 auto; }

    /* ─── HEADER ─── */
    .s-header {
      position: sticky; top: 0; z-index: 10;
      width: 100%; background: var(--s-bg);
    }
    .s-header-bar {
      display: flex; align-items: center; gap: 16px;
      min-height: 72px; padding: 20px 20px 20px 24px;
    }
    .s-brand {
      flex: 0 0 auto; display: flex; align-items: center; gap: 10px;
      margin-right: auto; color: var(--s-text); text-decoration: none;
      font-size: 13px; font-weight: 500;
    }
    .s-brand svg { flex-shrink: 0; }
    .s-nav { display: none; align-items: center; justify-content: center; }
    .s-nav ul { display: flex; align-items: center; margin: 0; padding: 0; list-style: none; }
    .s-nav a {
      display: flex; align-items: center; justify-content: center;
      height: 32px; padding: 0 15px; color: var(--s-muted);
      font-size: 13px; line-height: 1; white-space: nowrap; text-decoration: none;
    }
    .s-nav a:hover { color: var(--s-text); }
    .s-nav a.active { color: var(--s-text); }
    .s-header-actions { flex: 0 0 auto; display: flex; align-items: center; gap: 8px; }
    .s-header-btn {
      position: relative; flex-shrink: 0; display: inline-flex;
      align-items: center; justify-content: center; gap: 6px;
      height: 32px; padding: 0 12px; border: 0; border-radius: 0;
      appearance: none; font: inherit; font-size: 13px; line-height: 1.1;
      cursor: pointer; text-decoration: none; white-space: nowrap;
    }
    .s-header-btn strong { font-weight: 500; position: relative; z-index: 1; }
    .s-header-btn span { position: relative; z-index: 1; }
    .s-header-btn.neutral {
      color: var(--s-text); background: var(--s-layer-2);
      box-shadow: 0 0 0 .5px rgba(255,255,255,.08), 0 1px 1.5px rgba(0,0,0,.3);
    }
    .s-header-btn.neutral span { color: var(--s-muted); font-weight: 400; font-variant-numeric: tabular-nums; }
    .s-header-btn.contrast {
      color: #fff; background: var(--s-accent);
      box-shadow: 0 0 0 .5px rgba(255,255,255,.12), 0 1px 1.5px rgba(0,0,0,.3);
    }

    /* ─── HERO ─── */
    .s-hero {
      display: flex; flex-direction: column; align-items: flex-start;
      justify-content: flex-end; gap: 16px; min-height: 0;
      overflow: hidden; padding: 96px 24px 32px;
    }
    .s-hero-canvas {
      position: relative; display: flex; flex-direction: column; gap: 24px;
      width: 100%; min-width: 0;
    }
    .s-hero h1 {
      order: 1; color: var(--s-text);
      font-size: 48px; font-weight: 500; line-height: 1; letter-spacing: 0;
    }
    .s-hero-meta {
      display: flex; align-items: center; gap: 4px;
      width: fit-content; height: 24px; padding: 0 8px 0 4px;
      background: var(--s-layer-2); color: var(--s-muted);
      font-size: 13px; font-weight: 500; line-height: 1.1;
    }
    .s-hero-meta svg { width: 16px; height: 16px; flex: 0 0 auto; }
    .s-hero-copy {
      order: 3; color: var(--s-muted);
      font-size: 16px; font-weight: 400; line-height: 1.5;
    }
    .s-dot-pattern {
      order: 2; flex: 0 0 auto; width: 100%; height: 16px; overflow: hidden;
      background: var(--s-pattern);
      mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 6 6' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0H2V2H0V0Z' fill='black'/%3E%3C/svg%3E");
      mask-repeat: repeat; mask-size: 6px 6px;
      -webkit-mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 6 6' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0H2V2H0V0Z' fill='black'/%3E%3C/svg%3E");
      -webkit-mask-repeat: repeat; -webkit-mask-size: 6px 6px;
    }

    /* ─── LIVE TWO-COLUMN LAYOUT ─── */
    .s-live-grid {
      display: grid; grid-template-columns: 1fr 360px; gap: 0; width: 100%;
    }
    .s-live-main { min-width: 0; }
    .s-live-sidebar {
      position: sticky; top: 72px; align-self: start;
      max-height: calc(100vh - 72px); overflow-y: auto;
      border-left: 1px solid var(--s-line);
      scrollbar-width: thin; scrollbar-color: var(--s-line) transparent;
    }
    .s-live-sidebar::-webkit-scrollbar { width: 4px; }
    .s-live-sidebar::-webkit-scrollbar-track { background: transparent; }
    .s-live-sidebar::-webkit-scrollbar-thumb { background: var(--s-line); }
    .s-live-sidebar .s-panel { border-left: 0; border-right: 0; }
    .s-live-sidebar .s-panel + .s-panel { border-top: 0; }

    /* ─── SECTIONS ─── */
    .s-section {
      position: relative;
      box-shadow: inset 0 -1px var(--s-line), inset 1px 0 var(--s-line), inset -1px 0 var(--s-line);
      padding: 80px 40px; color: var(--s-text);
    }
    .s-section:first-of-type {
      box-shadow: 0 -1px var(--s-line), inset 0 -1px var(--s-line), inset 1px 0 var(--s-line), inset -1px 0 var(--s-line);
    }
    .s-section-title {
      max-width: 1200px; margin-bottom: 40px; color: var(--s-muted);
      font-size: 28px; font-weight: 400; line-height: 1.5; letter-spacing: 0;
    }
    .s-section-title strong { color: var(--s-text); font-weight: 500; }
    .s-section-title span { color: var(--s-muted); font-weight: 400; }

    /* ─── TAB NAV ─── */
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .tab-btn {
      display: flex; align-items: center; justify-content: center;
      height: 32px; padding: 0 15px; color: var(--s-muted);
      font-size: 13px; line-height: 1; white-space: nowrap;
      font-family: inherit; background: transparent; border: 0;
      cursor: pointer; text-decoration: none;
    }
    .tab-btn:hover { color: var(--s-text); }
    .tab-btn.active { color: var(--s-text); }

    /* ─── STREAM VIEWPORT ─── */
    .s-viewport {
      position: relative; width: 100%;
      background: #000; overflow: hidden; aspect-ratio: 16/9;
      cursor: crosshair;
      border: 1px solid var(--s-line);
    }
    canvas#view-canvas { width: 100%; height: 100%; display: block; }

    /* ─── HUD ─── */
    .hud {
      position: absolute; top: 0; left: 0;
      padding: 6px 12px; font-size: 11px; font-weight: 500;
      pointer-events: none; transition: all .14s ease;
      background: rgba(10,10,10,.8); color: var(--s-muted);
      border-right: 1px solid var(--s-line);
      border-bottom: 1px solid var(--s-line);
      font-variant-numeric: tabular-nums;
    }
    .hud.busy { background: rgba(107,138,255,.08); color: var(--s-accent); border-color: rgba(107,138,255,.2); }

    /* ─── REC BADGE ─── */
    .rec-dot {
      position: absolute; top: 0; right: 0;
      display: none; align-items: center; gap: 6px;
      background: rgba(239,68,68,.06); color: var(--s-red);
      padding: 6px 12px; font-size: 10px; font-weight: 600;
      letter-spacing: .06em; text-transform: uppercase;
      border-left: 1px solid rgba(239,68,68,.2);
      border-bottom: 1px solid rgba(239,68,68,.2);
    }
    .rec-dot::before { content: ''; width: 6px; height: 6px; background: var(--s-red); border-radius: 50%; animation: blink 1s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

    /* ─── RETICLE ─── */
    .reticle {
      position: absolute; width: 48px; height: 48px;
      border: 1px solid rgba(107,138,255,.35);
      transform: translate(-50%,-50%); pointer-events: none;
      transition: all .15s ease; opacity: .7;
    }
    .reticle::before, .reticle::after { content: ''; position: absolute; background: rgba(107,138,255,.35); }
    .reticle::before { width: 1px; height: 12px; left: 50%; top: 50%; transform: translate(-50%,-50%); }
    .reticle::after { width: 12px; height: 1px; left: 50%; top: 50%; transform: translate(-50%,-50%); }
    .reticle.focusing { border-color: var(--s-accent); box-shadow: 0 0 20px rgba(107,138,255,.15); animation: pulse .8s infinite alternate; opacity: 1; }
    .reticle.focusing::before, .reticle.focusing::after { background: var(--s-accent); }
    .reticle.focused { border-color: var(--s-green); opacity: 1; }
    .reticle.focused::before, .reticle.focused::after { background: var(--s-green); }
    @keyframes pulse { 0%{transform:translate(-50%,-50%) scale(.96)} 100%{transform:translate(-50%,-50%) scale(1.06)} }

    /* ─── DATA TABLE ROWS (OpenCode pattern) ─── */
    .s-table { display: grid; gap: 4px; width: 100%; margin-top: 36px; }
    .s-table-head {
      display: grid; align-items: center; gap: 16px; min-height: 28px;
      padding: 0 12px; color: var(--s-faint); font-size: 11px; font-weight: 500; line-height: 14px;
    }
    .s-row {
      display: grid; align-items: center; gap: 16px; min-height: 48px;
      padding: 8px 12px; border: 1px solid var(--s-line);
      background: var(--s-layer); font-size: 13px; color: var(--s-text);
    }
    .s-row:hover { border-color: var(--s-line-strong); background: var(--s-layer-2); }
    .s-row .s-label { color: var(--s-faint); font-size: 11px; font-weight: 500; white-space: nowrap; }
    .s-row .s-value { font-weight: 500; font-variant-numeric: tabular-nums; text-align: right; }

    /* Sharpness meter row */
    .s-meter-row { grid-template-columns: 80px 1fr 48px 80px auto; }
    .s-meter-track { height: 2px; background: rgba(255,255,255,.06); overflow: hidden; }
    .s-meter-fill { height: 100%; width: 0; background: var(--s-accent); transition: width .2s ease; }

    /* Focus row */
    .s-focus-row { grid-template-columns: 50px auto 1fr 48px 30px auto; }
    .s-focus-row input[type=range] { min-width: 80px; }

    /* ─── LEADER CARDS (for action buttons) ─── */
    .s-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 140px), 1fr)); gap: 8px; margin-top: 16px; }
    .s-card {
      position: relative; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 6px;
      min-height: 88px; padding: 16px; border: 1px solid var(--s-line);
      background: linear-gradient(180deg, rgba(255,255,255,.04), transparent), var(--s-layer);
      font-size: 11px; font-weight: 500; line-height: 1.1; color: var(--s-muted);
      cursor: pointer; text-decoration: none;
      box-shadow: 0 0 0 .5px rgba(0,0,0,.12), 0 1px 2px -1px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.04);
      transition: background .14s ease, border-color .14s ease, box-shadow .14s ease, transform .14s ease;
    }
    .s-card:hover {
      border-color: var(--s-line-strong);
      box-shadow: 0 0 0 .5px rgba(255,255,255,.08), 0 6px 16px rgba(0,0,0,.15), 0 2px 6px rgba(0,0,0,.1);
      transform: translateY(-1px); color: var(--s-text);
    }
    .s-card:active { transform: translateY(0); }
    .s-card:disabled, .s-card[disabled] { opacity: .3; cursor: not-allowed; transform: none !important; }
    .s-card svg { width: 20px; height: 20px; color: var(--s-accent); }
    .s-card.primary { background: linear-gradient(180deg, rgba(107,138,255,.15), rgba(107,138,255,.05)), var(--s-layer); border-color: rgba(107,138,255,.2); }
    .s-card.primary svg { color: var(--s-accent); }
    .s-card.primary:hover { border-color: rgba(107,138,255,.4); }
    .s-card.danger { border-color: rgba(239,68,68,.2); color: var(--s-red); }
    .s-card.danger svg { color: var(--s-red); }

    /* ─── BUTTON (small/inline) ─── */
    .btn {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 0; height: 24px; padding: 0 8px;
      border: 0; background: var(--s-layer-2); color: var(--s-faint);
      font: inherit; font-size: 11px; font-weight: 500; cursor: pointer;
      box-shadow: 0 0 0 .5px rgba(255,255,255,.06);
      transition: color .14s ease, background .14s ease;
    }
    .btn:hover { color: var(--s-text); background: rgba(255,255,255,.08); }
    .btn:disabled { opacity: .3; cursor: not-allowed; }
    .btn svg { width: 14px; height: 14px; flex-shrink: 0; }
    .btn-danger { background: rgba(239,68,68,.06); color: var(--s-red); box-shadow: 0 0 0 .5px rgba(239,68,68,.15); }
    .btn-danger:hover { background: rgba(239,68,68,.12); }

    /* ─── NUDGE BUTTONS ─── */
    .s-nudge {
      display: inline-flex; align-items: center; justify-content: center;
      height: 32px; min-width: 48px; padding: 0 8px;
      border: 1px solid var(--s-line); background: var(--s-layer);
      color: var(--s-muted); font: inherit; font-size: 11px; font-weight: 500;
      font-variant-numeric: tabular-nums; cursor: pointer;
      transition: all .14s ease;
    }
    .s-nudge:hover { border-color: var(--s-line-strong); color: var(--s-text); background: var(--s-layer-2); transform: translateY(-1px); }

    /* ─── CONTROL PANELS ─── */
    .s-panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; margin-top: 36px; }
    .s-panel {
      border: 1px solid var(--s-line);
      background: linear-gradient(180deg, rgba(255,255,255,.04), transparent), var(--s-layer);
      overflow: hidden;
    }
    .s-panel-head {
      padding: 12px 16px; font-size: 11px; font-weight: 600;
      color: var(--s-text); cursor: pointer; display: flex; align-items: center;
      gap: 8px; user-select: none; list-style: none;
      letter-spacing: .02em; text-transform: uppercase; min-height: 48px;
    }
    .s-panel-head svg { width: 14px; height: 14px; color: var(--s-accent); flex-shrink: 0; }
    .s-panel-head::after {
      content: ''; margin-left: auto; width: 14px; height: 14px;
      background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='none' stroke='%23777' stroke-width='1.5' stroke-linecap='round'%3E%3Cpath d='M3.5 5.25l3.5 3.5 3.5-3.5'/%3E%3C/svg%3E");
      transition: transform .2s ease;
    }
    details[open] .s-panel-head::after { transform: rotate(180deg); }
    details.s-panel:not([open]) .s-panel-head { border-bottom: none; }
    .s-panel-body { padding: 4px 16px 16px; }
    .s-ctrl {
      display: grid; grid-template-columns: 80px 1fr 48px;
      align-items: center; gap: 12px; min-height: 40px; padding: 2px 0;
    }
    .s-ctrl + .s-ctrl { border-top: 1px solid rgba(255,255,255,.04); }
    .s-ctrl label { font-size: 11px; color: var(--s-faint); font-weight: 400; }
    .s-ctrl input[type=range] { width: 100%; accent-color: var(--s-accent); cursor: pointer; }
    .s-ctrl .val { text-align: right; font-size: 11px; color: var(--s-muted); font-variant-numeric: tabular-nums; }
    .s-ctrl select {
      width: 100%; background: var(--s-bg); color: var(--s-text);
      border: 1px solid var(--s-line); padding: 6px 8px;
      font: inherit; font-size: 11px; cursor: pointer; outline: none;
    }
    .s-ctrl select:focus { border-color: var(--s-accent); }
    .s-ctrl-wide { grid-template-columns: 80px 1fr; }
    .s-ctrl-right { grid-template-columns: 1fr auto; }

    /* ─── TOGGLE ─── */
    .toggle { position: relative; width: 32px; height: 18px; flex-shrink: 0; }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .toggle span {
      position: absolute; inset: 0; background: rgba(255,255,255,.08);
      border-radius: 999px; cursor: pointer; transition: .2s ease;
      border: 1px solid var(--s-line);
    }
    .toggle span::before { content: ''; position: absolute; width: 12px; height: 12px; left: 2px; bottom: 2px; background: #fff; border-radius: 50%; transition: .2s ease; }
    .toggle input:checked+span { background: var(--s-accent); border-color: var(--s-accent); }
    .toggle input:checked+span::before { transform: translateX(14px); }

    /* ─── STATUS (like footer status) ─── */
    #status-msg {
      display: flex; align-items: center; gap: 10px;
      width: 100%; font-size: 11px; color: var(--s-muted);
      min-height: 28px; padding: 16px 12px; margin-top: 12px;
      font-variant-numeric: tabular-nums;
    }

    /* ─── GALLERY ─── */
    .s-gallery-bar {
      display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .s-gallery-bar select {
      background: var(--s-layer); color: var(--s-text);
      border: 1px solid var(--s-line); padding: 6px 10px;
      font: inherit; font-size: 11px;
    }
    .s-gallery-bar .stats {
      margin-left: auto; font-size: 11px; color: var(--s-faint);
      font-variant-numeric: tabular-nums;
    }
    .gallery-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 8px; width: 100%;
    }
    .m-card {
      border: 1px solid var(--s-line);
      background: linear-gradient(180deg, rgba(255,255,255,.04), transparent), var(--s-layer);
      overflow: hidden;
      transition: all .14s ease;
    }
    .m-card:hover {
      border-color: var(--s-line-strong);
      box-shadow: 0 0 0 .5px rgba(255,255,255,.08), 0 6px 16px rgba(0,0,0,.15);
      transform: translateY(-1px);
    }
    .m-thumb { position: relative; width: 100%; aspect-ratio: 4/3; background: #000; cursor: pointer; overflow: hidden; }
    .m-thumb img { width: 100%; height: 100%; object-fit: cover; transition: transform .3s ease; }
    .m-card:hover .m-thumb img { transform: scale(1.02); }
    .m-thumb .play { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; background: rgba(0,0,0,.4); }
    .m-thumb .play svg { width: 32px; height: 32px; color: #fff; opacity: .8; }
    .m-info { padding: 10px 12px 4px; }
    .m-name { font-size: 11px; color: var(--s-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 400; }
    .m-meta { font-size: 10px; color: var(--s-faint); margin-top: 3px; font-variant-numeric: tabular-nums; }
    .m-actions { display: flex; gap: 4px; padding: 6px 12px 10px; }
    .empty { grid-column: 1/-1; text-align: center; color: var(--s-faint); padding: 60px 20px; font-size: 13px; }

    /* ─── LIGHTBOX ─── */
    .lb {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,.92);
      z-index: 1000; flex-direction: column; align-items: center; justify-content: center;
      padding: 20px; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    }
    .lb.open { display: flex; }
    .lb-close {
      position: absolute; top: 16px; right: 20px; width: 32px; height: 32px;
      display: flex; align-items: center; justify-content: center;
      background: var(--s-layer-2); border: 0; color: var(--s-faint);
      font-size: 18px; cursor: pointer; font-family: inherit;
      box-shadow: 0 0 0 .5px rgba(255,255,255,.08);
      transition: color .14s ease;
    }
    .lb-close:hover { color: var(--s-text); }
    .lb-body { max-width: 95vw; max-height: 80vh; }
    .lb-body img, .lb-body video { max-width: 100%; max-height: 80vh; border: 1px solid var(--s-line); }
    .lb-bar { display: flex; align-items: center; gap: 16px; margin-top: 16px; font-size: 11px; color: var(--s-muted); }
    .lb-bar a { color: var(--s-accent-text); text-decoration: none; }
    .lb-bar a:hover { color: var(--s-text); }

    /* ─── RANGE INPUT ─── */
    input[type=range] {
      -webkit-appearance: none; appearance: none;
      height: 2px; background: var(--s-line); outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none; appearance: none;
      width: 10px; height: 10px; background: var(--s-accent); border: none; cursor: pointer;
    }
    input[type=range]::-moz-range-thumb {
      width: 10px; height: 10px; background: var(--s-accent); border: none; cursor: pointer;
    }

    /* ─── FOOTER ─── */
    .s-footer {
      position: relative; display: grid; gap: 24px;
      padding: 48px clamp(16px, 4vw, 40px) 24px;
      font-size: 11px;
    }
    .s-footer-pattern {
      height: 16px; overflow: hidden; background: var(--s-pattern);
      mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 6 6' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0H2V2H0V0Z' fill='black'/%3E%3C/svg%3E");
      mask-repeat: repeat; mask-size: 6px 6px;
      -webkit-mask-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 6 6' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0H2V2H0V0Z' fill='black'/%3E%3C/svg%3E");
      -webkit-mask-repeat: repeat; -webkit-mask-size: 6px 6px;
    }
    .s-footer-bottom {
      display: flex; align-items: center; justify-content: space-between;
      gap: 24px; font-size: 11px; line-height: 1;
    }
    .s-status { display: flex; align-items: center; gap: 10px; color: var(--s-muted); }
    .s-status::before { content: ''; width: 6px; height: 6px; background: #198b43; }

    /* ─── DESKTOP ─── */
    @media (min-width: 48rem) {
      .s-hero { padding: 96px 32px 24px; }
      .s-hero-canvas { position: relative; display: block; height: 180px; overflow: hidden; }
      .s-hero h1 { position: absolute; top: 0; left: 0; z-index: 1; padding: 0 12px 12px 0; background: var(--s-bg); }
      .s-hero-copy { position: absolute; right: 0; bottom: 0; z-index: 1; padding: 12px 0 0 16px; background: var(--s-bg); text-align: right; }
      .s-nav { display: flex; }
    }

    /* ─── MOBILE ─── */
    @media (max-width: 79.99rem) {
      .s-live-grid { grid-template-columns: 1fr 320px; }
    }
    @media (max-width: 63.99rem) {
      .s-live-grid { grid-template-columns: 1fr; }
      .s-live-sidebar {
        position: static; max-height: none; overflow-y: visible;
        border-left: 0; border-top: 1px solid var(--s-line);
      }
      .s-live-sidebar .s-panel { border-left: 1px solid var(--s-line); border-right: 1px solid var(--s-line); }
    }
    @media (max-width: 47.99rem) {
      .s-section { padding: 48px 16px; }
      .s-section-title { font-size: 20px; }
      .s-hero h1 { font-size: 32px; }
      .s-cards { grid-template-columns: repeat(auto-fit, minmax(min(100%, 100px), 1fr)); }
      .s-cards .s-card { min-height: 72px; padding: 12px; }
      .s-meter-row { grid-template-columns: 60px 1fr 40px 70px auto; }
      .s-focus-row { grid-template-columns: 40px auto 1fr 40px 25px auto; }
      .s-ctrl { grid-template-columns: 70px 1fr 40px; gap: 8px; }
    }
  </style>
</head>
<body>
<main data-page="microscope">

<!-- ─── HEADER ─── -->
<header class="s-header">
  <div class="s-header-bar">
    <a class="s-brand" href="#">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path opacity="0.2" d="M12 12H4V6H12V12Z" fill="currentColor"/><path d="M12 3H4V12H12V3ZM14 1H2V14H14V1Z" fill="currentColor"/></svg>
      microscope
    </a>
    <nav class="s-nav" aria-label="Sections">
      <ul>
        <li><button class="tab-btn active" onclick="switchTab('live',this)">Live View</button></li>
        <li><button class="tab-btn" onclick="switchTab('gallery',this)">Gallery</button></li>
      </ul>
    </nav>
    <div class="s-header-actions">
      <a class="s-header-btn neutral" href="/api/controls" target="_blank"><strong>API</strong><span>[json]</span></a>
      <a class="s-header-btn contrast" href="/api/camera_info" target="_blank"><strong>Camera Info</strong></a>
    </div>
  </div>
</header>

<div class="s-container">

<!-- ─── HERO ─── -->
<section class="s-hero">
  <p class="s-hero-meta">
    <svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16"><path fill-rule="evenodd" clip-rule="evenodd" d="M13 13H3V3H13V13ZM6.46777 6.81641V7.81641H7.5791V11.3721H8.5791V6.81641H6.46777ZM7.30078 4.62891V5.62891H8.85645V4.62891H7.30078Z" fill="currentColor"/></svg>
    IMX519 16MP &middot; Raspberry Pi 5
  </p>
  <div class="s-hero-canvas">
    <div class="s-dot-pattern" aria-hidden="true"></div>
    <h1>Control Station</h1>
    <p class="s-hero-copy">Live camera feed, precision focus control, 16MP capture,<br>and full exposure management.</p>
  </div>
</section>

<!-- ═══ LIVE TAB ═══ -->
<div id="tab-live" class="tab-content active">

<section class="s-section" style="box-shadow: 0 -1px var(--s-line), inset 0 -1px var(--s-line), inset 1px 0 var(--s-line), inset -1px 0 var(--s-line); padding: 40px 0 0;">
  <div style="display:flex; align-items:center; justify-content:space-between; padding: 0 40px; gap: 16px;">
    <h2 class="s-section-title" style="padding:0; margin:0;"><strong>Live Feed.</strong> <span>MJPEG stream with real-time controls.</span></h2>
    <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
      <span id="mode-label" style="font:500 12px/1 var(--s-mono); color:var(--s-muted); letter-spacing:.03em; text-transform:uppercase;">Quality</span>
      <label class="toggle"><input type="checkbox" id="stream-mode-toggle" onchange="toggleStreamMode(this.checked)"><span></span></label>
      <span style="font:500 12px/1 var(--s-mono); color:var(--s-faint); letter-spacing:.03em; text-transform:uppercase;">Fast</span>
    </div>
  </div>

  <div class="s-live-grid">
    <!-- ── LEFT: Feed ── -->
    <div class="s-live-main" style="padding: 0 0 40px 40px;">
      <div class="s-viewport" id="container">
        <canvas id="view-canvas"></canvas>
        <div class="hud" id="hud">Connecting...</div>
        <div class="rec-dot" id="rec-badge">REC</div>
        <div class="reticle" id="reticle" style="top:50%;left:50%"></div>
      </div>

      <!-- Sharpness meter -->
      <div class="s-table">
        <div class="s-row s-meter-row">
          <span class="s-label">SHARPNESS</span>
          <div class="s-meter-track"><div id="meter-fill" class="s-meter-fill"></div></div>
          <span class="s-value" id="sharp-val" style="color:var(--s-accent)">0.0</span>
          <span style="color:var(--s-faint);font-size:11px">peak <span id="peak-val" style="font-variant-numeric:tabular-nums">0.0</span></span>
          <button class="btn" onclick="fetch('/api/reset_peak')">Reset</button>
        </div>
      </div>

      <!-- Focus -->
      <div class="s-table" style="margin-top:4px">
        <div class="s-row s-focus-row">
          <span class="s-label">FOCUS</span>
          <div style="display:flex;gap:4px">
            <button class="s-nudge" onclick="nudgeFocus(-0.20)">\u2212.20</button>
            <button class="s-nudge" onclick="nudgeFocus(-0.05)">\u2212.05</button>
          </div>
          <input type="range" id="focus-slider" min="0" max="3200" value="160" oninput="onFocusSlider(this.value)">
          <span class="s-value" id="focus-label" style="color:var(--s-text)">1.60</span>
          <span class="s-label">dpt</span>
          <div style="display:flex;gap:4px">
            <button class="s-nudge" onclick="nudgeFocus(0.05)">+.05</button>
            <button class="s-nudge" onclick="nudgeFocus(0.20)">+.20</button>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="s-cards">
        <button class="s-card" id="btn-refocus" onclick="triggerSpotFocus(targetX,targetY)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/></svg>
          Autofocus
        </button>
        <button class="s-card" id="btn-broad" onclick="triggerSpotFocus(targetX,targetY,true)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
          Broad Scan
        </button>
        <button class="s-card primary" id="btn-snap" onclick="capture16MP()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>
          Capture 16MP
        </button>
        <button class="s-card" id="btn-rec" onclick="toggleRecording()">
          <svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="6"/></svg>
          Record
        </button>
        <a class="s-card" id="btn-dl" href="/download_latest" target="_blank" style="display:none">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Download
        </a>
      </div>

      <div id="status-msg"></div>
    </div>

    <!-- ── RIGHT: Controls Sidebar ── -->
    <div class="s-live-sidebar">
      <details class="s-panel" open>
        <summary class="s-panel-head">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="1"/><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg>
          Exposure
        </summary>
        <div class="s-panel-body">
          <div class="s-ctrl s-ctrl-wide"><label>Auto Exposure</label><label class="toggle"><input type="checkbox" id="ae-enable" checked onchange="setCtrl('AeEnable',this.checked);updateAeUI()"><span></span></label></div>
          <div class="s-ctrl ae-manual"><label>Exposure \u00b5s</label><input type="range" id="sl-exposure" min="0" max="1000" value="500" oninput="onExposureSlider(this.value)"><span class="val" id="v-exposure">20000</span></div>
          <div class="s-ctrl ae-manual"><label>Gain</label><input type="range" id="sl-gain" min="10" max="160" value="10" oninput="onSlider('AnalogueGain',this.value/10,'v-gain',1)"><span class="val" id="v-gain">1.0</span></div>
          <div class="s-ctrl ae-auto"><label>EV Comp</label><input type="range" id="sl-ev" min="-80" max="80" value="0" oninput="onSlider('ExposureValue',this.value/10,'v-ev',1)"><span class="val" id="v-ev">0.0</span></div>
          <div class="s-ctrl s-ctrl-wide ae-auto"><label>Metering</label><select id="sel-metering" onchange="setCtrl('AeMeteringMode',this.value)"><option value="CentreWeighted" selected>Centre-Weighted</option><option value="Spot">Spot</option><option value="Matrix">Matrix</option></select></div>
          <div class="s-ctrl s-ctrl-wide ae-auto"><label>Mode</label><select id="sel-aemode" onchange="setCtrl('AeExposureMode',this.value)"><option value="Normal" selected>Normal</option><option value="Short">Short</option><option value="Long">Long</option></select></div>
        </div>
      </details>

      <details class="s-panel" open>
        <summary class="s-panel-head">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/></svg>
          White Balance
        </summary>
        <div class="s-panel-body">
          <div class="s-ctrl s-ctrl-wide"><label>Auto WB</label><label class="toggle"><input type="checkbox" id="awb-enable" checked onchange="setCtrl('AwbEnable',this.checked);updateAwbUI()"><span></span></label></div>
          <div class="s-ctrl s-ctrl-wide awb-auto"><label>Preset</label><select id="sel-awb" onchange="setCtrl('AwbMode',this.value)"><option value="Auto" selected>Auto</option><option value="Daylight">Daylight</option><option value="Cloudy">Cloudy</option><option value="Tungsten">Tungsten</option><option value="Fluorescent">Fluorescent</option><option value="Indoor">Indoor</option><option value="Incandescent">Incandescent</option></select></div>
          <div class="s-ctrl awb-manual"><label>Red Gain</label><input type="range" id="sl-rgain" min="0" max="800" value="200" oninput="onSlider('ColourGainRed',this.value/100,'v-rgain',2)"><span class="val" id="v-rgain">2.00</span></div>
          <div class="s-ctrl awb-manual"><label>Blue Gain</label><input type="range" id="sl-bgain" min="0" max="800" value="200" oninput="onSlider('ColourGainBlue',this.value/100,'v-bgain',2)"><span class="val" id="v-bgain">2.00</span></div>
        </div>
      </details>

      <details class="s-panel" open>
        <summary class="s-panel-head">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r="2.5"/><path d="M17 2H7a5 5 0 00-5 5v10a5 5 0 005 5h10a5 5 0 005-5V7a5 5 0 00-5-5z"/></svg>
          Image
        </summary>
        <div class="s-panel-body">
          <div class="s-ctrl"><label>Brightness</label><input type="range" id="sl-brightness" min="-100" max="100" value="0" oninput="onSlider('Brightness',this.value/100,'v-brightness',2)"><span class="val" id="v-brightness">0.00</span></div>
          <div class="s-ctrl"><label>Contrast</label><input type="range" id="sl-contrast" min="0" max="400" value="100" oninput="onSlider('Contrast',this.value/100,'v-contrast',2)"><span class="val" id="v-contrast">1.00</span></div>
          <div class="s-ctrl"><label>Saturation</label><input type="range" id="sl-saturation" min="0" max="400" value="100" oninput="onSlider('Saturation',this.value/100,'v-saturation',2)"><span class="val" id="v-saturation">1.00</span></div>
          <div class="s-ctrl"><label>Sharpness</label><input type="range" id="sl-sharpness" min="0" max="800" value="100" oninput="onSlider('Sharpness',this.value/100,'v-sharpness',2)"><span class="val" id="v-sharpness">1.00</span></div>
        </div>
      </details>

      <details class="s-panel">
        <summary class="s-panel-head">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
          Advanced
        </summary>
        <div class="s-panel-body">
          <div class="s-ctrl s-ctrl-wide"><label>Denoise</label><select id="sel-nr" onchange="setCtrl('NoiseReductionMode',this.value)"><option value="0" selected>Off</option><option value="1">Fast</option><option value="2">High Quality</option><option value="3">Minimal</option><option value="4">ZSL</option></select></div>
          <div class="s-ctrl s-ctrl-wide"><label>HDR</label><select id="sel-hdr" onchange="setCtrl('HdrMode',this.value)"><option value="0" selected>Off</option><option value="1">Single Exposure</option><option value="2">Multi Exposure</option><option value="4">Night</option></select></div>
          <div class="s-ctrl s-ctrl-wide"><label>Resolution</label><select id="sel-res" onchange="changeResolution(this.value)"><option value="640x480">640 \u00d7 480</option><option value="1280x720">1280 \u00d7 720</option><option value="1920x1080">1920 \u00d7 1080</option><option value="2328x1748" selected>2328 \u00d7 1748</option></select></div>
          <div class="s-ctrl"><label>Stream QP</label><input type="range" id="sl-qp" min="2" max="31" value="10" oninput="document.getElementById('v-qp').innerText=this.value" onchange="changeQP(this.value)"><span class="val" id="v-qp">10</span></div>
          <div class="s-ctrl s-ctrl-wide"><label>Fast Stream</label><label class="toggle"><input type="checkbox" id="fast-stream" onchange="toggleFastStream(this.checked)"><span></span></label></div>
          <div class="s-ctrl"><label>Zoom</label><input type="range" id="sl-zoom" min="100" max="400" value="100" oninput="onSlider('DigitalZoom',this.value/100,'v-zoom',1)"><span class="val" id="v-zoom">1.0x</span></div>
          <div class="s-ctrl s-ctrl-right" style="margin-top:4px"><span></span><button class="btn" onclick="resetDefaults()">Reset All Defaults</button></div>
        </div>
      </details>
    </div>
  </div>
</section>
</div><!-- /tab-live -->

<!-- ═══ GALLERY TAB ═══ -->
<div id="tab-gallery" class="tab-content">
<section class="s-section" style="box-shadow: 0 -1px var(--s-line), inset 0 -1px var(--s-line), inset 1px 0 var(--s-line), inset -1px 0 var(--s-line);">
  <h2 class="s-section-title"><strong>Gallery.</strong> <span>Captured images and recorded video.</span></h2>

  <div class="s-gallery-bar">
    <select id="gallery-sort" onchange="loadGallery()"><option value="newest">Newest First</option><option value="oldest">Oldest First</option><option value="largest">Largest First</option><option value="name">By Name</option></select>
    <select id="gallery-type" onchange="loadGallery()"><option value="all">All Files</option><option value="image">Images Only</option><option value="video">Videos Only</option></select>
    <button class="btn" onclick="loadGallery()"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>Refresh</button>
    <span class="stats" id="gallery-stats"></span>
  </div>
  <div class="gallery-grid" id="gallery-grid"></div>
</section>
</div>

<!-- ═══ Lightbox ═══ -->
<div class="lb" id="lightbox">
  <button class="lb-close" onclick="closeLightbox()">&times;</button>
  <div class="lb-body" id="lightbox-content"></div>
  <div class="lb-bar">
    <span id="lb-name"></span>
    <span id="lb-size" style="color:var(--s-faint)"></span>
    <a id="lb-download" href="#" download>Download</a>
    <button class="btn btn-danger" id="lb-delete" onclick="deleteLightboxFile()">Delete</button>
  </div>
</div>

</div><!-- /s-container -->

<!-- ─── FOOTER ─── -->
<footer class="s-footer">
  <div class="s-footer-pattern"></div>
  <div class="s-footer-bottom">
    <div>
      <span class="s-status">Camera streaming</span>
    </div>
    <div style="color:var(--s-faint)">Microscope Control Station &middot; Port 1306</div>
  </div>
</footer>

</main>

<script>
const canvas=document.getElementById('view-canvas'),ctx=canvas.getContext('2d'),hud=document.getElementById('hud'),container=document.getElementById('container'),reticle=document.getElementById('reticle'),statusMsg=document.getElementById('status-msg'),recBadge=document.getElementById('rec-badge');
let frameCount=0,lastFpsTime=performance.now(),fps=0,targetX=.5,targetY=.5,isBusy=false,isRecording=false,currentLightboxFile=null,debounceTimers={},pollingPaused=false,streamController=null;

function switchTab(n,b){document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.tab-btn').forEach(t=>t.classList.remove('active'));document.getElementById('tab-'+n).classList.add('active');b.classList.add('active');if(n==='gallery')loadGallery()}

function setCtrl(n,v){clearTimeout(debounceTimers[n]);debounceTimers[n]=setTimeout(()=>{fetch('/api/set_control?name='+encodeURIComponent(n)+'&value='+encodeURIComponent(v))},40)}
function onSlider(n,v,id,d){const el=document.getElementById(id);let t=parseFloat(v).toFixed(d);if(n==='DigitalZoom')t+='x';el.innerText=t;setCtrl(n,v)}
function sliderToExposure(v){return Math.round(Math.exp(v/1000*Math.log(66666)))}
function exposureToSlider(e){return e<=1?0:Math.round(Math.log(e)/Math.log(66666)*1000)}
function onExposureSlider(v){const us=sliderToExposure(parseInt(v));document.getElementById('v-exposure').innerText=us;setCtrl('ExposureTime',us)}
function onFocusSlider(v){const d=(v/100).toFixed(2);document.getElementById('focus-label').innerText=d;setCtrl('LensPosition',d)}
function nudgeFocus(d){let c=parseFloat(document.getElementById('focus-label').innerText)||0;let v=Math.max(0,Math.min(32,c+d));document.getElementById('focus-label').innerText=v.toFixed(2);document.getElementById('focus-slider').value=Math.round(v*100);setCtrl('LensPosition',v.toFixed(2))}

function updateAeUI(){const ae=document.getElementById('ae-enable').checked;document.querySelectorAll('.ae-manual').forEach(r=>{r.querySelectorAll('input,select').forEach(el=>el.disabled=ae);r.style.opacity=ae?'.35':'1'});document.querySelectorAll('.ae-auto').forEach(r=>{r.querySelectorAll('input,select').forEach(el=>el.disabled=!ae);r.style.opacity=ae?'1':'.35'})}
function updateAwbUI(){const a=document.getElementById('awb-enable').checked;document.querySelectorAll('.awb-auto').forEach(r=>{r.querySelectorAll('input,select').forEach(el=>el.disabled=!a);r.style.opacity=a?'1':'.35'});document.querySelectorAll('.awb-manual').forEach(r=>{r.querySelectorAll('input,select').forEach(el=>el.disabled=a);r.style.opacity=a?'.35':'1'})}
updateAeUI();updateAwbUI();

container.addEventListener('click',e=>{if(isBusy)return;const r=container.getBoundingClientRect();targetX=Math.max(.05,Math.min(.95,(e.clientX-r.left)/r.width));targetY=Math.max(.05,Math.min(.95,(e.clientY-r.top)/r.height));reticle.style.left=(targetX*100)+'%';reticle.style.top=(targetY*100)+'%';triggerSpotFocus(targetX,targetY)});

function triggerSpotFocus(x,y,broad=false){if(isBusy)return;isBusy=true;reticle.className='reticle focusing';hud.className='hud busy';hud.innerText='Focusing ['+Math.round(x*100)+'%, '+Math.round(y*100)+'%]...';statusMsg.innerText='Running '+(broad?'broad':'precision')+' focus sweep...';setControlsDisabled(true);fetch('/api/focus_spot?x='+x.toFixed(3)+'&y='+y.toFixed(3)+(broad?'&broad=1':'')).then(r=>r.json()).then(d=>{isBusy=false;setControlsDisabled(false);reticle.className='reticle focused';hud.className='hud';hud.innerText='Focused: '+d.best_focus.toFixed(2)+' dpt | Score: '+d.score;document.getElementById('focus-label').innerText=d.best_focus.toFixed(2);document.getElementById('focus-slider').value=Math.round(d.best_focus*100);statusMsg.innerText='Focus locked at '+d.best_focus.toFixed(2)+' dpt (Score: '+d.score+') in '+d.duration+'s';setTimeout(()=>{reticle.className='reticle'},1500)}).catch(err=>{isBusy=false;setControlsDisabled(false);reticle.className='reticle';hud.className='hud';statusMsg.innerText='Focus failed: '+err})}
function setControlsDisabled(d){document.querySelectorAll('.s-card').forEach(b=>b.disabled=d)}

function capture16MP(){if(isBusy)return;isBusy=true;statusMsg.innerText='Capturing 16MP still...';hud.innerText='Capturing...';setControlsDisabled(true);fetch('/api/capture_16mp').then(r=>r.json()).then(d=>{isBusy=false;setControlsDisabled(false);hud.innerText=d.size_mb+' MB captured';statusMsg.innerText='Saved: '+d.name+' ('+d.size_mb+' MB) in '+d.duration+'s';document.getElementById('btn-dl').style.display='flex';document.getElementById('btn-dl').href='/download_latest?t='+Date.now()}).catch(err=>{isBusy=false;setControlsDisabled(false);statusMsg.innerText='Capture error: '+err})}

function toggleRecording(){if(isRecording){fetch('/api/record_stop').then(r=>r.json()).then(d=>{isRecording=false;recBadge.style.display='none';const b=document.getElementById('btn-rec');b.innerHTML='<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="6"/></svg>Record';b.classList.remove('danger');statusMsg.innerText='Video saved: '+(d.name||'')+' ('+d.size_mb+' MB)'})}else{fetch('/api/record_start').then(r=>r.json()).then(d=>{if(d.status==='ok'){isRecording=true;recBadge.style.display='flex';const b=document.getElementById('btn-rec');b.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>Stop';b.classList.add('danger');statusMsg.innerText='Recording...'}else{statusMsg.innerText='Record error: '+d.message}})}}

function changeResolution(v){const[w,h]=v.split('x').map(Number);statusMsg.innerText='Switching to '+w+'\u00d7'+h+'...';if(streamController)streamController.abort();fetch('/api/resolution?width='+w+'&height='+h).then(r=>r.json()).then(d=>{statusMsg.innerText='Resolution: '+d.resolution[0]+'\u00d7'+d.resolution[1];startStream()}).catch(err=>{statusMsg.innerText='Error: '+err;startStream()})}
function changeQP(v){statusMsg.innerText='Setting stream quality...';if(streamController)streamController.abort();fetch('/api/stream_quality?qp='+v).then(r=>r.json()).then(d=>{statusMsg.innerText='Stream QP: '+d.qp;startStream()}).catch(err=>{statusMsg.innerText='Error: '+err;startStream()})}
function toggleFastStream(on){statusMsg.innerText=on?'Switching to fast stream...':'Switching to quality stream...';hud.innerText='switching...';if(streamController)streamController.abort();fetch('/api/perf_mode?mode='+(on?'fast':'quality')).then(r=>r.json()).then(d=>{statusMsg.innerText=d.mode==='fast'?'Fast Stream: 1280\u00d7720 qp=20':'Quality: 2328\u00d71748 qp=10';startStream();syncControlsFromServer()}).catch(err=>{statusMsg.innerText='Error: '+err;startStream()})}
function toggleStreamMode(fast){document.getElementById('fast-stream').checked=fast;toggleFastStream(fast);updateModeLabel(fast)}
function updateModeLabel(fast){const ml=document.getElementById('mode-label');if(ml){ml.style.color=fast?'var(--s-faint)':'var(--s-accent)';ml.nextElementSibling.nextElementSibling.style.color=fast?'var(--s-accent)':'var(--s-faint)'}}
function resetDefaults(){fetch('/api/reset_defaults').then(r=>r.json()).then(()=>{statusMsg.innerText='All controls reset';setTimeout(syncControlsFromServer,200)})}

function syncControlsFromServer(){fetch('/api/controls').then(r=>r.json()).then(d=>{document.getElementById('ae-enable').checked=d.AeEnable;document.getElementById('awb-enable').checked=d.AwbEnable;document.getElementById('sel-awb').value=d.AwbMode;document.getElementById('sel-metering').value=d.AeMeteringMode;document.getElementById('sel-aemode').value=d.AeExposureMode;document.getElementById('sel-nr').value=d.NoiseReductionMode;document.getElementById('sel-hdr').value=d.HdrMode;document.getElementById('sl-brightness').value=Math.round(d.Brightness*100);document.getElementById('v-brightness').innerText=d.Brightness.toFixed(2);document.getElementById('sl-contrast').value=Math.round(d.Contrast*100);document.getElementById('v-contrast').innerText=d.Contrast.toFixed(2);document.getElementById('sl-saturation').value=Math.round(d.Saturation*100);document.getElementById('v-saturation').innerText=d.Saturation.toFixed(2);document.getElementById('sl-sharpness').value=Math.round(d.Sharpness*100);document.getElementById('v-sharpness').innerText=d.Sharpness.toFixed(2);document.getElementById('sl-gain').value=Math.round(d.AnalogueGain*10);document.getElementById('v-gain').innerText=d.AnalogueGain.toFixed(1);document.getElementById('sl-ev').value=Math.round(d.ExposureValue*10);document.getElementById('v-ev').innerText=d.ExposureValue.toFixed(1);document.getElementById('sl-exposure').value=exposureToSlider(d.ExposureTime);document.getElementById('v-exposure').innerText=d.ExposureTime;document.getElementById('sl-rgain').value=Math.round(d.ColourGainRed*100);document.getElementById('v-rgain').innerText=d.ColourGainRed.toFixed(2);document.getElementById('sl-bgain').value=Math.round(d.ColourGainBlue*100);document.getElementById('v-bgain').innerText=d.ColourGainBlue.toFixed(2);document.getElementById('sl-zoom').value=Math.round(d.DigitalZoom*100);document.getElementById('v-zoom').innerText=d.DigitalZoom.toFixed(1)+'x';document.getElementById('focus-slider').value=Math.round(d.focus*100);document.getElementById('focus-label').innerText=d.focus.toFixed(2);const rv=d.resolution[0]+'x'+d.resolution[1];const rs=document.getElementById('sel-res');if([...rs.options].some(o=>o.value===rv))rs.value=rv;if(d.qp!==undefined){document.getElementById('sl-qp').value=d.qp;document.getElementById('v-qp').innerText=d.qp;const isFast=(d.qp>=20&&d.resolution[0]<=1280);document.getElementById('fast-stream').checked=isFast;document.getElementById('stream-mode-toggle').checked=isFast;updateModeLabel(isFast)}updateAeUI();updateAwbUI()}).catch(()=>{})}

async function startStream(){if(streamController)streamController.abort();streamController=new AbortController();try{const response=await fetch('/stream.mjpg',{signal:streamController.signal});const reader=response.body.getReader();let buffer=new Uint8Array(0);while(true){const{done,value}=await reader.read();if(done)break;const nb=new Uint8Array(buffer.length+value.length);nb.set(buffer);nb.set(value,buffer.length);buffer=nb;let lastSoi=-1,lastEoi=-1;for(let i=0;i<buffer.length-1;i++){if(buffer[i]===0xFF&&buffer[i+1]===0xD8)lastSoi=i;if(lastSoi!==-1&&buffer[i]===0xFF&&buffer[i+1]===0xD9){lastEoi=i+2}}if(lastSoi!==-1&&lastEoi!==-1&&lastEoi>lastSoi){const fb=buffer.slice(lastSoi,lastEoi);buffer=buffer.slice(lastEoi);const blob=new Blob([fb],{type:'image/jpeg'});createImageBitmap(blob).then(bmp=>{if(canvas.width!==bmp.width){canvas.width=bmp.width;canvas.height=bmp.height}ctx.drawImage(bmp,0,0);bmp.close();frameCount++;const now=performance.now();if(now-lastFpsTime>=1000){fps=Math.round((frameCount*1000)/(now-lastFpsTime));frameCount=0;lastFpsTime=now;if(!isBusy)hud.innerText=fps+' fps'}}).catch(()=>{})}if(buffer.length>500000)buffer=new Uint8Array(0)}}catch(err){if(err.name!=='AbortError')setTimeout(startStream,800)}}
function restartStream(){if(streamController)streamController.abort();hud.innerText='reconnecting...';setTimeout(startStream,600)}
startStream();

function updateMetrics(){if(isBusy||pollingPaused)return;fetch('/api/metrics').then(r=>r.json()).then(d=>{document.getElementById('sharp-val').innerText=d.sharpness;document.getElementById('peak-val').innerText=d.peak;let pct=Math.min(100,(d.sharpness/Math.max(d.peak,1))*100);document.getElementById('meter-fill').style.width=pct+'%';if(document.getElementById('ae-enable').checked&&d.actual_exposure){document.getElementById('v-exposure').innerText=d.actual_exposure;document.getElementById('v-gain').innerText=d.actual_gain.toFixed(1)}if(document.getElementById('awb-enable').checked&&d.actual_colour_gains){document.getElementById('v-rgain').innerText=d.actual_colour_gains[0].toFixed(2);document.getElementById('v-bgain').innerText=d.actual_colour_gains[1].toFixed(2)}}).catch(()=>{})}
setInterval(updateMetrics,500);
document.querySelectorAll('input[type=range]').forEach(sl=>{sl.addEventListener('mousedown',()=>{pollingPaused=true});sl.addEventListener('touchstart',()=>{pollingPaused=true})});
document.addEventListener('mouseup',()=>{setTimeout(()=>{pollingPaused=false},300)});document.addEventListener('touchend',()=>{setTimeout(()=>{pollingPaused=false},300)});
syncControlsFromServer();

function loadGallery(){fetch('/api/media').then(r=>r.json()).then(files=>{const sort=document.getElementById('gallery-sort').value;const tf=document.getElementById('gallery-type').value;if(tf!=='all')files=files.filter(f=>f.type===tf);if(sort==='oldest')files.sort((a,b)=>a.mtime-b.mtime);else if(sort==='largest')files.sort((a,b)=>b.size-a.size);else if(sort==='name')files.sort((a,b)=>a.name.localeCompare(b.name));const totalMB=files.reduce((s,f)=>s+f.size_mb,0).toFixed(1);document.getElementById('gallery-stats').innerText=files.length+' files \u00b7 '+totalMB+' MB';const grid=document.getElementById('gallery-grid');grid.innerHTML='';if(!files.length){grid.innerHTML='<div class="empty">No media files found</div>';return}files.forEach(f=>{const c=document.createElement('div');c.className='m-card';const iv=f.type==='video';c.innerHTML='<div class="m-thumb" onclick="openLightbox(\''+f.name.replace(/'/g,"\\'")+"',"+iv+')"><img src="/api/thumb/'+encodeURIComponent(f.name)+'" loading="lazy" onerror="this.style.display=\'none\'">'+(iv?'<div class="play"><svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg></div>':'')+'</div><div class="m-info"><div class="m-name" title="'+f.name+'">'+f.name+'</div><div class="m-meta">'+f.size_mb+' MB \u00b7 '+f.date+'</div></div><div class="m-actions"><a class="btn" href="/media/'+encodeURIComponent(f.name)+'" download>Download</a><button class="btn btn-danger" onclick="event.stopPropagation();deleteFile(\''+f.name.replace(/'/g,"\\'")+'\')">Delete</button></div>';grid.appendChild(c)})}).catch(err=>{document.getElementById('gallery-grid').innerHTML='<div class="empty">Error loading gallery</div>'})}

function openLightbox(n,iv){currentLightboxFile=n;const c=document.getElementById('lightbox-content');const u='/media/'+encodeURIComponent(n);c.innerHTML=iv?'<video src="'+u+'" controls autoplay style="max-width:100%;max-height:80vh"></video>':'<img src="'+u+'">';document.getElementById('lb-name').innerText=n;document.getElementById('lb-download').href=u;document.getElementById('lb-download').download=n;document.getElementById('lightbox').classList.add('open');fetch('/api/media').then(r=>r.json()).then(files=>{const f=files.find(x=>x.name===n);if(f)document.getElementById('lb-size').innerText=f.size_mb+' MB \u00b7 '+f.date})}
function closeLightbox(){document.getElementById('lightbox').classList.remove('open');document.getElementById('lightbox-content').innerHTML='';currentLightboxFile=null}
function deleteLightboxFile(){if(!currentLightboxFile||!confirm('Delete '+currentLightboxFile+'?'))return;fetch('/api/delete_media?file='+encodeURIComponent(currentLightboxFile)).then(r=>r.json()).then(d=>{if(d.status==='ok'){closeLightbox();loadGallery();statusMsg.innerText='Deleted'}else statusMsg.innerText='Error: '+d.message})}
function deleteFile(n){if(!confirm('Delete '+n+'?'))return;fetch('/api/delete_media?file='+encodeURIComponent(n)).then(r=>r.json()).then(d=>{if(d.status==='ok'){loadGallery();statusMsg.innerText='Deleted '+n}else statusMsg.innerText='Error: '+d.message})}
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLightbox()});
</script>
</body>
</html>
"""
# HTTP Handler
# ═══════════════════════════════════════════════════════════════

class StreamingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        if len(args) > 0 and any(skip in str(args[0]) for skip in ["/stream.mjpg", "/api/metrics", "/api/thumb", "/assets/"]):
            return
        logging.info("%s - %s", self.address_string(), format % args)

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32768)
        except Exception:
            pass

    def _json_response(self, data, status=200):
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _serve_file(self, filepath, content_type=None, inline=False):
        if not os.path.exists(filepath):
            self.send_error(404, "Not found")
            return
        if content_type is None:
            content_type = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
        try:
            size = os.path.getsize(filepath)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", size)
            self.send_header("Access-Control-Allow-Origin", "*")
            if not inline:
                self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(filepath)}"')
            self.end_headers()
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        global current_focus_pos, latest_capture_path
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # ─── Pages ───
        if path == "/":
            content = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)

        # ─── MJPEG Stream ───
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private, no-transform")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    if frame:
                        self.wfile.write(b"--FRAME\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ─── Snapshot ───
        elif path == "/snapshot.jpg":
            with output.condition:
                frame = output.frame
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_error(503, "Frame not ready")

        # ─── Download Latest Capture ───
        elif path == "/download_latest":
            if latest_capture_path and os.path.exists(latest_capture_path):
                self._serve_file(latest_capture_path, "image/jpeg", inline=True)
            else:
                self.send_error(404, "No capture available")

        # ─── API: Metrics (lightweight, polled frequently) ───
        elif path == "/api/metrics":
            try:
                meta = picam2.capture_metadata()
            except Exception:
                meta = {}
            self._json_response({
                "sharpness": output.sharpness,
                "peak": output.peak_sharpness,
                "fps": round(output.fps, 1),
                "focus": round(current_focus_pos, 2),
                "actual_exposure": meta.get("ExposureTime", 0),
                "actual_gain": round(meta.get("AnalogueGain", 1.0), 2),
                "actual_colour_gains": [round(g, 2) for g in meta.get("ColourGains", (0, 0))],
                "recording": recording_active,
            })

        # ─── API: Full Control State ───
        elif path == "/api/controls":
            try:
                meta = picam2.capture_metadata()
            except Exception:
                meta = {}
            data = dict(control_state)
            data["focus"] = round(current_focus_pos, 2)
            data["actual_exposure"] = meta.get("ExposureTime", 0)
            data["actual_gain"] = round(meta.get("AnalogueGain", 1.0), 2)
            data["actual_colour_gains"] = [round(g, 2) for g in meta.get("ColourGains", (0, 0))]
            data["actual_colour_temp"] = meta.get("ColourTemperature", 0)
            data["sharpness"] = output.sharpness
            data["peak_sharpness"] = output.peak_sharpness
            data["fps"] = round(output.fps, 1)
            data["recording"] = recording_active
            data["resolution"] = list(current_resolution)
            data["qp"] = current_qp
            self._json_response(data)

        # ─── API: Camera Info (for AI agents) ───
        elif path == "/api/camera_info":
            props = picam2.camera_properties
            ctrls = {}
            for name, (lo, hi, default) in picam2.camera_controls.items():
                try:
                    ctrls[name] = {"min": lo, "max": hi, "default": default}
                except Exception:
                    ctrls[name] = {"min": str(lo), "max": str(hi), "default": str(default)}
            self._json_response({
                "model": props.get("Model", "unknown"),
                "pixel_array_size": list(props.get("PixelArraySize", (0, 0))),
                "current_resolution": list(current_resolution),
                "controls": ctrls,
            })

        # ─── API: Set Control ───
        elif path == "/api/set_control":
            name = qs.get("name", [None])[0]
            value = qs.get("value", [None])[0]
            if name and value is not None:
                result = set_camera_control(name, value)
                self._json_response(result)
            else:
                self._json_response({"status": "error", "message": "Missing name or value"}, 400)

        # ─── API: Focus (legacy compat) ───
        elif path == "/api/focus":
            dpt_val = qs.get("dpt", qs.get("pos", [None]))[0]
            if dpt_val is not None:
                set_camera_control("LensPosition", dpt_val)
            self._json_response({"status": "ok", "focus": round(current_focus_pos, 2)})

        # ─── API: Spot Autofocus ───
        elif path in ("/api/focus_spot", "/api/sweep_spot"):
            sx = float(qs["x"][0]) if "x" in qs else output.spot_x
            sy = float(qs["y"][0]) if "y" in qs else output.spot_y
            broad = "broad" in qs and qs["broad"][0] in ("1", "true")
            res = run_spot_autofocus(sx, sy, broad=broad)
            self._json_response(res)

        # ─── API: Reset Peak ───
        elif path == "/api/reset_peak":
            output.peak_sharpness = output.sharpness
            self._json_response({"status": "ok"})

        # ─── API: Reset Defaults ───
        elif path == "/api/reset_defaults":
            reset_controls_to_defaults()
            self._json_response({"status": "ok"})

        # ─── API: Capture 16MP ───
        elif path == "/api/capture_16mp":
            res = capture_highres_still()
            self._json_response(res)

        # ─── API: Recording ───
        elif path == "/api/record_start":
            self._json_response(start_recording())

        elif path == "/api/record_stop":
            self._json_response(stop_recording())

        # ─── API: Resolution ───
        elif path == "/api/resolution":
            w = int(qs.get("width", [1280])[0])
            h = int(qs.get("height", [720])[0])
            try:
                res = change_resolution(w, h)
                self._json_response(res)
            except Exception as e:
                self._json_response({"status": "error", "message": str(e)}, 500)

        # ─── API: Stream Quality ───
        elif path == "/api/stream_quality":
            qp_val = int(qs.get("qp", [10])[0])
            try:
                res = change_stream_quality(qp_val)
                self._json_response(res)
            except Exception as e:
                self._json_response({"status": "error", "message": str(e)}, 500)

        # ─── API: Performance Mode (low-res + high compression) ───
        elif path == "/api/perf_mode":
            mode = qs.get("mode", ["fast"])[0]
            try:
                if mode == "fast":
                    res = switch_stream_mode(1280, 720, 20)
                    res["mode"] = "fast"
                else:
                    res = switch_stream_mode(2328, 1748, 10)
                    res["mode"] = "quality"
                self._json_response(res)
            except Exception as e:
                logging.exception("perf_mode error")
                self._json_response({"status": "error", "message": str(e)}, 500)

        # ─── API: Media List ───
        elif path == "/api/media":
            self._json_response(list_media_files())

        # ─── API: Thumbnail ───
        elif path.startswith("/api/thumb/"):
            filename = unquote(path[11:])
            if ".." in filename or "/" in filename:
                self.send_error(400, "Bad filename")
                return
            thumb = get_or_create_thumbnail(filename)
            if thumb:
                self._serve_file(thumb, "image/jpeg", inline=True)
            else:
                # Return a 1x1 transparent pixel as fallback
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", 0)
                self.end_headers()

        # ─── API: Delete Media ───
        elif path == "/api/delete_media":
            filename = qs.get("file", [None])[0]
            if filename:
                if ".." in filename or "/" in filename:
                    self._json_response({"status": "error", "message": "Bad filename"}, 400)
                else:
                    self._json_response(delete_media_file(filename))
            else:
                self._json_response({"status": "error", "message": "No file specified"}, 400)

        # ─── Static Assets (fonts etc.) ───
        elif path.startswith("/assets/"):
            filename = unquote(path[8:])
            if ".." in filename or "/" in filename:
                self.send_error(400, "Bad filename")
                return
            assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
            filepath = os.path.join(assets_dir, filename)
            realpath = os.path.realpath(filepath)
            if realpath.startswith(os.path.realpath(assets_dir)) and os.path.exists(filepath):
                self._serve_file(filepath, inline=True)
            else:
                self.send_error(404, "Not found")

        # ─── Media File Download ───
        elif path.startswith("/media/"):
            filename = unquote(path[7:])
            if ".." in filename or "/" in filename:
                self.send_error(400, "Bad filename")
                return
            filepath = os.path.join(MEDIA_DIR, filename)
            realpath = os.path.realpath(filepath)
            if realpath.startswith(os.path.realpath(MEDIA_DIR)) and os.path.exists(filepath):
                self._serve_file(filepath, inline=True)
            else:
                self.send_error(404, "Not found")

        else:
            self.send_error(404, "Not Found")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    global picam2, current_focus_pos
    logging.info("Starting Microscope Control Station...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": current_resolution},
        raw={"size": (2328, 1748)},
        transform=Transform(hflip=True, vflip=True)
    )
    picam2.configure(config)
    picam2.start()

    current_focus_pos = 1.60
    try:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": current_focus_pos})
        logging.info("Initial lens position: %.2f dpt", current_focus_pos)
    except Exception as e:
        logging.warning("Could not set initial focus: %s", e)

    encoder = LibavMjpegEncoder(framerate=30, qp=current_qp)
    picam2.start_recording(encoder, FileOutput(output))

    server_address = ("0.0.0.0", 1306)
    server = ThreadingHTTPServer(server_address, StreamingHandler)
    logging.info("═══════════════════════════════════════")
    logging.info("  Microscope Control Station READY")
    logging.info("  http://0.0.0.0:1306")
    logging.info("═══════════════════════════════════════")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if recording_active:
            stop_recording()
        server.server_close()
        picam2.stop_recording()
        picam2.stop()
        picam2.close()


if __name__ == "__main__":
    main()

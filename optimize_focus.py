#!/usr/bin/env python3
import time
import os
import json
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls, Transform

def compute_sharpness(img_gray):
    """
    Computes focus sharpness score using Variance of Laplacian
    on the central region of interest where the specimen is.
    """
    h, w = img_gray.shape
    # Central 60% ROI to avoid eyepiece borders/vignetting
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    roi = img_gray[y1:y2, x1:x2].astype(np.float32)

    # Discrete Laplacian operator
    lap = (
        roi[1:-1, 2:] + roi[1:-1, :-2] +
        roi[2:, 1:-1] + roi[:-2, 1:-1] -
        4.0 * roi[1:-1, 1:-1]
    )
    # Variance of Laplacian
    var_lap = float(np.var(lap))

    # Tenengrad gradient (Sobel-like)
    gx = np.diff(roi, axis=1)
    gy = np.diff(roi, axis=0)
    tenengrad = float(np.mean(gx**2) + np.mean(gy**2))

    return var_lap, tenengrad

def main():
    output_dir = "/home/thomas/microscope/focus_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Picamera2 for autofocus optimization...")
    picam2 = Picamera2()
    # Use 1920x1080 for sharp preview evaluation, with 180° rotation
    config = picam2.create_video_configuration(
        main={"size": (1920, 1080), "format": "RGB888"},
        transform=Transform(hflip=True, vflip=True)
    )
    picam2.configure(config)
    picam2.start()

    # Step 1: Coarse sweep from 0.0 to 15.0 dioptres (step 0.5)
    coarse_steps = np.linspace(0.0, 15.0, 31)
    results = []

    print(f"\n--- Starting Coarse Sweep ({len(coarse_steps)} positions) ---")
    for pos in coarse_steps:
        pos = float(round(pos, 2))
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": pos})
        # Allow VCM voice coil to settle and grab fresh frame
        for _ in range(5):
            frame = picam2.capture_array()
        
        # Convert RGB to Grayscale
        gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        var_lap, tenengrad = compute_sharpness(gray)
        results.append({
            "pos": pos,
            "var_lap": var_lap,
            "tenengrad": tenengrad
        })
        print(f"LensPosition: {pos:5.2f} dpt | VarLap: {var_lap:8.2f} | Tenengrad: {tenengrad:8.2f}")

    # Find peak coarse position
    best_coarse = max(results, key=lambda x: x["var_lap"])
    print(f"\nCoarse peak at: {best_coarse['pos']} dpt (Score: {best_coarse['var_lap']:.2f})")

    # Step 2: Fine sweep around the coarse peak (±1.0 dpt in steps of 0.1 dpt)
    fine_min = max(0.0, best_coarse["pos"] - 1.0)
    fine_max = min(15.0, best_coarse["pos"] + 1.0)
    fine_steps = np.arange(fine_min, fine_max + 0.05, 0.1)

    print(f"\n--- Starting Fine Sweep from {fine_min:.2f} to {fine_max:.2f} ({len(fine_steps)} positions) ---")
    fine_results = []
    for pos in fine_steps:
        pos = float(round(pos, 2))
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": pos})
        for _ in range(5):
            frame = picam2.capture_array()
        
        gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        var_lap, tenengrad = compute_sharpness(gray)
        fine_results.append({
            "pos": pos,
            "var_lap": var_lap,
            "tenengrad": tenengrad
        })
        print(f"Fine LensPosition: {pos:5.2f} dpt | VarLap: {var_lap:8.2f} | Tenengrad: {tenengrad:8.2f}")

    best_overall = max(fine_results, key=lambda x: x["var_lap"])
    print(f"\n==========================================")
    print(f"OPTIMAL FOCUS FOUND: {best_overall['pos']} dioptres")
    print(f"Peak Sharpness Score: {best_overall['var_lap']:.2f}")
    print(f"==========================================\n")

    # Save comparison sample images (low, coarse peak, fine best, max)
    sample_points = [
        ("focus_0.00_inf.jpg", 0.0),
        ("focus_coarse_peak.jpg", best_coarse["pos"]),
        ("focus_optimal_1080p.jpg", best_overall["pos"]),
        ("focus_15.00_macro.jpg", 15.0)
    ]
    for filename, pos in sample_points:
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(pos)})
        for _ in range(6):
            frame = picam2.capture_array()
        im = Image.fromarray(frame)
        im.save(os.path.join(output_dir, filename), quality=95)
        print(f"Saved comparison frame: {filename}")

    # Stop preview and prepare for full 16MP Still capture at optimal focus
    picam2.stop()
    picam2.close()

    print("\nCapturing FULL 16-MEGAPIXEL STILL (4656x3496) at optimal focus...")
    picam2_still = Picamera2()
    still_cfg = picam2_still.create_still_configuration(
        main={"size": (4656, 3496), "format": "RGB888"},
        transform=Transform(hflip=True, vflip=True)
    )
    picam2_still.configure(still_cfg)
    picam2_still.start()
    picam2_still.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": float(best_overall["pos"])
    })
    time.sleep(1.0) # Settle AE and lens

    final_path = "/home/thomas/microscope/microscope_best_16mp.jpg"
    picam2_still.capture_file(final_path)
    picam2_still.stop()
    picam2_still.close()

    print(f"Final 16MP image saved successfully: {final_path}")

    # Save focus sweep log data
    log_data = {
        "coarse_results": results,
        "fine_results": fine_results,
        "best_coarse": best_coarse,
        "best_overall": best_overall
    }
    with open("/home/thomas/microscope/focus_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    main()

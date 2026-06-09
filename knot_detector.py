"""
Surgical Knot Detection & Distance Measurement
Integrated version for needle analyzer system
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
DURATION_SECONDS = 5        # Analyze last 5 seconds of video
NUM_FRAMES       = 5        # Number of frames to sample
CONF             = 0.1      # Detection confidence threshold
IOU_THRESH       = 0.4      # NMS IoU threshold
SLICE_SIZE       = 1080     # Slice width & height
OVERLAP          = 0.2      # 20% overlap between slices
MODEL_PATH       = "best_knot_detection.pt"  # Knot detection model


def slice_image(image, slice_size, overlap):
    """Slice image into overlapping patches for detection"""
    slices = []
    step = int(slice_size * (1 - overlap))
    h, w = image.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + slice_size, w)
            y2 = min(y + slice_size, h)
            x1 = max(0, x2 - slice_size)
            y1 = max(0, y2 - slice_size)
            crop = image[y1:y2, x1:x2]
            slices.append((x1, y1, crop))
    return slices


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression to filter overlapping detections"""
    if len(boxes) == 0:
        return []
    boxes_arr  = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes_arr[:,0], boxes_arr[:,1], boxes_arr[:,2], boxes_arr[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_arr.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def detect_knots(model, frame):
    """Run sliced inference on a frame, return centers sorted top to bottom"""
    all_boxes  = []
    all_scores = []

    for ox, oy, crop in slice_image(frame, SLICE_SIZE, OVERLAP):
        results = model(crop, conf=CONF, verbose=False)
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                all_scores.append(float(box.conf[0]))

    keep = nms(all_boxes, all_scores, IOU_THRESH)

    # Sort top to bottom by Y center
    keep = sorted(keep, key=lambda i: (all_boxes[i][1] + all_boxes[i][3]) / 2)

    centers = []
    for i in keep:
        x1, y1, x2, y2 = all_boxes[i]
        centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

    return centers


def draw_results(frame, knot_centers, distances, frame_label):
    """Draw knot detection results on frame"""
    vis = frame.copy()

    # Draw small red dots and labels for each knot
    for idx, (cx, cy) in enumerate(knot_centers):
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        label = f"K{idx + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(vis, label, (cx - tw - 10, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw lines between consecutive knots with distance labels
    for i, dist in enumerate(distances):
        c1, c2 = knot_centers[i], knot_centers[i + 1]
        cv2.line(vis, c1, c2, (0, 0, 255), 2, cv2.LINE_AA)
        mx, my = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2
        cv2.putText(vis, f"{dist:.1f}px", (mx + 10, my + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw info panel
    panel_lines = [f"Frame: {frame_label}", f"Knots Detected: {len(knot_centers)}"]
    for i, d in enumerate(distances):
        panel_lines.append(f"  K{i+1} -> K{i+2}: {d:.1f}px")

    line_h, pad = 25, 10
    panel_h = pad * 2 + line_h * len(panel_lines)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)
    for i, txt in enumerate(panel_lines):
        cv2.putText(vis, txt, (pad, pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def analyze_knots_integrated(birdseye_video_path, output_folder):
    """
    Analyze knots in the bird's eye video (last 5 seconds)
    
    Args:
        birdseye_video_path: Path to bird's eye view video
        output_folder: Where to save results
    
    Returns:
        dict with knot analysis results
    """
    print("\n" + "="*60)
    print("KNOT ANALYSIS")
    print("="*60)
    print(f"Processing video: {os.path.basename(birdseye_video_path)}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Knot detection model not found: {MODEL_PATH}")
        print("   Skipping knot analysis")
        return {
            'success': False,
            'error': 'Model not found'
        }
    
    # Load model
    print(f"Loading knot detection model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    
    # Open video
    cap = cv2.VideoCapture(birdseye_video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {birdseye_video_path}")
        return {
            'success': False,
            'error': 'Cannot open video'
        }
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    
    print(f"Video FPS: {fps:.1f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration_s:.1f}s")
    print(f"Analyzing last {DURATION_SECONDS} seconds using {NUM_FRAMES} frames")
    
    # Calculate frame indices to sample
    window_frames = int(DURATION_SECONDS * fps)
    start_frame = max(0, total_frames - window_frames)
    sample_indices = np.linspace(start_frame, total_frames - 1, NUM_FRAMES, dtype=int)
    
    print(f"Frame range: {sample_indices[0]} - {sample_indices[-1]}")
    
    # Process each sampled frame
    all_results = []
    
    for fi, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Could not read frame {frame_idx}, skipping")
            continue
        
        timestamp = frame_idx / fps
        print(f"\n  Frame {fi+1}/{NUM_FRAMES}  (index={frame_idx}, t={timestamp:.2f}s)")
        
        # Detect knots
        knot_centers = detect_knots(model, frame)
        
        # Calculate distances between consecutive knots
        distances = [
            float(np.sqrt((knot_centers[i+1][0] - knot_centers[i][0])**2 +
                          (knot_centers[i+1][1] - knot_centers[i][1])**2))
            for i in range(len(knot_centers) - 1)
        ]
        
        print(f"    Knots detected: {len(knot_centers)}")
        for i, (cx, cy) in enumerate(knot_centers):
            print(f"      K{i+1}: ({cx}, {cy})")
        for i, d in enumerate(distances):
            print(f"      K{i+1} -> K{i+2}: {d:.1f}px")
        
        # Draw results
        label = f"t={timestamp:.2f}s (frame {frame_idx})"
        vis_frame = draw_results(frame, knot_centers, distances, label)
        
        all_results.append({
            "frame_num": fi + 1,
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "knot_centers": knot_centers,
            "distances": distances,
            "vis_frame": vis_frame
        })
    
    cap.release()
    
    # Select best frame (most knots detected)
    if not all_results:
        print("⚠️  No frames processed")
        return {
            'success': False,
            'error': 'No frames processed'
        }
    
    best = max(all_results, key=lambda r: len(r["knot_centers"]))
    
    # Save best frame
    video_name = os.path.splitext(os.path.basename(birdseye_video_path))[0]
    output_filename = f"{video_name}_knot_analysis.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, best["vis_frame"])
    
    print("\n" + "="*60)
    print("BEST FRAME SELECTED")
    print("="*60)
    print(f"Frame: {best['frame_num']}  |  t={best['timestamp']:.2f}s")
    print(f"Knots detected: {len(best['knot_centers'])}")
    for i, (cx, cy) in enumerate(best["knot_centers"]):
        print(f"  K{i+1}: ({cx}, {cy})")
    for i, d in enumerate(best["distances"]):
        print(f"  K{i+1} -> K{i+2}: {d:.1f}px")
    print(f"\n✅ Saved: {output_filename}")
    print("="*60)
    
    return {
        'success': True,
        'output_file': output_filename,
        'num_knots': len(best['knot_centers']),
        'knot_centers': best['knot_centers'],
        'distances': best['distances'],
        'timestamp': best['timestamp']
    }

from ultralytics import YOLO
import cv2
import numpy as np
import os
import re

def parse_suture_times(txt_file_path):
    """
    Parse the holding segments TXT file to extract suture times
    
    Returns:
        List of tuples: [(suture_num, start_time, end_time, duration), ...]
    """
    suture_times = []
    
    with open(txt_file_path, 'r') as f:
        content = f.read()
    
    # Find the SUTURE TIME ANALYSIS section
    suture_section = content.split("SUTURE TIME ANALYSIS:")
    
    if len(suture_section) < 2:
        return []
    
    # Extract suture lines
    lines = suture_section[1].split('\n')
    
    for line in lines:
        # Match pattern: Suture 1: 0.00s - 59.33s (Duration: 59.33s)
        match = re.match(r'Suture (\d+):\s*([\d.]+)s\s*-\s*([\d.]+)s\s*\(Duration:\s*([\d.]+)s\)', line)
        if match:
            suture_num = int(match.group(1))
            start_time = float(match.group(2))
            end_time = float(match.group(3))
            duration = float(match.group(4))
            suture_times.append((suture_num, start_time, end_time, duration))
    
    return suture_times

def find_max_needle_length(video_path, model):
    """
    Find maximum needle length in the video
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_needle_length = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on the frame
        results = model(frame, verbose=False, classes=[0, 3])
        
        # Process each result
        for result in results:
            if result.masks is not None and result.boxes is not None:
                masks_xy = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()
                class_names = result.names
                
                for idx, (mask_coords, cls_id) in enumerate(zip(masks_xy, classes)):
                    class_name = class_names[int(cls_id)]
                    
                    if class_name.lower() == 'needle':
                        if len(mask_coords) > 1:
                            points = mask_coords.astype(np.int32)
                            
                            # Find the starting point
                            scores = points[:, 0] + points[:, 1] * 0.5
                            start_point_idx = np.argmin(scores)
                            start_point = points[start_point_idx]
                            
                            # Calculate distances from start point to all other points
                            distances_from_start = np.sqrt(
                                (points[:, 0] - start_point[0])**2 + 
                                (points[:, 1] - start_point[1])**2
                            )
                            
                            current_max_length = np.max(distances_from_start)
                            
                            if current_max_length > max_needle_length:
                                max_needle_length = current_max_length
    
    cap.release()
    return max_needle_length

def process_suture_segment(video_path, model, suture_num, start_time, max_needle_length):
    """
    Process first 7 seconds of a suture segment to find frame closest to 33% holding
    
    Returns:
        Tuple: (frame_number, holding_percentage, frame_image)
    """
    colors = {
        'Needle': (0, 255, 0),
        'Needle_holder': (255, 0, 0)
    }
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int((start_time + 7) * fps)
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    closest_to_33_frame = None
    closest_to_33_percentage = None
    closest_diff = float('inf')
    closest_frame_number = None
    
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference on the frame
        results = model(frame, verbose=False)
        
        # Store data for overlap detection
        needle_mask = np.zeros((height, width), dtype=np.uint8)
        needle_holder_mask = np.zeros((height, width), dtype=np.uint8)
        needle_start_point = None
        
        # Process each result
        for result in results:
            if result.masks is not None and result.boxes is not None:
                masks_xy = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()
                class_names = result.names
                
                for idx, (mask_coords, cls_id) in enumerate(zip(masks_xy, classes)):
                    class_name = class_names[int(cls_id)]
                    color = colors.get(class_name, (255, 255, 255))
                    
                    if len(mask_coords) > 0:
                        points = mask_coords.astype(np.int32)
                        contour = points.reshape(-1, 1, 2)
                        
                        # Fill mask for this object
                        if class_name.lower() == 'needle':
                            cv2.fillPoly(needle_mask, [points], 255)
                        elif class_name.lower() == 'needle_holder':
                            cv2.fillPoly(needle_holder_mask, [points], 255)
                        
                        # Draw contour
                        cv2.drawContours(frame, [contour], -1, color, 2)
                        
                        # Add label
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.putText(frame, class_name, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Store needle starting point
                        if class_name.lower() == 'needle':
                            scores = points[:, 0] + points[:, 1] * 0.5
                            start_point_idx = np.argmin(scores)
                            needle_start_point = points[start_point_idx]
                            
                            # Draw starting point
                            cv2.circle(frame, tuple(needle_start_point), 6, (255, 255, 0), -1)
                            cv2.circle(frame, tuple(needle_start_point), 8, (255, 255, 255), 2)
        
        # Find overlap and calculate percentage
        grip_percentage = 0
        if needle_start_point is not None:
            overlap_mask = cv2.bitwise_and(needle_mask, needle_holder_mask)
            overlap_coords = np.where(overlap_mask > 0)
            
            if len(overlap_coords[0]) > 0:
                overlap_y = overlap_coords[0]
                overlap_x = overlap_coords[1]
                
                # Find the lowest point
                lowest_point_idx = np.argmax(overlap_y)
                grip_point_y = int(overlap_y[lowest_point_idx])
                grip_point_x = int(overlap_x[lowest_point_idx])
                
                # Draw red dot at gripping point
                cv2.circle(frame, (grip_point_x, grip_point_y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (grip_point_x, grip_point_y), 10, (255, 255, 255), 2)
                
                # Calculate distance from start point to grip point
                distance_to_grip = np.sqrt(
                    (grip_point_x - needle_start_point[0])**2 + 
                    (grip_point_y - needle_start_point[1])**2
                )
                
                # Calculate percentage
                if max_needle_length > 0:
                    grip_percentage = (distance_to_grip / max_needle_length) * 100
                    grip_percentage = min(grip_percentage, 100)
                
                # Draw line from start point to grip point
                cv2.line(frame, tuple(needle_start_point), (grip_point_x, grip_point_y), 
                        (255, 0, 255), 2)
        
        # Display percentage on frame
        if needle_start_point is not None:
            text = f"Holding: {grip_percentage:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            text_x = 10
            text_y = 40
            
            # Black background
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
            
            # Check if this is closest to 33%
            diff = abs(grip_percentage - 33.0)
            if diff < closest_diff:
                closest_diff = diff
                closest_to_33_percentage = grip_percentage
                closest_to_33_frame = frame.copy()
                closest_frame_number = current_frame
        
        current_frame += 1
    
    cap.release()
    
    return closest_frame_number, closest_to_33_percentage, closest_to_33_frame

def extract_33_percent_frames(video_path, txt_file_path, output_folder, model_path="best_v3.pt", progress_callback=None):
    """
    Main function to extract 33% holding frames for each suture
    
    Args:
        video_path: Path to video file
        txt_file_path: Path to holding segments TXT file
        output_folder: Folder to save output images
        model_path: Path to YOLO model
        progress_callback: Optional callback function(progress, message)
    
    Returns:
        List of result dictionaries
    """
    def update_progress(percent, message):
        if progress_callback:
            progress_callback(percent, message)
    
    # Load model
    update_progress(5, "Loading YOLO model...")
    model = YOLO(model_path)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse suture times
    update_progress(10, "Reading suture times...")
    suture_times = parse_suture_times(txt_file_path)
    
    if len(suture_times) == 0:
        update_progress(100, "No suture times found")
        return []
    
    # Find maximum needle length
    update_progress(15, "Finding maximum needle length...")
    max_needle_length = find_max_needle_length(video_path, model)
    
    # Process each suture
    results_summary = []
    total_sutures = len(suture_times)
    
    for idx, (suture_num, start_time, end_time, duration) in enumerate(suture_times):
        progress = 15 + int((idx / total_sutures) * 70)
        update_progress(progress, f"Processing Suture {suture_num}...")
        
        frame_num, holding_pct, frame_img = process_suture_segment(
            video_path, model, suture_num, start_time, max_needle_length
        )
        
        if frame_img is not None:
            # Resize image for web display (max width: 1280px, maintain aspect ratio)
            height, width = frame_img.shape[:2]
            max_width = 1280
            
            if width > max_width:
                # Calculate new dimensions
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                
                # Resize image
                frame_img_resized = cv2.resize(frame_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                frame_img_resized = frame_img
            
            # Save the resized frame
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_Suture_{suture_num}_33percent.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, frame_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            results_summary.append({
                'suture_num': suture_num,
                'frame_number': frame_num,
                'holding_percentage': holding_pct,
                'output_file': output_filename
            })
    
    # Save summary report
    update_progress(90, "Saving summary report...")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    summary_file = os.path.join(output_folder, f"{base_name}_33percent_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("33% HOLDING FRAME EXTRACTION SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Maximum needle length: {max_needle_length:.2f} pixels\n")
        f.write(f"Total sutures processed: {len(suture_times)}\n")
        f.write("="*60 + "\n\n")
        
        f.write("RESULTS:\n")
        f.write("-"*60 + "\n")
        
        for result in results_summary:
            f.write(f"Suture {result['suture_num']}:\n")
            f.write(f"  Frame number: {result['frame_number']}\n")
            f.write(f"  Holding percentage: {result['holding_percentage']:.1f}%\n")
            f.write(f"  Saved as: {result['output_file']}\n")
            f.write("\n")
    
    update_progress(100, "Complete!")
    
    return results_summary


def extract_33_percent_frames_integrated(video_path, output_folder, suture_times, model, max_needle_length):
    """
    Integrated version for use within analyzer.py
    Uses already loaded model and calculated max_needle_length
    
    Args:
        video_path: Path to video file
        output_folder: Folder to save output images
        suture_times: List of tuples [(start_time, end_time, duration), ...]
        model: Already loaded YOLO model
        max_needle_length: Pre-calculated maximum needle length
    
    Returns:
        List of result dictionaries
    """
    results_summary = []
    
    # Process each suture
    for idx, (start_time, end_time, duration) in enumerate(suture_times):
        suture_num = idx + 1
        
        frame_num, holding_pct, frame_img = process_suture_segment(
            video_path, model, suture_num, start_time, max_needle_length
        )
        
        if frame_img is not None:
            # Resize image for web display (max width: 1280px, maintain aspect ratio)
            height, width = frame_img.shape[:2]
            max_width = 1280
            
            if width > max_width:
                # Calculate new dimensions
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                
                # Resize image
                frame_img_resized = cv2.resize(frame_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                frame_img_resized = frame_img
            
            # Save the resized frame
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = f"{base_name}_Suture_{suture_num}_33percent.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, frame_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            results_summary.append({
                'suture_num': suture_num,
                'frame_number': frame_num,
                'holding_percentage': holding_pct,
                'output_file': output_filename
            })
    
    return results_summary

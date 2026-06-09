from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import re

# ===== VALIDATION CONFIGURATION =====
VALIDATION_WINDOW_SEC = 6  # Check for horizontal orientation within 6 seconds (was 4)
HORIZONTAL_THRESHOLD = 30  # Degrees from horizontal (0-30 degrees is considered horizontal)
MIN_HORIZONTAL_DURATION_SEC = 0.2  # Needle must stay horizontal for at least 0.2 seconds (was 0.5 → ~6 frames at 30fps)
CONSECUTIVE_FRAMES_THRESHOLD = 3  # Needle tip must be inside bbox for at least 3 consecutive frames
HORIZONTAL_EXPANSION_PERCENT = 100  # Horizontal expansion
VERTICAL_EXPANSION_PERCENT = 50    # Vertical expansion percentage
INCISION_DETECTION_FRAMES = 150  # Detect incisions in first 150 frames (~5 seconds at 30fps)
# ====================================

def parse_holding_segments(txt_file_path):
    """
    Parse holding segments from TXT file
    
    Returns:
        List of tuples: [(segment_num, start_time, end_time, duration), ...]
    """
    segments = []
    
    with open(txt_file_path, 'r') as f:
        content = f.read()
    
    # Find the HOLDING SEGMENTS section
    segments_section = content.split("HOLDING SEGMENTS:")
    
    if len(segments_section) < 2:
        return []
    
    # Extract segment lines
    lines = segments_section[1].split('\n')
    
    for line in lines:
        # Match pattern: Segment 1: 1.67s - 17.30s (Duration: 15.63s)
        match = re.match(r'Segment (\d+):\s*([\d.]+)s\s*-\s*([\d.]+)s\s*\(Duration:\s*([\d.]+)s\)', line)
        if match:
            segment_num = int(match.group(1))
            start_time = float(match.group(2))
            end_time = float(match.group(3))
            duration = float(match.group(4))
            segments.append((segment_num, start_time, end_time, duration))
    
    return segments

def extend_segment_to_min_duration(start_time, end_time, min_duration=10.0, video_duration=None):
    """
    Extend segment to minimum duration if needed
    
    Args:
        start_time: Segment start time
        end_time: Segment end time
        min_duration: Minimum duration (default 10 seconds)
        video_duration: Total video duration (to avoid exceeding)
    
    Returns:
        (extended_start, extended_end)
    """
    current_duration = end_time - start_time
    
    if current_duration >= min_duration:
        # Already long enough
        return start_time, end_time
    else:
        # Extend end time
        extended_end = start_time + min_duration
        
        # Make sure we don't exceed video duration
        if video_duration is not None and extended_end > video_duration:
            extended_end = video_duration
        
        return start_time, extended_end

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    box format: (x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def is_box_contained(box1, box2, threshold=0.9):
    """
    Check if box1 is contained within box2.
    Returns True if box1's area overlaps with box2 by more than threshold.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    
    # If box1 is mostly inside box2
    overlap_ratio = intersection_area / box1_area if box1_area > 0 else 0.0
    
    return overlap_ratio > threshold

def remove_overlapping_boxes(boxes, overlap_threshold=0.85):
    """
    Remove boxes that are completely contained within other boxes.
    Keep the larger box in case of overlap.
    """
    if len(boxes) <= 1:
        return boxes
    
    # Sort boxes by area (descending) - keep larger boxes
    boxes_with_area = [(box, box[2] * box[3]) for box in boxes]
    boxes_with_area.sort(key=lambda x: x[1], reverse=True)
    
    filtered_boxes = []
    
    for i, (box1, area1) in enumerate(boxes_with_area):
        is_contained = False
        
        # Check if this box is contained in any of the already kept boxes
        for box2 in filtered_boxes:
            if is_box_contained(box1, box2, overlap_threshold):
                is_contained = True
                break
        
        if not is_contained:
            filtered_boxes.append(box1)
    
    return filtered_boxes

def calculate_needle_angle(mask_coords):
    """Calculate angle of needle from horizontal"""
    if len(mask_coords) < 2:
        return None
    
    points = mask_coords.astype(np.float32)
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    angle_rad = math.atan2(vy, vx)
    angle_deg = abs(math.degrees(angle_rad))
    
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg

def get_needle_tip(mask_coords):
    """Get needle tip (bottom-right most point)"""
    if len(mask_coords) == 0:
        return None
    
    scores = mask_coords[:, 1] * 2 + mask_coords[:, 0]
    tip_idx = np.argmax(scores)
    tip_point = mask_coords[tip_idx]
    
    return tuple(tip_point.astype(int))

def is_point_in_bbox(point, bbox):
    """
    Check if a point (x, y) is inside a bounding box (x, y, w, h).
    """
    if point is None:
        return False
    
    px, py = point
    x, y, w, h = bbox
    
    return x <= px <= (x + w) and y <= py <= (y + h)

def get_needle_line_points(mask_coords, tip_point, line_length=150):
    """Calculate needle trajectory line from tip"""
    if len(mask_coords) < 2 or tip_point is None:
        return None, None
    
    points = mask_coords.astype(np.float32)
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    
    direction = np.array([vx[0], vy[0]])
    direction = direction / np.linalg.norm(direction)
    
    start_pt = tuple(tip_point)
    end_x = int(tip_point[0] - direction[0] * line_length)
    end_y = int(tip_point[1] - direction[1] * line_length)
    end_pt = (end_x, end_y)
    
    return start_pt, end_pt

def detect_incisions_in_segment(video_path, model, start_time, end_time, fps):
    """
    Detect incision boxes in the early part of the segment
    
    Returns:
        List of incision boxes: [(x, y, w, h), ...]
    """
    cap = cv2.VideoCapture(video_path)
    
    # Look for incisions in first few seconds of segment
    search_start_frame = int(start_time * fps)
    search_end_frame = min(int(end_time * fps), search_start_frame + INCISION_DETECTION_FRAMES)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, search_start_frame)
    
    detected_incisions = []
    current_frame = search_start_frame
    
    print(f"    Searching for incisions in frames {search_start_frame} to {search_end_frame}...")
    
    while current_frame <= search_end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        
        for result in results:
            if result.masks is not None and result.boxes is not None:
                masks_xy = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()
                class_names = result.names
                
                for idx, (mask_coords, cls_id) in enumerate(zip(masks_xy, classes)):
                    class_name = class_names[int(cls_id)]
                    
                    if class_name.lower() == 'incisions' and len(mask_coords) > 0:
                        points = mask_coords.astype(np.int32)
                        contour = points.reshape(-1, 1, 2)
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if similar box already exists
                        box_exists = False
                        for stored_box in detected_incisions:
                            sx, sy, sw, sh = stored_box
                            if abs(x - sx) < 80 and abs(y - sy) < 80:
                                box_exists = True
                                break
                        
                        if not box_exists:
                            detected_incisions.append((x, y, w, h))
                            print(f"    Found incision at frame {current_frame}: ({x}, {y}), size: {w}x{h}")
        
        current_frame += 1
    
    cap.release()
    
    # Remove overlapping boxes
    if len(detected_incisions) > 1:
        print(f"    Removing overlapping incision boxes...")
        detected_incisions = remove_overlapping_boxes(detected_incisions, overlap_threshold=0.85)
    
    print(f"    Total incisions detected: {len(detected_incisions)}")
    
    return detected_incisions

def process_segment_for_insertion_angle(video_path, model, segment_num, start_time, end_time,
                                        output_folder=None, save_debug_video=False,
                                        fallback_incision_boxes=None):
    """
    Process a segment to detect needle insertion angle with proper validation
    Uses the same logic as main_v5.py

    Args:
        output_folder:          Where to save debug video (if save_debug_video=True)
        save_debug_video:       If True, saves annotated debug video
        fallback_incision_boxes: Pre-scanned incision boxes from the global video-start
                                 scan; used when this segment's own incision scan finds
                                 nothing (wound may already have sutures and look different)

    Returns:
        dict with: angle, frame_number, frame_image, detected (True/False), validation_info
    """
    colors = {
        'Needle': (0, 255, 0),
        'Needle_holder': (255, 0, 0),
        'incisions': (0, 0, 255)
    }
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if debug mode enabled
    video_writer = None
    if save_debug_video and output_folder:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        debug_video_filename = f"{base_name}_Segment_{segment_num}_insertion_DEBUG.mp4"
        debug_video_path = os.path.join(output_folder, debug_video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))
        
        print(f"  📹 DEBUG VIDEO: Saving to {debug_video_filename}")
    
    print(f"\n  Segment {segment_num}: Detecting incisions...")
    
    # Step 1: Detect incisions in this segment
    incision_boxes = detect_incisions_in_segment(video_path, model, start_time, end_time, fps)
    
    if len(incision_boxes) == 0 and fallback_incision_boxes:
        print(f"  ↩  No incisions at segment start — using {len(fallback_incision_boxes)} "
              f"globally-detected box(es)")
        incision_boxes = list(fallback_incision_boxes)

    if len(incision_boxes) == 0:
        print(f"  ⚠️  No incisions detected in segment {segment_num}")
        print(f"      Will measure angle without incision validation (fallback mode)")
        # Fallback to simple detection without validation
        return process_segment_simple_fallback(video_path, model, segment_num, start_time, end_time)
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Tracking variables for insertion validation
    insertion_candidates = []
    confirmed_insertions = []  # Changed to list to support multiple insertions per segment
    
    current_frame = start_frame
    
    print(f"  Processing frames {start_frame} to {end_frame} for insertion validation...")
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create annotated frame for debug video if enabled
        if video_writer:
            annotated_frame = frame.copy()
        
        # Run inference
        results = model(frame, verbose=False)
        
        needle_mask_coords = None
        needle_tip = None
        needle_angle = None
        
        # Process results - just collect data, drawing happens at end of loop
        for result in results:
            if result.masks is not None and result.boxes is not None:
                masks_xy = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()
                class_names = result.names
                
                for idx, (mask_coords, cls_id) in enumerate(zip(masks_xy, classes)):
                    class_name = class_names[int(cls_id)]
                    
                    if len(mask_coords) > 0:
                        # Store needle data
                        if class_name.lower() == 'needle':
                            needle_mask_coords = mask_coords
                            needle_tip = get_needle_tip(mask_coords)
                            needle_angle = calculate_needle_angle(mask_coords)
        
        # Insertion validation logic (same as main_v5.py)
        if needle_tip is not None and needle_angle is not None:
            for inc_idx, incision_box in enumerate(incision_boxes):
                # Check if this specific incision already has a confirmed insertion
                already_confirmed = any(ins['incision_idx'] == inc_idx for ins in confirmed_insertions)
                if already_confirmed:
                    continue  # Skip this incision, check next one
                
                # Calculate extended bbox
                x, y, w, h = incision_box
                extra_width = int(w * (HORIZONTAL_EXPANSION_PERCENT / 100))
                half_extra_width = extra_width // 2
                extra_height = int(h * (VERTICAL_EXPANSION_PERCENT / 100))
                half_extra_height = extra_height // 2

                new_x = max(0, x - half_extra_width)
                new_y = max(0, y - half_extra_height)
                new_w = w + extra_width
                new_h = h + extra_height

                # Keep within frame boundaries
                if new_x + new_w > width:
                    new_w = width - new_x
                if new_y + new_h > height:
                    new_h = height - new_y

                extended_bbox = (new_x, new_y, new_w, new_h)
                
                tip_inside = is_point_in_bbox(needle_tip, extended_bbox)
                
                # Find existing candidate
                candidate_idx = None
                for i, candidate in enumerate(insertion_candidates):
                    if candidate['incision_idx'] == inc_idx:
                        candidate_idx = i
                        break
                
                if tip_inside:
                    if candidate_idx is None:
                        # CAPTURE ENTRY ANGLE IMMEDIATELY AT FIRST CONTACT
                        insertion_candidates.append({
                            'incision_idx': inc_idx,
                            'trigger_frame': current_frame,
                            'consecutive_inside_frames': 1,
                            'horizontal_start_frame': None,
                            'horizontal_frame_count': 0,
                            'entry_angle': needle_angle,
                            'entry_tip': needle_tip,
                            'entry_incision_box': incision_box,
                            'entry_mask_coords': needle_mask_coords
                        })
                        trigger_time = current_frame / fps
                        print(f"    Frame {current_frame} ({trigger_time:.2f}s): Needle tip entered incision")
                        print(f"      Entry angle captured: {needle_angle:.1f}°")
                    else:
                        insertion_candidates[candidate_idx]['consecutive_inside_frames'] += 1
                    
                    # Check validation
                    current_candidate_idx = candidate_idx if candidate_idx is not None else len(insertion_candidates) - 1
                    
                    if insertion_candidates[current_candidate_idx]['consecutive_inside_frames'] >= CONSECUTIVE_FRAMES_THRESHOLD:
                        is_horizontal = needle_angle <= HORIZONTAL_THRESHOLD
                        
                        if is_horizontal:
                            if insertion_candidates[current_candidate_idx]['horizontal_start_frame'] is None:
                                insertion_candidates[current_candidate_idx]['horizontal_start_frame'] = current_frame
                                insertion_candidates[current_candidate_idx]['horizontal_frame_count'] = 1
                                print(f"    Frame {current_frame}: Needle became horizontal ({needle_angle:.1f}°)")
                            else:
                                insertion_candidates[current_candidate_idx]['horizontal_frame_count'] += 1
                                
                                horizontal_duration = insertion_candidates[current_candidate_idx]['horizontal_frame_count'] / fps
                                if horizontal_duration >= MIN_HORIZONTAL_DURATION_SEC:
                                    # CONFIRMED INSERTION!
                                    trigger_frame = insertion_candidates[current_candidate_idx]['trigger_frame']
                                    entry_angle = insertion_candidates[current_candidate_idx]['entry_angle']
                                    entry_tip = insertion_candidates[current_candidate_idx]['entry_tip']
                                    entry_mask = insertion_candidates[current_candidate_idx]['entry_mask_coords']
                                    
                                    print(f"    ✅ INSERTION CONFIRMED at frame {current_frame}")
                                    print(f"       Entry angle: {entry_angle:.1f}° (captured at frame {trigger_frame})")
                                    
                                    confirmed_insertions.append({
                                        'incision_idx': inc_idx,
                                        'trigger_frame': trigger_frame,
                                        'confirmation_frame': current_frame,
                                        'entry_angle': entry_angle,
                                        'entry_tip': entry_tip,
                                        'incision_box': incision_box,
                                        'entry_mask_coords': entry_mask
                                    })
                                    
                                    # Don't break - continue checking other incisions
                                    break  # Break from validation loop for this incision only
                        else:
                            elapsed_frames = current_frame - insertion_candidates[current_candidate_idx]['trigger_frame']
                            elapsed_time = elapsed_frames / fps
                            
                            if elapsed_time > VALIDATION_WINDOW_SEC:
                                print(f"    ❌ Candidate rejected - no horizontal within {VALIDATION_WINDOW_SEC}s")
                                insertion_candidates.pop(current_candidate_idx)
                # else:
                #     if candidate_idx is not None:
                #         print(f"    ❌ Candidate rejected - needle left bbox")
                #         insertion_candidates.pop(candidate_idx)

                else:
                    if candidate_idx is not None:
                        # Allow 1-2 frame exits before rejecting (handles tracking jitter)
                        cand = insertion_candidates[candidate_idx]
                        cand['consecutive_outside'] = cand.get('consecutive_outside', 0) + 1
                        if cand['consecutive_outside'] >= 3:
                            print(f"    ❌ Candidate rejected - needle left bbox")
                            insertion_candidates.pop(candidate_idx)
                        else:
                            continue  # Stay in candidate, wait to see if it re-enters
        
        # =================================================================
        # DRAW ANNOTATIONS FOR DEBUG VIDEO (at end of loop, after all logic)
        # =================================================================
        if video_writer:
            # Draw all object detections (needle, holder, etc.) with contours
            for result in results:
                if result.masks is not None and result.boxes is not None:
                    masks_xy = result.masks.xy
                    classes = result.boxes.cls.cpu().numpy()
                    class_names = result.names
                    
                    for idx, (mask_coords, cls_id) in enumerate(zip(masks_xy, classes)):
                        class_name = class_names[int(cls_id)]
                        color = colors.get(class_name, (255, 255, 255))
                        
                        if len(mask_coords) > 0 and class_name.lower() != 'incisions':
                            points = mask_coords.astype(np.int32)
                            contour = points.reshape(-1, 1, 2)
                            cv2.drawContours(annotated_frame, [contour], -1, color, 2)
                            
                            x, y, w, h = cv2.boundingRect(contour)
                            cv2.putText(annotated_frame, class_name, (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw needle tip (magenta dot)
            if needle_tip:
                cv2.circle(annotated_frame, needle_tip, 8, (255, 0, 255), -1)
            
            # Draw incision boxes
            for inc_idx, incision_box in enumerate(incision_boxes):
                x, y, w, h = incision_box
                
                # Calculate expanded detection zone
                extra_width = int(w * (HORIZONTAL_EXPANSION_PERCENT / 100))
                half_extra_width = extra_width // 2
                extra_height = int(h * (VERTICAL_EXPANSION_PERCENT / 100))
                half_extra_height = extra_height // 2
                
                new_x = max(0, x - half_extra_width)
                new_y = max(0, y - half_extra_height)
                new_w = w + extra_width
                new_h = h + extra_height
                
                # Keep within frame boundaries
                if new_x + new_w > width:
                    new_w = width - new_x
                if new_y + new_h > height:
                    new_h = height - new_y
                
                # Draw expanded detection zone (thin orange/yellow box)
                cv2.rectangle(annotated_frame, (new_x, new_y), (new_x + new_w, new_y + new_h), 
                             (0, 165, 255), 2)  # Orange
                cv2.putText(annotated_frame, "Detection Zone", (new_x, new_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
                # Check if this incision has a confirmed insertion
                has_confirmed = any(ins['incision_idx'] == inc_idx for ins in confirmed_insertions)
                
                # Draw original incision box (thick solid - red = not inserted, green = inserted)
                if has_confirmed:
                    box_color = (0, 255, 0)  # Green
                    label = "INSERTED!"
                else:
                    box_color = (0, 0, 255)  # Red  
                    label = "incisions"
                
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 3)
                cv2.putText(annotated_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw angle visualization for ALL confirmed insertions
            y_offset = 40
            for confirmed_insertion in confirmed_insertions:
                entry_tip = confirmed_insertion['entry_tip']
                entry_mask = confirmed_insertion['entry_mask_coords']
                
                if entry_tip is not None and entry_mask is not None and len(entry_mask) > 0:
                    start_pt, end_pt = get_needle_line_points(entry_mask, entry_tip, line_length=150)
                    
                    if start_pt is not None and end_pt is not None:
                        # Draw purple/magenta trajectory line
                        cv2.line(annotated_frame, start_pt, end_pt, (255, 0, 255), 3)
                        
                        # Find lowest point (highest y-value)
                        if start_pt[1] >= end_pt[1]:
                            lowest_pt = start_pt
                        else:
                            lowest_pt = end_pt
                        
                        # Calculate line length
                        # import math
                        line_len = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                        
                        # Draw yellow horizontal reference line (same length, centered at lowest point)
                        half_len = line_len / 2
                        yellow_start = (int(lowest_pt[0] - half_len), lowest_pt[1])
                        yellow_end = (int(lowest_pt[0] + half_len), lowest_pt[1])
                        cv2.line(annotated_frame, yellow_start, yellow_end, (0, 255, 255), 3)
                        
                        # Calculate absolute angle (0-180°)
                        dx = end_pt[0] - start_pt[0]
                        dy = end_pt[1] - start_pt[1]
                        
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        
                        if angle_deg < 0:
                            angle_deg += 360
                        absolute_angle = angle_deg % 180
                        
                        # Draw angle arc (centered at lowest point)
                        arc_radius = 40
                        cv2.ellipse(annotated_frame, lowest_pt, (arc_radius, arc_radius), 0, 0, 
                                   -int(absolute_angle), (0, 255, 255), 2)
                        
                        # Display angle text with black background
                        incision_num = confirmed_insertion['incision_idx'] + 1
                        angle_text = f"Incision #{incision_num} Entry Angle: {absolute_angle:.1f}"
                        text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                        
                        cv2.rectangle(annotated_frame, (10, y_offset - 35), (20 + text_size[0], y_offset + 10), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, angle_text, (15, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        y_offset += 60  # Move down for next insertion text
            
            # Frame number at bottom
            cv2.putText(annotated_frame, f"Frame: {current_frame}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to debug video if enabled
        if video_writer:
            video_writer.write(annotated_frame)
        
        current_frame += 1
        
        # Continue processing full duration - don't break early
        # There might be multiple insertions in the same segment
    
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"  ✅ DEBUG VIDEO saved successfully")
    
    # Generate output frame with visualization
    if len(confirmed_insertions) > 0:
        # Use the first confirmed insertion for the still image
        confirmed_insertion = confirmed_insertions[0]
        
        # Create visualization frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, confirmed_insertion['trigger_frame'])
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Draw visualization on the entry frame
            entry_mask = confirmed_insertion['entry_mask_coords']
            entry_tip = confirmed_insertion['entry_tip']
            
            # Draw needle contour
            if len(entry_mask) > 0:
                points = entry_mask.astype(np.int32)
                contour = points.reshape(-1, 1, 2)
                cv2.drawContours(frame, [contour], -1, colors['Needle'], 2)
            
            # Get needle trajectory line
            start_pt, end_pt = get_needle_line_points(entry_mask, entry_tip, line_length=150)
            
            if start_pt is not None and end_pt is not None:
                # Draw purple trajectory line
                cv2.line(frame, start_pt, end_pt, (255, 0, 255), 3)
                
                # Find lowest point
                if start_pt[1] >= end_pt[1]:
                    lowest_pt = start_pt
                else:
                    lowest_pt = end_pt
                
                # Calculate line length
                line_len = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                
                # Draw yellow horizontal reference line
                half_len = line_len / 2
                yellow_start = (int(lowest_pt[0] - half_len), lowest_pt[1])
                yellow_end = (int(lowest_pt[0] + half_len), lowest_pt[1])
                cv2.line(frame, yellow_start, yellow_end, (0, 255, 255), 3)
                
                # Calculate angle
                dx = end_pt[0] - start_pt[0]
                dy = end_pt[1] - start_pt[1]
                
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                
                if angle_deg < 0:
                    angle_deg += 360
                absolute_angle = angle_deg % 180
                
                # Draw angle arc
                arc_radius = 40
                cv2.ellipse(frame, lowest_pt, (arc_radius, arc_radius), 0, 0, 
                           -int(absolute_angle), (0, 255, 255), 2)
                
                # Draw angle text
                angle_text = f"Entry Angle: {confirmed_insertion['entry_angle']:.1f} (VALIDATED)"
                text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(frame, (10, 5), (20 + text_size[0], 50), (0, 0, 0), -1)
                cv2.putText(frame, angle_text, (15, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return {
            'detected': True,
            'angle': confirmed_insertion['entry_angle'],
            'frame_number': confirmed_insertion['trigger_frame'],
            'frame_image': frame,
            'validated': True,
            'validation_method': 'incision_entry_with_horizontal_check'
        }
    else:
        print(f"  ⚠️  No validated insertion found in segment {segment_num}")
        return {
            'detected': False,
            'angle': None,
            'frame_number': None,
            'frame_image': None,
            'validated': False,
            'validation_method': 'none'
        }

def process_segment_simple_fallback(video_path, model, segment_num, start_time, end_time):
    """
    Fallback method when no incisions detected - captures FIRST angle instead of last
    This is more likely to be the entry angle than the last angle
    """
    colors = {
        'Needle': (0, 255, 0),
        'Needle_holder': (255, 0, 0)
    }

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame range - look at first 10 seconds of segment
    # (extended from 3s so needle is more likely to be visible)
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), start_frame + int(10 * fps))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    first_angle_frame = None
    first_angle_value = None
    first_frame_number = None
    angle_detected = False
    
    current_frame = start_frame
    
    print(f"  Fallback mode: Looking for FIRST angle in first 3 seconds...")
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        
        needle_mask_coords = None
        needle_tip = None
        
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
                        
                        cv2.drawContours(frame, [contour], -1, color, 2)
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.putText(frame, class_name, (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if class_name.lower() == 'needle':
                            needle_mask_coords = mask_coords
                            needle_tip = get_needle_tip(mask_coords)
        
        # Capture FIRST angle detected, not last
        if needle_mask_coords is not None and needle_tip is not None and len(needle_mask_coords) > 0 and not angle_detected:
            start_pt, end_pt = get_needle_line_points(needle_mask_coords, needle_tip, line_length=150)
            
            if start_pt is not None and end_pt is not None:
                cv2.line(frame, start_pt, end_pt, (255, 0, 255), 3)
                
                if start_pt[1] >= end_pt[1]:
                    lowest_pt = start_pt
                else:
                    lowest_pt = end_pt
                
                line_len = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                
                half_len = line_len / 2
                yellow_start = (int(lowest_pt[0] - half_len), lowest_pt[1])
                yellow_end = (int(lowest_pt[0] + half_len), lowest_pt[1])
                cv2.line(frame, yellow_start, yellow_end, (0, 255, 255), 3)
                
                dx = end_pt[0] - start_pt[0]
                dy = end_pt[1] - start_pt[1]
                
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                
                if angle_deg < 0:
                    angle_deg += 360
                absolute_angle = angle_deg % 180
                
                arc_radius = 40
                cv2.ellipse(frame, lowest_pt, (arc_radius, arc_radius), 0, 0, 
                           -int(absolute_angle), (0, 255, 255), 2)
                
                angle_text = f"Entry Angle: {absolute_angle:.1f} (UNVALIDATED)"
                text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(frame, (10, 5), (20 + text_size[0], 50), (0, 0, 0), -1)
                cv2.putText(frame, angle_text, (15, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)  # Orange for unvalidated
                
                # Store FIRST angle and STOP looking
                first_angle_frame = frame.copy()
                first_angle_value = absolute_angle
                first_frame_number = current_frame
                angle_detected = True
                
                print(f"  Found first angle at frame {current_frame}: {absolute_angle:.1f}°")
                break  # Stop after finding first angle
        
        current_frame += 1
    
    cap.release()
    
    return {
        'detected': angle_detected,
        'angle': first_angle_value,
        'frame_number': first_frame_number,
        'frame_image': first_angle_frame,
        'validated': False,
        'validation_method': 'first_angle_fallback'
    }

def extract_insertion_angles_integrated(video_path, output_folder, segments, model, video_duration, save_debug_video=False):
    """
    Integrated version for use within analyzer.py with proper validation
    
    Args:
        video_path: Path to video
        output_folder: Where to save images
        segments: List of (start_time, end_time) tuples
        model: Already loaded YOLO model
        video_duration: Total video duration
        save_debug_video: If True, saves annotated debug video (default: False)
    
    Returns:
        List of result dictionaries
    """
    results_summary = []

    print("\n" + "="*60)
    print("NEEDLE INSERTION ANGLE DETECTION (VALIDATED)")
    print("="*60)
    print(f"Using validation logic from main_v5.py:")
    print(f"  - Detects incision boxes")
    print(f"  - Tracks needle tip entry")
    print(f"  - Captures angle at entry moment")
    print(f"  - Validates with horizontal check ({HORIZONTAL_THRESHOLD}°)")
    print(f"  - Requires {MIN_HORIZONTAL_DURATION_SEC}s horizontal duration")
    if save_debug_video:
        print(f"  - 📹 DEBUG MODE: Saving annotated videos")
    print("="*60)

    # ── Global incision scan (first ~10 s of video, done once) ────────────
    # Wound location is fixed throughout the procedure.  For sutures 2-N the
    # wound may already have stitches and the "incisions" class may not fire
    # within the segment's own start window, so we share the boxes found early
    # in the video with every segment as a fallback.
    cap_tmp = cv2.VideoCapture(video_path)
    fps_tmp  = cap_tmp.get(cv2.CAP_PROP_FPS)
    cap_tmp.release()
    global_scan_end_frame = int(min(30.0, video_duration) * fps_tmp)
    print(f"\nGlobal incision scan: first {global_scan_end_frame} frames "
          f"({min(30.0, video_duration):.0f}s)...")
    global_incision_boxes = detect_incisions_in_segment(
        video_path, model, 0.0, min(30.0, video_duration), fps_tmp
    )
    print(f"  → {len(global_incision_boxes)} global incision box(es) found")

    for idx, (start_time, end_time) in enumerate(segments, 1):
        segment_num = idx

        print(f"\nProcessing Segment {segment_num}: {start_time:.2f}s - {end_time:.2f}s")

        # Extend segment to 15 seconds if needed
        extended_start, extended_end = extend_segment_to_min_duration(
            start_time, end_time, min_duration=15.0, video_duration=video_duration
        )

        if extended_end != end_time:
            print(f"  Extended to: {extended_start:.2f}s - {extended_end:.2f}s")

        # Process segment with validation (and optional debug video)
        result = process_segment_for_insertion_angle(
            video_path, model, segment_num, extended_start, extended_end,
            output_folder, save_debug_video,
            fallback_incision_boxes=global_incision_boxes
        )
        
        if result['detected'] and result['frame_image'] is not None:
            # Resize image for web display
            frame_img = result['frame_image']
            height, width = frame_img.shape[:2]
            max_width = 1280
            
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                frame_img_resized = cv2.resize(frame_img, (new_width, new_height), 
                                               interpolation=cv2.INTER_AREA)
            else:
                frame_img_resized = frame_img
            
            # Save the frame
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Add validation status to filename
            if result.get('validated', False):
                output_filename = f"{base_name}_Segment_{segment_num}_insertion_VALIDATED.jpg"
            else:
                output_filename = f"{base_name}_Segment_{segment_num}_insertion_UNVALIDATED.jpg"
            
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, frame_img_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            results_summary.append({
                'segment_num': segment_num,
                'detected': True,
                'angle': result['angle'],
                'frame_number': result['frame_number'],
                'output_file': output_filename,
                'validated': result.get('validated', False),
                'validation_method': result.get('validation_method', 'unknown')
            })
            
            print(f"  ✅ Angle detected: {result['angle']:.1f}°")
            print(f"     Validation: {'PASSED' if result.get('validated') else 'FAILED (using fallback)'}")
            print(f"     Saved: {output_filename}")
        else:
            # No angle detected
            results_summary.append({
                'segment_num': segment_num,
                'detected': False,
                'angle': None,
                'frame_number': None,
                'output_file': None,
                'validated': False,
                'validation_method': 'none'
            })
            print(f"  ❌ No angle detected in this segment")
    
    print("\n" + "="*60)
    print("INSERTION ANGLE DETECTION COMPLETE")
    print("="*60)
    
    validated_count = sum(1 for r in results_summary if r.get('validated', False))
    detected_count = sum(1 for r in results_summary if r['detected'])
    
    print(f"Total segments: {len(results_summary)}")
    print(f"Angles detected: {detected_count}")
    print(f"Validated insertions: {validated_count}")
    print(f"Fallback detections: {detected_count - validated_count}")
    print("="*60 + "\n")
    
    return results_summary

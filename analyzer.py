from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load model once globally
MODEL_PATH = "best_v3.pt"
model = None

def load_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

def analyze_video_with_progress(video_path, output_folder, progress_callback=None, frames_per_second=1, max_gap_duration=5.0, birdseye_video_path=None):
    """
    Analyze video for needle holding segments with progress updates
    
    Args:
        video_path: Path to the input video file
        output_folder: Folder where results will be saved
        progress_callback: Function to call with (progress_percent, message)
        frames_per_second: Number of frames to process per second (default: 3)
        max_gap_duration: Maximum gap in seconds to ignore (default: 5.0)
        birdseye_video_path: Optional path to bird's eye view video for hand rotation analysis
    
    Returns:
        Path to the generated text file
    """
    
    def update_progress(percent, message):
        if progress_callback:
            progress_callback(percent, message)
    
    # Load model
    update_progress(15, "Loading AI model...")
    model = load_model()
    
    update_progress(20, "Opening video file...")
    print(f"Processing video: {video_path}")
    print("="*60)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"  Processing: {frames_per_second} frames per second")
    print("="*60)
    print("Detecting holding states...")
    print("="*60)
    
    update_progress(25, f"Analyzing video ({total_frames/fps:.1f}s duration)...")
    
    # Calculate frame skip interval
    frame_interval = max(1, fps // frames_per_second)
    
    # Variables for tracking
    holding_states = []
    processed_count = 0
    max_needle_length = 0  # Track maximum needle length
    
    # Process only specific frames
    frames_to_process = list(range(0, total_frames, frame_interval))
    total_to_process = len(frames_to_process)
    
    for frame_idx, frame_num in enumerate(frames_to_process):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_count += 1
        current_time = frame_num / fps
        
        progress_percent = 25 + int((frame_idx / total_to_process) * 50)
        if processed_count % 5 == 0 or processed_count == 1:
            update_progress(
                progress_percent, 
                f"Processing frames: {processed_count}/{total_to_process} ({current_time:.1f}s)"
            )
        
        results = model(frame, verbose=False)
        
        needle_mask = np.zeros((height, width), dtype=np.uint8)
        needle_holder_mask = np.zeros((height, width), dtype=np.uint8)
        needle_detected = False
        needle_holder_detected = False
        
        for result in results:
            if result.masks is not None and result.boxes is not None:
                masks_xy = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()
                class_names = result.names
                
                for mask_idx, (mask_coords, cls_id) in enumerate(zip(masks_xy, classes)):
                    class_name = class_names[int(cls_id)]
                    
                    if len(mask_coords) > 0:
                        points = mask_coords.astype(np.int32)
                        
                        # Fill mask for this object
                        if class_name.lower() == 'needle':
                            cv2.fillPoly(needle_mask, [points], 255)
                            needle_detected = True
                            
                            # Calculate max needle length
                            if len(points) > 1:
                                scores = points[:, 0] + points[:, 1] * 0.5
                                start_point_idx = np.argmin(scores)
                                start_point = points[start_point_idx]
                                
                                distances_from_start = np.sqrt(
                                    (points[:, 0] - start_point[0])**2 + 
                                    (points[:, 1] - start_point[1])**2
                                )
                                
                                current_max_length = np.max(distances_from_start)
                                if current_max_length > max_needle_length:
                                    max_needle_length = current_max_length
                                    
                        elif class_name.lower() == 'needle_holder':
                            cv2.fillPoly(needle_holder_mask, [points], 255)
                            needle_holder_detected = True
        
        # Check for overlap (holding position)
        is_holding = False
        
        if needle_detected and needle_holder_detected:
            # Find the intersection (overlap) of both masks
            overlap_mask = cv2.bitwise_and(needle_mask, needle_holder_mask)
            overlap_count = cv2.countNonZero(overlap_mask)
            
            # If there's any overlap, needle is being held
            if overlap_count > 0:
                is_holding = True
        
        # Store holding state
        holding_states.append((frame_num, is_holding, current_time))
        
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count}/{len(frames_to_process)} frames ({current_time:.1f}s)...")
    
    cap.release()
    
    print(f"Detection complete! Processed {processed_count} frames.")
    print("="*60)
    print("Analyzing holding segments...")
    print(f"(Ignoring gaps shorter than {max_gap_duration} seconds)")
    print("="*60)
    
    update_progress(80, "Analyzing holding segments...")
    
    # Expand holding states to all frames for accurate segment detection
    all_frame_states = []
    last_state = False
    state_idx = 0
    
    for frame_num in range(total_frames):
        timestamp = frame_num / fps
        
        # Check if we have a processed state for this frame
        if state_idx < len(holding_states) and holding_states[state_idx][0] == frame_num:
            last_state = holding_states[state_idx][1]
            state_idx += 1
        
        all_frame_states.append((frame_num, last_state, timestamp))
    
    segments = []
    current_segment_start = None
    last_holding_time = None
    
    for frame_num, is_holding, timestamp in all_frame_states:
        if is_holding:
            if current_segment_start is None:
                # Start new segment
                current_segment_start = timestamp
            # Update last holding time
            last_holding_time = timestamp
        else:
            # Not holding
            if current_segment_start is not None and last_holding_time is not None:
                # Check gap duration
                gap_duration = timestamp - last_holding_time
                
                if gap_duration > max_gap_duration:
                    # Gap too long, end current segment
                    duration = last_holding_time - current_segment_start
                    if duration >= 2.0:  # Only keep segments >= 2 seconds
                        segments.append((current_segment_start, last_holding_time))
                    current_segment_start = None
                    last_holding_time = None
    
    # Close final segment if still open
    if current_segment_start is not None and last_holding_time is not None:
        duration = last_holding_time - current_segment_start
        if duration >= 2.0:  # Only keep segments >= 2 seconds
            segments.append((current_segment_start, last_holding_time))
    
    update_progress(82, "Calculating suture time windows...")

    # Display segments
    print(f"\nFound {len(segments)} holding segment(s):")
    print("-" * 60)
    
    total_holding_time = 0
    for i, (start, end) in enumerate(segments, 1):
        duration = end - start
        total_holding_time += duration
        print(f"Segment {i}: {start:.2f}s - {end:.2f}s (Duration: {duration:.2f}s)")
    
    print("-" * 60)
    video_duration = total_frames / fps
    print(f"Total holding time: {total_holding_time:.2f}s")
    print(f"Video duration: {video_duration:.2f}s")
    print(f"Holding percentage: {(total_holding_time / video_duration) * 100:.1f}%")
    
    # Calculate suture times based on segments
    suture_times = []
    
    if len(segments) == 0:
        # No segments detected - no sutures
        pass
    elif len(segments) == 1:
        # Only one segment - entire video is suture 1
        suture_times.append((0, video_duration, video_duration))
    else:
        # Multiple segments - calculate suture times
        for i in range(len(segments)):
            if i == 0:
                # First suture: from start (or segment_start - 2, whichever is greater than 0) to next segment start - 2
                suture_start = max(0, segments[i][0] - 2)
                suture_end = segments[i + 1][0] - 2
            elif i == len(segments) - 1:
                # Last suture: from current segment start - 2 to end of video
                suture_start = segments[i][0] - 2
                suture_end = video_duration
            else:
                # Middle sutures: from current segment start - 2 to next segment start - 2
                suture_start = segments[i][0] - 2
                suture_end = segments[i + 1][0] - 2
            
            suture_duration = suture_end - suture_start
            suture_times.append((suture_start, suture_end, suture_duration))
    
    update_progress(84, "Extracting 33% holding frames...")
    
    # Extract 33% holding frames for each suture
    suture_holding_data = []
    if len(suture_times) > 0:
        from frame_extractor import extract_33_percent_frames_integrated
        suture_holding_data = extract_33_percent_frames_integrated(
            video_path, output_folder, suture_times, model, max_needle_length
        )
    
    update_progress(87, "Detecting needle insertion angles...")
    
    # Extract needle insertion angles for each segment
    insertion_angle_data = []
    if len(segments) > 0:
        from insertion_angle_detector import extract_insertion_angles_integrated
        insertion_angle_data = extract_insertion_angles_integrated(
            video_path, output_folder, segments, model, video_duration
        )
    
    update_progress(90, "Analyzing hand rotation (if bird's eye video provided)...")
    
    # Extract hand rotation data if bird's eye video is provided
    hand_rotation_data = []
    
    # Use provided bird's eye video path
    if birdseye_video_path and os.path.exists(birdseye_video_path) and len(segments) > 0:
        from hand_rotation_analyzer import analyze_hand_rotation_integrated
        hand_rotation_data = analyze_hand_rotation_integrated(
            birdseye_video_path, output_folder, segments, video_duration
        )
    
    update_progress(93, "Analyzing knots (if bird's eye video provided)...")
    
    # Detect knots if bird's eye video is provided
    knot_data = {}
    if birdseye_video_path and os.path.exists(birdseye_video_path):
        from knot_detector import analyze_knots_integrated
        knot_data = analyze_knots_integrated(birdseye_video_path, output_folder)
    
    update_progress(95, "Analyzing expert hand movements (if bird's eye video provided)...")

    # Analyze expert hand movements if bird's eye video is provided
    expert_movement_data = []
    if birdseye_video_path and os.path.exists(birdseye_video_path) and len(segments) > 0 and len(suture_times) > 0:
        try:
            from expert_analyzer import analyze_expert_movements_integrated
            cap_be = cv2.VideoCapture(birdseye_video_path)
            birdseye_duration = cap_be.get(cv2.CAP_PROP_FRAME_COUNT) / cap_be.get(cv2.CAP_PROP_FPS)
            cap_be.release()

            expert_movement_data = analyze_expert_movements_integrated(
                birdseye_video_path, output_folder, segments, suture_times, birdseye_duration
            )
        except ImportError:
            print("⚠️  expert_analyzer module not found - skipping expert movement analysis")
        except Exception as e:
            print(f"⚠️  Error in expert movement analysis: {e}")

    update_progress(97, "Detecting needle entry/exit points...")

    # Detect needle entry (red dot) and exit (blue dot) for each suture.
    # Window = holding segment start → segment end + 8 s (capped at video duration).
    # This matches the insertion angle detector's start frame (same incision scan,
    # same entry angle) while giving 8 extra seconds to catch the needle exit.
    ENTRY_EXIT_EXTRA_SEC = 8.0
    entry_exit_data = []
    if len(segments) > 0:
        try:
            from needle_entry_exit_analyzer import analyze_entry_exit_integrated
            entry_exit_windows = [
                (seg_start,
                 min(seg_end + ENTRY_EXIT_EXTRA_SEC, video_duration),
                 min(seg_end + ENTRY_EXIT_EXTRA_SEC, video_duration) - seg_start)
                for seg_start, seg_end in segments
            ]
            entry_exit_data = analyze_entry_exit_integrated(
                video_path, output_folder, entry_exit_windows, model
            )
        except ImportError:
            print("⚠️  needle_entry_exit_analyzer module not found - skipping entry/exit analysis")
        except Exception as e:
            print(f"⚠️  Error in entry/exit analysis: {e}")

    # Save segments to text file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    segments_file = os.path.join(output_folder, f"{base_name}_holding_segments.txt")
    
    with open(segments_file, 'w') as f:
        f.write("NEEDLE HOLDING TIME SEGMENTS\n")
        f.write("="*60 + "\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Video duration: {video_duration:.2f}s\n")
        f.write(f"Frames processed: {processed_count} ({frames_per_second} per second)\n")
        f.write(f"Maximum gap ignored: {max_gap_duration}s\n")
        f.write(f"Detection method: Needle and Needle_holder overlap\n")
        f.write("="*60 + "\n\n")
        
        f.write("HOLDING SEGMENTS:\n")
        f.write("-"*60 + "\n")
        
        if len(segments) == 0:
            f.write("No holding segments detected.\n")
        else:
            for i, (start, end) in enumerate(segments, 1):
                duration = end - start
                f.write(f"Segment {i}: {start:.2f}s - {end:.2f}s (Duration: {duration:.2f}s)\n")
                
                # Add insertion angle info if available
                if i <= len(insertion_angle_data):
                    angle_info = insertion_angle_data[i-1]
                    if angle_info['detected']:
                        f.write(f"  Insertion Angle: {angle_info['angle']:.1f}°\n")
                        f.write(f"  Angle Frame: Frame {angle_info['frame_number']}\n")
                        f.write(f"  Image: {angle_info['output_file']}\n")
                    else:
                        f.write(f"  Insertion Angle: Not Detected\n")
                
                # Add hand rotation info if available
                if i <= len(hand_rotation_data):
                    rotation_info = hand_rotation_data[i-1]
                    if rotation_info['success']:
                        stats = rotation_info['stats']
                        f.write(f"  Hand Rotation:\n")
                        f.write(f"    Pronated: {stats['pronated_pct']:.1f}%\n")
                        f.write(f"    Supinated: {stats['supinated_pct']:.1f}%\n")
                        if stats['angle_stats']['mean'] is not None:
                            f.write(f"    Angle Range: {stats['angle_stats']['min']:.1f}° - {stats['angle_stats']['max']:.1f}°\n")
                        f.write(f"    Graph: {rotation_info['png_file']}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write(f"Total holding time: {total_holding_time:.2f}s\n")
        f.write(f"Holding percentage: {(total_holding_time / video_duration) * 100:.1f}%\n")
        
        # Add suture time analysis
        f.write("\n\n")
        f.write("SUTURE TIME ANALYSIS:\n")
        f.write("-"*60 + "\n")
        
        if len(suture_times) == 0:
            f.write("No sutures detected (no holding segments found).\n")
        else:
            for i, (start, end, duration) in enumerate(suture_times, 1):
                f.write(f"Suture {i}: {start:.2f}s - {end:.2f}s (Duration: {duration:.2f}s)\n")
                
                # Add 33% holding percentage if available
                if i <= len(suture_holding_data):
                    holding_info = suture_holding_data[i-1]
                    f.write(f"  33% Holding Frame: Frame {holding_info['frame_number']}\n")
                    f.write(f"  Actual Holding: {holding_info['holding_percentage']:.1f}%\n")
                    f.write(f"  Image: {holding_info['output_file']}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write(f"Total sutures: {len(suture_times)}\n")
            
            if len(suture_times) > 0:
                avg_suture_time = sum(duration for _, _, duration in suture_times) / len(suture_times)
                f.write(f"Average suture time: {avg_suture_time:.2f}s\n")
            
            if len(suture_holding_data) > 0:
                avg_holding = sum(s['holding_percentage'] for s in suture_holding_data) / len(suture_holding_data)
                f.write(f"Average 33% target holding: {avg_holding:.1f}%\n")
        
        # Add knot analysis section
        if knot_data and knot_data.get('success'):
            f.write("\n\n")
            f.write("KNOT ANALYSIS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Analysis from: Last 5 seconds of bird's eye video\n")
            f.write(f"Number of knots detected: {knot_data['num_knots']}\n")
            f.write(f"Image: {knot_data['output_file']}\n")
            f.write("\n")
            f.write("Knot Positions:\n")
            for i, (cx, cy) in enumerate(knot_data['knot_centers']):
                f.write(f"  K{i+1}: ({cx}, {cy})\n")
            f.write("\n")
            f.write("Knot Distances:\n")
            for i, dist in enumerate(knot_data['distances']):
                f.write(f"  K{i+1} -> K{i+2}: {dist:.1f}px\n")
    
        # Add expert hand movement analysis section
        if expert_movement_data and len(expert_movement_data) > 0:
            f.write("\n\n")
            f.write("EXPERT HAND MOVEMENT ANALYSIS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Analysis window: (segment_end + 5s) to (suture_end - 5s)\n")
            f.write(f"Total sutures analyzed: {len(expert_movement_data)}\n")
            f.write("\n")

            for movement_info in expert_movement_data:
                if movement_info['success']:
                    f.write(f"Suture {movement_info['suture_num']}:\n")
                    f.write(f"  Left hand loops: {movement_info['left_loops']}\n")
                    f.write(f"  Right hand loops: {movement_info['right_loops']}\n")
                    f.write(f"  Total loops: {movement_info['total_loops']}\n")
                    f.write(f"  Graph: {movement_info['output_file']}\n")
                    if movement_info.get('output_video'):
                        f.write(f"  Video: {movement_info['output_video']}\n")
                else:
                    f.write(f"Suture {movement_info['suture_num']}: {movement_info.get('error', 'Analysis failed')}\n")
                f.write("\n")

        # Add needle entry/exit analysis section
        if entry_exit_data and len(entry_exit_data) > 0:
            f.write("\n\n")
            f.write("NEEDLE ENTRY/EXIT ANALYSIS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Total sutures analyzed: {len(entry_exit_data)}\n")
            f.write("\n")

            for ee in entry_exit_data:
                if ee['success']:
                    f.write(f"Suture {ee['suture_num']}:\n")
                    if ee['entry_angle'] is not None:
                        f.write(f"  Entry Angle: {ee['entry_angle']:.1f}°\n")
                    else:
                        f.write(f"  Entry Angle: Not Detected\n")
                    if ee['entry_tip']:
                        f.write(f"  Entry Point: {ee['entry_tip']}\n")
                    if ee['exit_point']:
                        f.write(f"  Exit Point: {ee['exit_point']}\n")
                    if ee['output_file']:          # only write when image was saved
                        f.write(f"  Image: {ee['output_file']}\n")
                else:
                    f.write(f"Suture {ee['suture_num']}: {ee.get('error', 'Analysis failed')}\n")
                f.write("\n")
    
    update_progress(99, "Saving results...")
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Results saved to: {segments_file}")
    print("="*60)
    
    update_progress(100, "Complete!")
    
    return segments_file

# Backward compatibility - keep original function
def analyze_video(video_path, output_folder, frames_per_second=3, max_gap_duration=5.0):
    """Original function without progress callback for backward compatibility"""
    return analyze_video_with_progress(video_path, output_folder, None, frames_per_second, max_gap_duration)

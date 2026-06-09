import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: mediapipe not installed. Hand rotation analysis will be skipped.")

def analyze_hand_rotation_for_segment(birdseye_video_path, segment_num, start_time, end_time, 
                                      output_folder, video_name, save_debug_video=False):
    """
    Analyze hand rotation (pronation/supination) for a specific time segment
    
    Args:
        birdseye_video_path: Path to bird's eye view video
        segment_num: Segment number
        start_time: Start time in seconds
        end_time: End time in seconds  
        output_folder: Where to save results
        video_name: Base name for output files
        save_debug_video: If True, save annotated video
    
    Returns:
        dict with analysis results
    """
    if not MEDIAPIPE_AVAILABLE:
        return {
            'segment_num': segment_num,
            'success': False,
            'error': 'mediapipe not installed'
        }
    
    print(f"\n  Analyzing hand rotation for Segment {segment_num}: {start_time:.2f}s - {end_time:.2f}s")
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(birdseye_video_path)
    if not cap.isOpened():
        return {
            'segment_num': segment_num,
            'success': False,
            'error': 'Cannot open bird\'s eye video'
        }
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Set up video writer if debug mode
    writer = None
    if save_debug_video:
        debug_video_path = os.path.join(output_folder, f"{video_name}_Segment_{segment_num}_rotation_DEBUG.mp4")
        writer = cv2.VideoWriter(
            debug_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (fw, fh)
        )
        print(f"  📹 DEBUG VIDEO: Saving to {debug_video_path}")
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.7
    )
    
    # Define region of interest (right half of frame)
    x_min = fw // 2 - int(fw * 0.15)
    
    # Jump to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    results = []
    init_norm = None
    current_frame = start_frame
    frame_count = 0
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        # Find hand in region of interest
        selected_hand = None
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # Check if hand is in the right region
                wx = int(hand_landmarks.landmark[0].x * fw)
                if x_min <= wx <= fw:
                    selected_hand = hand_landmarks
                    break
        
        angle = None
        state = "No hand"
        
        if selected_hand:
            lms = selected_hand.landmark
            
            # Get key points: wrist (0), index base (5), pinky base (17)
            P0 = np.array([lms[0].x, lms[0].y, lms[0].z])
            P5 = np.array([lms[5].x, lms[5].y, lms[5].z])
            P17 = np.array([lms[17].x, lms[17].y, lms[17].z])
            
            # Calculate hand plane normal vector
            v1 = P5 - P0
            v2 = P17 - P0
            norm = np.cross(v1, v2)
            norm_len = np.linalg.norm(norm)
            
            if norm_len > 0:
                norm /= norm_len
                
                # Initialize reference normal from first detection
                if init_norm is None:
                    init_norm = norm
                
                # Calculate rotation angle from initial position
                dot_product = np.dot(norm, init_norm)
                dot_product = np.clip(dot_product, -1, 1)
                angle = np.degrees(np.arccos(dot_product))
                
                # Determine pronation/supination state
                state = "Supinated" if norm[2] < 0 else "Pronated"
                
                # Draw hand landmarks on frame if debug video enabled
                if writer:
                    mp_drawing.draw_landmarks(frame, selected_hand, mp_hands.HAND_CONNECTIONS)
                    
                    # Add state text
                    cv2.putText(frame, f"{state} - {angle:.1f}°", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            if writer:
                cv2.putText(frame, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Store result
        results.append({
            'frame': current_frame,
            'relative_frame': frame_count,
            'time': current_frame / fps,
            'angle': angle,
            'state': state
        })
        
        # Write debug video frame
        if writer:
            cv2.putText(frame, f"Frame: {current_frame}", (10, fh - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
        
        current_frame += 1
        frame_count += 1
    
    cap.release()
    if writer:
        writer.release()
        print(f"  ✅ DEBUG VIDEO saved")
    hands.close()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Generate PNG graph
    png_path = os.path.join(output_folder, f"{video_name}_Segment_{segment_num}_rotation.png")
    create_rotation_graph(df, png_path, segment_num, start_time, end_time)
    
    # Calculate statistics
    stats = calculate_rotation_stats(df)
    
    return {
        'segment_num': segment_num,
        'success': True,
        'png_file': os.path.basename(png_path),
        'stats': stats,
        'data': df
    }

def create_rotation_graph(df, output_path, segment_num, start_time, end_time):
    """Create rotation angle graph with pronation/supination highlighting"""
    
    valid = df['angle'].notna()
    
    plt.figure(figsize=(14, 7))
    plt.title(f"Hand Rotation Analysis - Segment {segment_num}\n({start_time:.2f}s - {end_time:.2f}s)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Rotation Angle (°)", fontsize=12)
    
    # Background shading by state
    state_colors = {
        "Pronated": "#ffe6e6",
        "Supinated": "#e6f2ff",
        "No hand": "#f2f2f2"
    }
    
    prev_state = None
    start_idx = 0
    
    for idx, state in enumerate(df['state']):
        if state != prev_state and prev_state is not None:
            plt.axvspan(df['time'].iloc[start_idx], df['time'].iloc[idx-1],
                       color=state_colors.get(prev_state, "#ffffff"), alpha=0.4)
            start_idx = idx
        prev_state = state
    
    if len(df) > 0:
        plt.axvspan(df['time'].iloc[start_idx], df['time'].iloc[-1],
                   color=state_colors.get(prev_state, "#ffffff"), alpha=0.4)
    
    # Plot angle line
    if valid.any():
        plt.plot(df['time'][valid], df['angle'][valid], 
                marker='o', linewidth=2, color='black', markersize=4, label='Rotation Angle')
    
    # Mark state transitions
    for i in range(1, len(df)):
        if df['state'].iloc[i] != df['state'].iloc[i-1]:
            plt.axvline(df['time'].iloc[i], color='gray', linestyle='--', alpha=0.7, linewidth=1)
            plt.text(df['time'].iloc[i], plt.ylim()[1] * 0.95, df['state'].iloc[i],
                    rotation=90, verticalalignment='top', fontsize=9, color='gray')
    
    # Legend
    legend_patches = [
        Patch(facecolor=col, alpha=0.4, label=state) 
        for state, col in state_colors.items()
    ]
    plt.legend(handles=legend_patches + [plt.Line2D([0], [0], color='black', lw=2, label='Angle')],
              loc='best')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Graph saved: {os.path.basename(output_path)}")

def calculate_rotation_stats(df):
    """Calculate statistics from rotation data"""
    
    # Count frames in each state
    state_counts = df['state'].value_counts()
    total_frames = len(df)
    
    # Calculate percentages
    pronated_pct = (state_counts.get('Pronated', 0) / total_frames * 100) if total_frames > 0 else 0
    supinated_pct = (state_counts.get('Supinated', 0) / total_frames * 100) if total_frames > 0 else 0
    no_hand_pct = (state_counts.get('No hand', 0) / total_frames * 100) if total_frames > 0 else 0
    
    # Angle statistics (only for valid detections)
    valid_angles = df[df['angle'].notna()]['angle']
    
    if len(valid_angles) > 0:
        angle_stats = {
            'min': valid_angles.min(),
            'max': valid_angles.max(),
            'mean': valid_angles.mean(),
            'median': valid_angles.median()
        }
    else:
        angle_stats = {
            'min': None,
            'max': None,
            'mean': None,
            'median': None
        }
    
    return {
        'pronated_pct': pronated_pct,
        'supinated_pct': supinated_pct,
        'no_hand_pct': no_hand_pct,
        'angle_stats': angle_stats,
        'total_frames': total_frames,
        'valid_detections': len(valid_angles)
    }

def analyze_hand_rotation_integrated(birdseye_video_path, output_folder, segments, 
                                     video_duration, save_debug_video=False):
    """
    Integrated version for use within analyzer.py
    Processes hand rotation for each segment (insertion_time - 1s to +10s)
    
    Args:
        birdseye_video_path: Path to bird's eye video
        output_folder: Where to save results
        segments: List of (start_time, end_time) tuples from holding segments
        video_duration: Total video duration
        save_debug_video: If True, saves debug videos
    
    Returns:
        List of result dictionaries
    """
    if not MEDIAPIPE_AVAILABLE:
        print("⚠️  MediaPipe not installed - skipping hand rotation analysis")
        return []
    
    print("\n" + "="*60)
    print("HAND ROTATION ANALYSIS (PRONATION/SUPINATION)")
    print("="*60)
    print(f"Processing bird's eye video: {os.path.basename(birdseye_video_path)}")
    print(f"Analyzing {len(segments)} segment(s)")
    if save_debug_video:
        print(f"📹 DEBUG MODE: Saving annotated videos")
    print("="*60)
    
    results_summary = []
    video_name = os.path.splitext(os.path.basename(birdseye_video_path))[0]
    
    for idx, (start_time, end_time) in enumerate(segments, 1):
        # Process from (start_time - 1s) to (start_time + 10s)
        analysis_start = max(0, start_time - 1.0)
        analysis_end = min(video_duration, start_time + 10.0)
        
        print(f"\nSegment {idx}: Original {start_time:.2f}s - {end_time:.2f}s")
        print(f"           Analyzing {analysis_start:.2f}s - {analysis_end:.2f}s")
        
        result = analyze_hand_rotation_for_segment(
            birdseye_video_path,
            idx,
            analysis_start,
            analysis_end,
            output_folder,
            video_name,
            save_debug_video
        )
        
        results_summary.append(result)
        
        if result['success']:
            stats = result['stats']
            print(f"  ✅ Success!")
            print(f"     Pronated: {stats['pronated_pct']:.1f}%")
            print(f"     Supinated: {stats['supinated_pct']:.1f}%")
            print(f"     No hand: {stats['no_hand_pct']:.1f}%")
            if stats['angle_stats']['mean'] is not None:
                print(f"     Angle range: {stats['angle_stats']['min']:.1f}° - {stats['angle_stats']['max']:.1f}°")
        else:
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("HAND ROTATION ANALYSIS COMPLETE")
    print("="*60)
    successful = sum(1 for r in results_summary if r['success'])
    print(f"Successfully analyzed: {successful}/{len(results_summary)} segments")
    print("="*60 + "\n")
    
    return results_summary

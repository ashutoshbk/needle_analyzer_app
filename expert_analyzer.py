"""
Expert Hand Movement Analyzer
Analyzes hand movements and loop detection for specific time windows
Integrated version for suture analysis
- Updated to match Expert_process3.py video output exactly:
  * Step detection (6 steps: loop groups + crossovers)
  * Steps Detected: X/6 overlay
  * Step checklist in top-right corner
  * Loop diameter calculations
  * Dynamic combined average diameter display
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import shutil

mp_hands = mp.solutions.hands


def remux_for_web(input_path):
    """Re-encode to H.264 + faststart so all browsers can play it inline."""
    tmp_path = input_path + "_h264.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-vcodec", "libx264",
             "-pix_fmt", "yuv420p",
             "-preset", "fast",
             "-crf", "23",
             "-acodec", "aac",
             "-movflags", "+faststart",
             tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0 and os.path.exists(tmp_path):
            import shutil
            shutil.move(tmp_path, input_path)
            print(f"    ✅ Re-encoded to H.264: {os.path.basename(input_path)}")
        else:
            print(f"    ⚠️  ffmpeg encode failed, keeping original")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"    ⚠️  Could not encode video: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def detect_loops(fx, fy, fr, hand_name):
    fx = np.array(fx)
    fy = np.array(fy)
    fr = np.array(fr)
    valid_mask = ~np.isnan(fx) & ~np.isnan(fy)

    if valid_mask.sum() < 5:
        print(f"    {hand_name}: Insufficient data")
        return [], np.zeros_like(fr, dtype=int)

    cx, cy = np.nanmean(fx), np.nanmean(fy)
    xv = fx[valid_mask] - cx
    yv = fy[valid_mask] - cy
    fv = fr[valid_mask]
    angles = np.unwrap(np.arctan2(yv, xv))

    min_loop_frames = 10
    min_angle_change = 2
    loop_segments = []
    completed_loops_count = np.zeros_like(fv, dtype=int)

    start_idx = 0
    completed_count = 0

    for i in range(1, len(angles)):
        angle_change = abs(angles[i] - angles[start_idx])
        frame_span = fv[i] - fv[start_idx]
        crossed = int(angles[i] // (2 * np.pi)) != int(angles[start_idx] // (2 * np.pi))

        if crossed and (frame_span >= min_loop_frames) and (angle_change >= min_angle_change):
            x_seg = xv[start_idx:i] - np.nanmean(xv[start_idx:i])
            y_seg = yv[start_idx:i] - np.nanmean(yv[start_idx:i])
            ds = np.sum(x_seg[1:] * y_seg[:-1] - y_seg[1:] * x_seg[:-1])
            dir_lbl = 'CCW' if ds > 0 else 'CW'
            completed_count += 1
            loop_segments.append((fv[start_idx], fv[i], dir_lbl, completed_count))
            completed_loops_count[start_idx:i] = completed_count
            start_idx = i

    full_completed = np.zeros(len(fr), dtype=int)
    full_completed[valid_mask] = completed_loops_count

    return loop_segments, full_completed


def calculate_loop_diameter(fx, fy, start_frame, end_frame):
    loop_x, loop_y = [], []
    for i in range(int(start_frame), int(end_frame) + 1):
        if i < len(fx) and not np.isnan(fx[i]) and not np.isnan(fy[i]):
            loop_x.append(fx[i])
            loop_y.append(fy[i])
    if len(loop_x) < 2:
        return 0
    max_distance = 0
    for i in range(len(loop_x)):
        for j in range(i + 1, len(loop_x)):
            dist = np.sqrt((loop_x[i] - loop_x[j]) ** 2 + (loop_y[i] - loop_y[j]) ** 2)
            if dist > max_distance:
                max_distance = dist
    return max_distance


def detect_independent_steps(lx, ly, rx, ry, right_loop_nums, frames, fps, right_loops):
    loop_group_time_gap = 2.0
    loop_group_frame_gap = int(loop_group_time_gap * fps)

    step_completed = [False] * 7
    step_frame_completed = [-1] * 7

    loop_occurrence_frames = []
    prev_loop_count = 0
    for i in range(len(frames)):
        if right_loop_nums[i] > prev_loop_count:
            loop_occurrence_frames.append((i, right_loop_nums[i]))
            prev_loop_count = right_loop_nums[i]

    loop_groups = []
    if loop_occurrence_frames:
        current_group_start = loop_occurrence_frames[0][0]
        current_group_end = loop_occurrence_frames[0][0]
        loops_in_group = 1

        for i in range(1, len(loop_occurrence_frames)):
            frame_of_loop = loop_occurrence_frames[i][0]
            if frame_of_loop - current_group_end <= loop_group_frame_gap:
                current_group_end = frame_of_loop
                loops_in_group += 1
            else:
                loop_groups.append((current_group_start, current_group_end, loops_in_group))
                current_group_start = frame_of_loop
                current_group_end = frame_of_loop
                loops_in_group = 1
        loop_groups.append((current_group_start, current_group_end, loops_in_group))

    print(f"\n    Loop Groups Detected: {len(loop_groups)}")
    for idx, (start, end, count) in enumerate(loop_groups, 1):
        print(f"      Group {idx}: frames {start}-{end}, {count} loop(s)")

    if len(loop_groups) >= 1:
        step_completed[1] = True
        step_frame_completed[1] = loop_groups[0][1]
    if len(loop_groups) >= 2:
        step_completed[3] = True
        step_frame_completed[3] = loop_groups[1][1]
    if len(loop_groups) >= 3:
        step_completed[5] = True
        step_frame_completed[5] = loop_groups[2][1]

    x_distance_lr = 300
    x_distance_rl = 300
    x_start_threshold = 100

    crossover_lr_events = []
    in_position_left = False
    position_start_frame = -1

    for i in range(len(frames)):
        lx_curr = lx[i]
        rx_curr = rx[i]
        if not np.isnan(lx_curr) and not np.isnan(rx_curr):
            x_diff = lx_curr - rx_curr
            if x_diff < -x_start_threshold:
                if not in_position_left:
                    in_position_left = True
                    position_start_frame = i
            elif x_diff > x_distance_lr:
                if in_position_left:
                    crossover_lr_events.append((position_start_frame, i))
                    in_position_left = False

    crossover_rl_events = []
    in_position_right = False
    position_start_frame = -1

    for i in range(len(frames)):
        lx_curr = lx[i]
        rx_curr = rx[i]
        if not np.isnan(lx_curr) and not np.isnan(rx_curr):
            x_diff = lx_curr - rx_curr
            if x_diff > x_start_threshold:
                if not in_position_right:
                    in_position_right = True
                    position_start_frame = i
            elif x_diff < -x_distance_rl:
                if in_position_right:
                    crossover_rl_events.append((position_start_frame, i))
                    in_position_right = False

    if len(crossover_lr_events) >= 1:
        step_completed[2] = True
        step_frame_completed[2] = crossover_lr_events[0][1]
    if len(crossover_rl_events) >= 1:
        step_completed[4] = True
        step_frame_completed[4] = crossover_rl_events[0][1]
    if len(crossover_lr_events) >= 2:
        step_completed[6] = True
        step_frame_completed[6] = crossover_lr_events[1][1]

    return step_completed, step_frame_completed, loop_groups


def analyze_hand_movements_for_suture(video_path, start_time, end_time, suture_num, output_folder):
    """
    Analyze hand movements for a specific time window of a suture.
    Output video matches Expert_process3.py exactly.

    Args:
        video_path: Path to bird's eye video
        start_time: Start time in seconds
        end_time: End time in seconds
        suture_num: Suture number
        output_folder: Where to save results

    Returns:
        dict with analysis results
    """
    print(f"\n  Analyzing hand movements for Suture {suture_num}: {start_time:.2f}s - {end_time:.2f}s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'suture_num': suture_num, 'success': False, 'error': 'Cannot open video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.7
    )

    left_x, left_y = [], []
    right_x, right_y = [], []
    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 0
    current_frame = start_frame

    # First pass: collect tracking data
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_fingertip = None
        right_fingertip = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                lm = hand_landmarks.landmark[8]
                x, y = int(lm.x * fw), int(lm.y * fh)
                if hand_label == "Right":
                    left_fingertip = (x, y)
                elif hand_label == "Left":
                    right_fingertip = (x, y)

        left_x.append(left_fingertip[0] if left_fingertip else np.nan)
        left_y.append(left_fingertip[1] if left_fingertip else np.nan)
        right_x.append(right_fingertip[0] if right_fingertip else np.nan)
        right_y.append(right_fingertip[1] if right_fingertip else np.nan)

        frames.append(frame_idx)
        frame_idx += 1
        current_frame += 1

    cap.release()
    hands.close()

    fr = np.array(frames)

    # Detect loops for both hands
    right_loops, right_loop_nums = detect_loops(right_x, right_y, fr, "Right Hand")
    left_loops, left_loop_nums = detect_loops(left_x, left_y, fr, "Left Hand")

    # --- Loop nums: carry last known count (matching Expert_process3.py behaviour) ---
    def carry_loop_nums(loop_nums_arr, fx):
        result = loop_nums_arr.copy()
        last = 0
        for i in range(len(result)):
            if not np.isnan(fx[i]):
                last = result[i]
            else:
                result[i] = last
        return result

    right_loop_nums = carry_loop_nums(right_loop_nums, right_x)
    left_loop_nums  = carry_loop_nums(left_loop_nums,  left_x)

    # Calculate loop diameters
    right_loop_diameters = []
    for st, ed, lbl, loopnum in right_loops:
        diameter = calculate_loop_diameter(np.array(right_x), np.array(right_y), st, ed)
        right_loop_diameters.append((loopnum, st, ed, diameter))
        print(f"    Right Loop {loopnum} diameter: {diameter:.2f} pixels")

    left_loop_diameters = []
    for st, ed, lbl, loopnum in left_loops:
        diameter = calculate_loop_diameter(np.array(left_x), np.array(left_y), st, ed)
        left_loop_diameters.append((loopnum, st, ed, diameter))
        print(f"    Left Loop {loopnum} diameter: {diameter:.2f} pixels")

    all_loop_diameters = right_loop_diameters + left_loop_diameters
    total_loops = len(all_loop_diameters)
    avg_combined_diameter = np.mean([d[3] for d in all_loop_diameters]) if all_loop_diameters else 0

    # Step detection
    lx_arr = np.array(left_x)
    ly_arr = np.array(left_y)
    rx_arr = np.array(right_x)
    ry_arr = np.array(right_y)

    step_completed, step_frames, loop_groups = detect_independent_steps(
        lx_arr, ly_arr, rx_arr, ry_arr, right_loop_nums, fr, fps, right_loops
    )

    # Helper functions
    def is_loop_in_progress(fidx, loop_segs):
        for st, ed, lbl, loopnum in loop_segs:
            if st <= fidx <= ed:
                return True, loopnum
        return False, 0

    def is_middle_of_loop(fidx, loop_segs):
        for st, ed, lbl, loopnum in loop_segs:
            mid = int((st + ed) / 2)
            if fidx == mid:
                return True, loopnum, st, ed
        return False, 0, 0, 0

    # Second pass: write annotated video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_filename = f"{video_name}_Suture_{suture_num}_hand_movements.mp4"
    output_video_path = os.path.join(output_folder, output_video_filename)

    cap2 = cv2.VideoCapture(video_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))

    tail_length = 20
    left_tail = []
    right_tail = []
    left_loop_trails = {}
    right_loop_trails = {}
    loop_middle_distances = {}

    step_names = [
        "",
        "1. K(right) Hand Loop Group 1",
        "2. T(left) Hand Separation L->R (1st)",
        "3. K(right) Hand Loop Group 2",
        "4. T(left) Hand Separation R->L",
        "5. K(right) Hand Loop Group 3",
        "6. T(left) Hand Separation L->R (2nd)"
    ]

    current_frame = start_frame
    frame_idx2 = 0

    while current_frame <= end_frame:
        ret, frame = cap2.read()
        if not ret:
            break

        lx = left_x[frame_idx2]  if frame_idx2 < len(left_x)  else np.nan
        ly = left_y[frame_idx2]  if frame_idx2 < len(left_y)  else np.nan
        rx = right_x[frame_idx2] if frame_idx2 < len(right_x) else np.nan
        ry = right_y[frame_idx2] if frame_idx2 < len(right_y) else np.nan

        # Collect loop trails
        for st, ed, lbl, loopnum in left_loops:
            if st <= frame_idx2 <= ed and not np.isnan(lx) and not np.isnan(ly):
                left_loop_trails.setdefault(loopnum, []).append((int(lx), int(ly)))
        for st, ed, lbl, loopnum in right_loops:
            if st <= frame_idx2 <= ed and not np.isnan(rx) and not np.isnan(ry):
                right_loop_trails.setdefault(loopnum, []).append((int(rx), int(ry)))

        # Draw LEFT hand (BLUE)
        if not np.isnan(lx) and not np.isnan(ly):
            left_pt = (int(lx), int(ly))
            left_tail.append(left_pt)
            if len(left_tail) > tail_length:
                left_tail.pop(0)
            cv2.circle(frame, left_pt, 8, (255, 0, 0), -1)
            cv2.putText(frame, f"L:({int(lx)},{int(ly)})", (int(lx) + 10, int(ly) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            for i in range(1, len(left_tail)):
                cv2.line(frame, left_tail[i - 1], left_tail[i], (255, 0, 0), 2)
            if len(left_tail) > 1:
                cv2.arrowedLine(frame, left_tail[-2], left_tail[-1], (255, 255, 0), 3, tipLength=0.4)
        else:
            left_tail.clear()

        # Draw RIGHT hand (RED)
        if not np.isnan(rx) and not np.isnan(ry):
            right_pt = (int(rx), int(ry))
            right_tail.append(right_pt)
            if len(right_tail) > tail_length:
                right_tail.pop(0)
            cv2.circle(frame, right_pt, 8, (0, 0, 255), -1)
            cv2.putText(frame, f"R:({int(rx)},{int(ry)})", (int(rx) + 10, int(ry) + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for i in range(1, len(right_tail)):
                cv2.line(frame, right_tail[i - 1], right_tail[i], (0, 0, 255), 2)
            if len(right_tail) > 1:
                cv2.arrowedLine(frame, right_tail[-2], right_tail[-1], (0, 255, 255), 3, tipLength=0.4)
        else:
            right_tail.clear()

        # Frame number (top-left, cyan)
        cv2.putText(frame, f"Frame: {frame_idx2}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Loop counts
        rc = right_loop_nums[frame_idx2] if frame_idx2 < len(right_loop_nums) else 0
        lc = left_loop_nums[frame_idx2]  if frame_idx2 < len(left_loop_nums)  else 0
        cv2.putText(frame, f"Right Hand Loops: {rc}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"Left Hand Loops: {lc}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Dynamic combined average diameter
        completed_all_diameters = [d[3] for d in all_loop_diameters if d[2] <= frame_idx2]
        current_avg_combined = np.mean(completed_all_diameters) if completed_all_diameters else 0
        cv2.putText(frame, f"Avg Loop Diameter (Both Hands): {current_avg_combined:.1f}px", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        # Steps detected count
        steps_detected = sum(step_completed[1:])
        cv2.putText(frame, f"Steps Detected: {steps_detected}/6", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Step checklist — top-right corner
        y_offset = 50
        for step_num in range(1, 7):
            step_detected_now = step_completed[step_num] and step_frames[step_num] <= frame_idx2
            if step_detected_now:
                color = (0, 255, 0)
                text = f"{step_names[step_num]} OK"
            else:
                color = (100, 100, 100)
                text = f"{step_names[step_num]}"
            cv2.putText(frame, text, (fw - 500, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
            y_offset += 35

        if steps_detected == 6:
            cv2.putText(frame, "ALL STEPS DETECTED!", (fw - 500, y_offset + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hand distance + loop active overlay
        right_loop_active, right_loop_num = is_loop_in_progress(frame_idx2, right_loops)
        left_loop_active,  left_loop_num  = is_loop_in_progress(frame_idx2, left_loops)
        right_at_middle, right_mid_num, _, _ = is_middle_of_loop(frame_idx2, right_loops)
        left_at_middle,  left_mid_num,  _, _ = is_middle_of_loop(frame_idx2, left_loops)

        if not np.isnan(lx) and not np.isnan(rx) and not np.isnan(ly) and not np.isnan(ry):
            hand_distance = np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

            if right_at_middle:
                k = f"right_{right_mid_num}"
                if k not in loop_middle_distances:
                    loop_middle_distances[k] = (hand_distance, frame_idx2)
                    print(f"    Right Loop {right_mid_num} middle distance: {hand_distance:.1f}px at frame {frame_idx2}")
            if left_at_middle:
                k = f"left_{left_mid_num}"
                if k not in loop_middle_distances:
                    loop_middle_distances[k] = (hand_distance, frame_idx2)
                    print(f"    Left Loop {left_mid_num} middle distance: {hand_distance:.1f}px at frame {frame_idx2}")

            if right_loop_active or left_loop_active:
                # cv2.line(frame, (int(lx), int(ly)), (int(rx), int(ry)), (0, 255, 0), 3)

                loop_info = ""
                display_distance = None
                display_frame = None
                display_key = None

                if right_loop_active:
                    loop_info = f"Right Loop #{right_loop_num}"
                    k = f"right_{right_loop_num}"
                    if k in loop_middle_distances:
                        display_distance, display_frame = loop_middle_distances[k]
                        display_key = k
                if left_loop_active:
                    if loop_info:
                        loop_info += " & "
                    loop_info += f"Left Loop #{left_loop_num}"
                    k = f"left_{left_loop_num}"
                    if k in loop_middle_distances:
                        display_distance, display_frame = loop_middle_distances[k]
                        display_key = k

                # cv2.putText(frame, f"LOOP ACTIVE: {loop_info}", (fw // 2 - 300, 240),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                if display_distance is not None and display_frame is not None:
                    side, num = display_key.split("_")
                    # cv2.putText(frame,
                    #     f"{side.capitalize()} Loop #{num} | Frame {display_frame} | {display_distance:.1f} px",
                    #     (fw // 2 - 300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                else:
                    # cv2.putText(frame, "Hand Distance: Calculating...",
                    #     (fw // 2 - 250, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    print("")

        # Save loop trail images at loop end
        for st, ed, lbl, loopnum in right_loops:
            if frame_idx2 == ed and loopnum in right_loop_trails:
                img = np.zeros((fh, fw, 3), dtype=np.uint8)
                pts = right_loop_trails[loopnum]
                for i in range(1, len(pts)):
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 3)
                cv2.putText(img, f"Frames: {st}-{ed}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                # cv2.imwrite(os.path.join(output_folder,
                #     f"{video_name}_Suture_{suture_num}_RIGHT_LOOP_{loopnum}.png"), img)

        for st, ed, lbl, loopnum in left_loops:
            if frame_idx2 == ed and loopnum in left_loop_trails:
                img = np.zeros((fh, fw, 3), dtype=np.uint8)
                pts = left_loop_trails[loopnum]
                for i in range(1, len(pts)):
                    cv2.line(img, pts[i - 1], pts[i], (255, 0, 0), 3)
                cv2.putText(img, f"Frames: {st}-{ed}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                # cv2.imwrite(os.path.join(output_folder,
                #     f"{video_name}_Suture_{suture_num}_LEFT_LOOP_{loopnum}.png"), img)

        # Timestamp + suture label (bottom)
        t = current_frame / fps
        cv2.putText(frame, f"t={t:.2f}s  Suture {suture_num}", (10, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx2 += 1
        current_frame += 1

    cap2.release()
    writer.release()
    remux_for_web(output_video_path)

    # Generate visualization PNG
    output_filename = f"{video_name}_Suture_{suture_num}_hand_movements.png"
    output_path = os.path.join(output_folder, output_filename)

    create_movement_graph(
        left_x, left_y, right_x, right_y,
        frames, left_loops, right_loops,
        left_loop_nums, right_loop_nums,
        output_path, suture_num, start_time, end_time, fps,
        loop_groups, step_completed, step_frames
    )

    print(f"    Left hand loops:  {len(left_loops)}")
    print(f"    Right hand loops: {len(right_loops)}")
    print(f"    Total steps detected: {sum(step_completed[1:])}/6")
    print(f"    ✅ Graph saved: {output_filename}")
    print(f"    ✅ Video saved: {output_video_filename}")

    return {
        'suture_num': suture_num,
        'success': True,
        'output_file': output_filename,
        'output_video': output_video_filename,
        'left_loops': len(left_loops),
        'right_loops': len(right_loops),
        'total_loops': len(left_loops) + len(right_loops),
        'left_loop_details': left_loops,
        'right_loop_details': right_loops,
        'steps_detected': sum(step_completed[1:]),
        'loop_groups': len(loop_groups),
        'avg_combined_diameter': avg_combined_diameter,
    }


def create_movement_graph(left_x, left_y, right_x, right_y, frames,
                          left_loops, right_loops, left_completed, right_completed,
                          output_path, suture_num, start_time, end_time, fps,
                          loop_groups=None, step_completed=None, step_frames=None):
    """Create visualization of hand movements (matches Expert_process3.py style)"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    time = np.array(frames) / fps + start_time

    step_colors = ['', 'purple', 'green', 'purple', 'orange', 'purple', 'green']
    step_labels = ['', 'Step1:Group1', 'Step2:L→R1', 'Step3:Group2', 'Step4:R←L', 'Step5:Group3', 'Step6:L→R2']

    def add_loop_spans(ax, loops, color):
        for sf, ef, direction, count in loops:
            t_start = sf / fps + start_time
            t_end   = ef / fps + start_time
            ax.axvspan(t_start, t_end, alpha=0.2, color=color)
            ax.text((t_start + t_end) / 2, ax.get_ylim()[1] * 0.9,
                    f"{direction}(L{count})", ha='center', va='top', fontsize=9)

    def add_step_lines(ax):
        if step_completed and step_frames:
            for sn in range(1, 7):
                if step_completed[sn] and step_frames[sn] > 0:
                    t = step_frames[sn] / fps + start_time
                    ax.axvline(t, color=step_colors[sn], linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.text(t, ax.get_ylim()[0] * 1.02, step_labels[sn],
                            rotation=90, va='bottom', ha='right',
                            color=step_colors[sn], fontsize=8, weight='bold')

    # Left X
    ax = axes[0, 0]
    ax.plot(time, left_x, 'b-', linewidth=1, label='Left Hand X')
    ax.set_ylabel('X Position (pixels)', fontsize=11)
    ax.set_title('Left Hand - X Position Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    add_loop_spans(ax, left_loops, 'green')
    add_step_lines(ax)

    # Left Y
    ax = axes[0, 1]
    ax.plot(time, left_y, 'b-', linewidth=1, label='Left Hand Y')
    ax.set_ylabel('Y Position (pixels)', fontsize=11)
    ax.set_title('Left Hand - Y Position Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    add_loop_spans(ax, left_loops, 'green')
    add_step_lines(ax)

    # Right X
    ax = axes[1, 0]
    ax.plot(time, right_x, 'r-', linewidth=1, label='Right Hand X')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('X Position (pixels)', fontsize=11)
    ax.set_title('Right Hand - X Position Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    add_loop_spans(ax, right_loops, 'orange')
    add_step_lines(ax)

    # Right Y
    ax = axes[1, 1]
    ax.plot(time, right_y, 'r-', linewidth=1, label='Right Hand Y')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Y Position (pixels)', fontsize=11)
    ax.set_title('Right Hand - Y Position Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    add_loop_spans(ax, right_loops, 'orange')
    add_step_lines(ax)

    steps_n = sum(step_completed[1:]) if step_completed else 0
    groups_n = len(loop_groups) if loop_groups else 0

    fig.suptitle(
        f'Hand Movement Analysis - Suture {suture_num} ({start_time:.2f}s - {end_time:.2f}s)\n'
        f'Left Hand: {len(left_loops)} loops | Right Hand: {len(right_loops)} loops | '
        f'Steps: {steps_n}/6 | Loop Groups: {groups_n}',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_expert_movements_integrated(birdseye_video_path, output_folder, segments, suture_times, birdseye_duration=None):
    """
    Analyze expert hand movements for each suture.
    Runs on time window: (segment_end + 5) to (suture_end - 5)

    Args:
        birdseye_video_path: Path to bird's eye video
        output_folder: Where to save results
        segments: List of (start_time, end_time) tuples from holding segments
        suture_times: List of (start_time, end_time, duration) tuples for sutures
        birdseye_duration: Total duration of bird's eye video (optional)

    Returns:
        List of result dictionaries
    """
    print("\n" + "=" * 60)
    print("EXPERT HAND MOVEMENT ANALYSIS")
    print("=" * 60)
    print(f"Processing video: {os.path.basename(birdseye_video_path)}")
    print(f"Analyzing {len(suture_times)} suture(s)")
    print("=" * 60)

    results_summary = []

    for idx, ((seg_start, seg_end), (sut_start, sut_end, sut_duration)) in enumerate(zip(segments, suture_times), 1):
        analysis_start = seg_end + 5.0
        analysis_end   = sut_end - 5.0

        if birdseye_duration is not None:
            analysis_start = min(analysis_start, birdseye_duration)
            analysis_end   = min(analysis_end,   birdseye_duration)

        if analysis_end <= analysis_start:
            print(f"\nSuture {idx}: SKIPPED - Window too short")
            print(f"  Segment ends: {seg_end:.2f}s  |  Suture ends: {sut_end:.2f}s")
            print(f"  Window would be: {analysis_start:.2f}s to {analysis_end:.2f}s (invalid)")
            results_summary.append({'suture_num': idx, 'success': False, 'error': 'Window too short'})
            continue

        print(f"\nSuture {idx}:")
        print(f"  Segment:  {seg_start:.2f}s - {seg_end:.2f}s")
        print(f"  Suture:   {sut_start:.2f}s - {sut_end:.2f}s")
        print(f"  Analysis: {analysis_start:.2f}s - {analysis_end:.2f}s "
              f"(duration: {analysis_end - analysis_start:.2f}s)")

        result = analyze_hand_movements_for_suture(
            birdseye_video_path,
            analysis_start,
            analysis_end,
            idx,
            output_folder
        )
        results_summary.append(result)

    print("\n" + "=" * 60)
    print("EXPERT HAND MOVEMENT ANALYSIS COMPLETE")
    print("=" * 60)
    successful = sum(1 for r in results_summary if r['success'])
    print(f"Successfully analyzed: {successful}/{len(results_summary)} sutures")
    total_loops = sum(r.get('total_loops', 0) for r in results_summary if r['success'])
    print(f"Total loops detected: {total_loops}")
    print("=" * 60 + "\n")

    return results_summary

"""
Needle Entry / Exit Point Detector — Integrated Version
Detects needle entry (red dot) and exit (blue dot) for each suture segment.
Returns only the last annotated frame as a JPG — no video output.
Adapted from needle_entry_exit_detection_main_v3.py
"""

import os
import cv2
import numpy as np
import math

# ── Configuration ──────────────────────────────────────────────────────────
INCISION_DETECTION_FRAMES    = 150   # frames at segment start to scan for incisions
VALIDATION_WINDOW_SEC        = 8.0   # reject insertion candidate if no horizontal in N s
HORIZONTAL_THRESHOLD         = 30    # degrees from horizontal → "horizontal"
MIN_HORIZONTAL_DURATION_SEC  = 0.2   # needle must stay horizontal for at least N s
CONSECUTIVE_FRAMES_THRESHOLD = 3     # tip must be inside bbox for N consecutive frames
HORIZONTAL_EXPANSION_PERCENT = 100   # widen each incision bbox by this %
VERTICAL_EXPANSION_PERCENT   = 50    # heighten each incision bbox by this %
POST_INSERTION_WINDOW_SEC    = 8.0   # seconds after entry to search for exit
NEEDLE_LENGTH_TOLERANCE      = 0.20  # ±20 % tolerance on reference length for exit
MIN_BLUE_DOT_DELAY_SEC       = 0.3   # minimum gap (s) between entry and exit
SHOW_ANGLE_LINES             = True  # draw magenta trajectory + yellow reference lines
# ────────────────────────────────────────────────────────────────────────────

_COLORS = {
    'needle':        (0, 255, 0),
    'needle_holder': (255, 0, 0),
    'forceps':       (0, 165, 255),
    'incisions':     (0, 0, 255),
}


# ── Helper functions ────────────────────────────────────────────────────────

def _needle_length(mc):
    if mc is None or len(mc) < 2:
        return None
    pts = mc.astype(np.float32)
    cands = [pts[np.argmin(pts[:, 0])], pts[np.argmax(pts[:, 0])],
             pts[np.argmin(pts[:, 1])], pts[np.argmax(pts[:, 1])]]
    max_d = 0.0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            d = math.hypot(cands[i][0] - cands[j][0], cands[i][1] - cands[j][1])
            if d > max_d:
                max_d = d
    return max_d if max_d > 0 else None


def _needle_tip(mc):
    if mc is None or len(mc) == 0:
        return None
    return tuple(mc[np.argmax(mc[:, 1] * 2 + mc[:, 0])].astype(int))


def _needle_angle(mc):
    if mc is None or len(mc) < 2:
        return None
    [vx, vy, _, _] = cv2.fitLine(mc.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    a = abs(math.degrees(math.atan2(vy, vx)))
    return 180 - a if a > 90 else a


def _rightmost(mc):
    if mc is None or len(mc) == 0:
        return None
    return tuple(mc[np.argmax(mc[:, 0])].astype(int))


def _angle_line_pts(mc, tip, length=150):
    if mc is None or len(mc) < 2 or tip is None:
        return None, None
    [vx, vy, _, _] = cv2.fitLine(mc.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    a = math.atan2(vy, vx)
    return tip, (int(tip[0] - length * math.cos(a)), int(tip[1] - length * math.sin(a)))


def _tip_in_bbox(tip, bbox):
    if tip is None:
        return False
    px, py = tip
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def _expand_bbox(box, h_pct, v_pct, fw, fh):
    x, y, w, h = box
    ew = int(w * h_pct / 100)
    eh = int(h * v_pct / 100)
    nx = max(0, x - ew // 2)
    ny = max(0, y - eh // 2)
    nw = min(w + ew, fw - nx)
    nh = min(h + eh, fh - ny)
    return nx, ny, nw, nh


def _remove_contained(boxes, thresh=0.85):
    """Keep only boxes that are not contained within a larger box."""
    if len(boxes) <= 1:
        return boxes
    ordered = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in ordered:
        x1, y1, w1, h1 = box
        contained = False
        for kx, ky, kw, kh in kept:
            xl = max(x1, kx);  yt = max(y1, ky)
            xr = min(x1 + w1, kx + kw);  yb = min(y1 + h1, ky + kh)
            if xr > xl and yb > yt:
                inter = (xr - xl) * (yb - yt)
                if (inter / (w1 * h1)) > thresh:
                    contained = True
                    break
        if not contained:
            kept.append(box)
    return kept


def _scan_for_incisions(video_path, model, start_frame, end_frame):
    """
    Scan frames [start_frame, end_frame] for incision bounding boxes.
    Returns a list of (x, y, w, h) tuples after NMS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    found = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fc = start_frame
    while fc <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        for result in model(frame, verbose=False):
            if result.masks is None or result.boxes is None:
                continue
            for mc, cls_id in zip(result.masks.xy,
                                   result.boxes.cls.cpu().numpy()):
                if result.names[int(cls_id)] == 'incisions' and len(mc) > 0:
                    pts = mc.astype(np.int32)
                    x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
                    if not any(abs(x - sx) < 80 and abs(y - sy) < 80
                               for sx, sy, *_ in found):
                        found.append((x, y, w, h))
        fc += 1
    cap.release()
    return _remove_contained(found) if len(found) > 1 else found


# ── Core per-suture function ────────────────────────────────────────────────

def analyze_entry_exit_for_suture(video_path, output_folder, suture_num,
                                   start_time, end_time, model,
                                   global_incision_boxes=None):
    """
    Run needle entry/exit detection on one suture time window.
    Saves the last annotated frame as a JPG only when an insertion is confirmed.
    If no insertion is confirmed, returns output_file=None (no blank image saved).

    Args:
        video_path:             Path to front-view video.
        output_folder:          Folder to save the output JPG.
        suture_num:             Suture index (1-based).
        start_time:             Window start in seconds.
        end_time:               Window end in seconds.
        model:                  Already-loaded YOLO model.
        global_incision_boxes:  Pre-scanned incision boxes from video start
                                (used as fallback when segment scan finds nothing).

    Returns:
        dict with keys: suture_num, success, output_file,
                        entry_angle, entry_tip, exit_point, num_insertions
    """
    print(f"\n  [Entry/Exit] Suture {suture_num}: {start_time:.2f}s – {end_time:.2f}s")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'suture_num': suture_num, 'success': False,
                'error': 'Cannot open video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    s_frm = int(start_time * fps)
    e_frm = int(end_time   * fps)

    # ── Step 0: Detect incision boxes ─────────────────────────────────────
    # Try segment start first
    inc_scan_end = min(s_frm + INCISION_DETECTION_FRAMES, e_frm)
    fixed_incision_boxes = _scan_for_incisions(video_path, model, s_frm, inc_scan_end)

    if not fixed_incision_boxes:
        # Segment start had no incisions — use global pre-scanned boxes (video start)
        if global_incision_boxes:
            fixed_incision_boxes = list(global_incision_boxes)
            print(f"    No incisions at segment start — using {len(fixed_incision_boxes)} "
                  f"globally-detected box(es)")
        else:
            print(f"    ⚠️  No incisions found anywhere — insertion detection may fail")

    print(f"    Incision boxes: {len(fixed_incision_boxes)}")

    # ── per-suture state (all local — no globals) ─────────────────────────
    insertion_candidates = []
    confirmed_insertions = []
    needle_len_history   = []
    reference_len        = None
    # ──────────────────────────────────────────────────────────────────────

    # ══════════ SINGLE PASS — detection + blue-dot locking ════════════════
    # Combines the old Pass 1 (detection) and Pass 2 (drawing/locking) into
    # one video read. frame_store is eliminated; only the last frame is kept.
    print(f"    Pass — detection...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'suture_num': suture_num, 'success': False,
                'error': 'Cannot reopen video'}

    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frm)
    fc           = s_frm
    last_frame   = None
    last_dets    = []   # detections for the very last frame (used for annotation)

    while fc <= e_frm:
        ret, frame = cap.read()
        if not ret:
            break

        cur_needle = None
        cur_dets   = []

        for result in model(frame, verbose=False):
            if result.masks is None or result.boxes is None:
                continue
            for mc, cls_id in zip(result.masks.xy,
                                   result.boxes.cls.cpu().numpy()):
                cname = result.names[int(cls_id)]
                if len(mc) == 0:
                    continue
                pts     = mc.astype(np.int32)
                contour = pts.reshape(-1, 1, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cur_dets.append({
                    'class_name':  cname,
                    'mask_coords': mc,
                    'contour':     contour,
                    'bbox':        (x, y, w, h)
                })
                if cname == 'needle':
                    cur_needle = {
                        'tip':         _needle_tip(mc),
                        'angle':       _needle_angle(mc),
                        'mask_coords': mc
                    }

        # Track needle length before first confirmed insertion
        if cur_needle and not confirmed_insertions:
            nl = _needle_length(cur_needle['mask_coords'])
            if nl:
                needle_len_history.append(nl)

        # ── insertion detection ────────────────────────────────────────────
        if cur_needle and fixed_incision_boxes:
            tip   = cur_needle['tip']
            angle = cur_needle['angle']

            for inc_idx, inc_box in enumerate(fixed_incision_boxes):
                if any(ins['incision_idx'] == inc_idx
                       for ins in confirmed_insertions):
                    continue

                ext    = _expand_bbox(inc_box, HORIZONTAL_EXPANSION_PERCENT,
                                      VERTICAL_EXPANSION_PERCENT, fw, fh)
                inside = _tip_in_bbox(tip, ext)
                ci     = next((i for i, c in enumerate(insertion_candidates)
                               if c['incision_idx'] == inc_idx), None)

                if inside:
                    if ci is None:
                        insertion_candidates.append({
                            'incision_idx':              inc_idx,
                            'trigger_frame':             fc,
                            'consecutive_inside_frames': 1,
                            'horizontal_start_frame':    None,
                            'horizontal_frame_count':    0,
                            'entry_angle':               angle,
                            'entry_tip':                 tip,
                            'entry_mask_coords':         cur_needle['mask_coords']
                        })
                        ci = len(insertion_candidates) - 1
                    else:
                        insertion_candidates[ci]['consecutive_inside_frames'] += 1

                    cand = insertion_candidates[ci]
                    if cand['consecutive_inside_frames'] >= CONSECUTIVE_FRAMES_THRESHOLD:
                        is_horiz = (angle is not None and
                                    angle <= HORIZONTAL_THRESHOLD)
                        if is_horiz:
                            if cand['horizontal_start_frame'] is None:
                                cand['horizontal_start_frame']  = fc
                                cand['horizontal_frame_count']  = 1
                            else:
                                cand['horizontal_frame_count'] += 1
                                if (cand['horizontal_frame_count'] / fps
                                        >= MIN_HORIZONTAL_DURATION_SEC):
                                    confirmed_insertions.append({
                                        'incision_idx':      inc_idx,
                                        'trigger_frame':     cand['trigger_frame'],
                                        'entry_angle':       cand['entry_angle'],
                                        'entry_tip':         cand['entry_tip'],
                                        'entry_mask_coords': cand['entry_mask_coords'],
                                        'locked_blue_dot':   None
                                    })
                                    if needle_len_history:
                                        reference_len = float(max(needle_len_history))
                                        print(f"    ✅ Insertion #{inc_idx+1} confirmed "
                                              f"ref_len={reference_len:.1f}px")
                                    insertion_candidates.pop(ci)
                        else:
                            elapsed = (fc - cand['trigger_frame']) / fps
                            if elapsed > VALIDATION_WINDOW_SEC:
                                insertion_candidates.pop(ci)
                else:
                    if ci is not None:
                        insertion_candidates.pop(ci)

        # ── blue dot locking (inline — no second pass needed) ─────────────
        if cur_needle and reference_len is not None:
            cm = cur_needle['mask_coords']
            cl = _needle_length(cm)
            if cl is not None:
                for ins in confirmed_insertions:
                    if ins['locked_blue_dot'] is not None:
                        continue
                    frames_since = fc - ins['trigger_frame']
                    min_d_f = int(MIN_BLUE_DOT_DELAY_SEC * fps)
                    post_f  = int(POST_INSERTION_WINDOW_SEC * fps)
                    if min_d_f <= frames_since <= post_f:
                        lo = reference_len * (1 - NEEDLE_LENGTH_TOLERANCE)
                        hi = reference_len * (1 + NEEDLE_LENGTH_TOLERANCE)
                        len_ok  = lo <= cl <= hi
                        li      = np.argmin(cm[:, 0])
                        ri      = np.argmax(cm[:, 0])
                        y_tol   = cl * 0.10
                        horiz_ok = abs(float(cm[li][1]) - float(cm[ri][1])) <= y_tol
                        if len_ok and horiz_ok:
                            rm  = _rightmost(cm)
                            lm  = tuple(cm[li].astype(int))
                            entry_tip = ins['entry_tip']
                            if rm and entry_tip:
                                dist = entry_tip[0] - lm[0]
                                bx   = rm[0] - dist
                                if bx > entry_tip[0]:
                                    ins['locked_blue_dot'] = (bx, rm[1])
                                    print(f"    🔵 Blue dot locked at "
                                          f"{ins['locked_blue_dot']}")

        # Keep only the last frame and its detections (no frame_store)
        last_frame = frame.copy()
        last_dets  = cur_dets
        fc += 1

    cap.release()
    print(f"    Pass done — insertions confirmed: {len(confirmed_insertions)}")

    if last_frame is None:
        return {'suture_num': suture_num, 'success': False,
                'error': 'No frames processed'}

    # ── If no insertion confirmed, return without saving any image ─────────
    # This prevents blank/uninformative frames from being shown in the UI.
    if not confirmed_insertions:
        print(f"    ❌ No confirmed insertion — image not saved")
        return {
            'suture_num':     suture_num,
            'success':        True,
            'output_file':    None,   # no image
            'entry_angle':    None,
            'entry_tip':      None,
            'exit_point':     None,
            'num_insertions': 0,
        }

    # ══════════════════ ANNOTATE LAST FRAME ══════════════════════════════
    frame = last_frame

    # Draw object contours (skip incisions)
    for det in last_dets:
        if det['class_name'] == 'incisions':
            continue
        color = _COLORS.get(det['class_name'], (255, 255, 255))
        cv2.drawContours(frame, [det['contour']], -1, color, 2)
        x, y, w, h = det['bbox']
        cv2.putText(frame, det['class_name'], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Needle length display
    needle_det = next((d for d in last_dets if d['class_name'] == 'needle'), None)
    if needle_det:
        ll = _needle_length(needle_det['mask_coords'])
        if ll:
            cv2.putText(frame, f"Live: {ll:.1f}px", (10, fh - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if reference_len:
        cv2.putText(frame, f"Ref: {reference_len:.1f}px", (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Dots, labels, and angle lines for each confirmed insertion
    y_off = 40
    for ins in confirmed_insertions:
        entry_tip = ins['entry_tip']

        # Red dot — entry point, labelled A{suture_num}
        if entry_tip:
            cv2.circle(frame, entry_tip, 10, (0, 0, 255), -1)
            cv2.circle(frame, entry_tip, 12, (255, 255, 255), 2)
            label_a = f"A{suture_num}"
            ts_a = cv2.getTextSize(label_a, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame,
                          (entry_tip[0] + 15, entry_tip[1] - ts_a[1] - 4),
                          (entry_tip[0] + 15 + ts_a[0] + 6, entry_tip[1] + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, label_a,
                        (entry_tip[0] + 18, entry_tip[1] + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Blue dot — exit point, labelled B{suture_num}
        if ins['locked_blue_dot']:
            bd = ins['locked_blue_dot']
            cv2.circle(frame, bd, 10, (255, 0, 0), -1)
            cv2.circle(frame, bd, 12, (255, 255, 255), 2)
            label_b = f"B{suture_num}"
            ts_b = cv2.getTextSize(label_b, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(frame,
                          (bd[0] + 15, bd[1] - ts_b[1] - 4),
                          (bd[0] + 15 + ts_b[0] + 6, bd[1] + 4),
                          (0, 0, 0), -1)
            cv2.putText(frame, label_b,
                        (bd[0] + 18, bd[1] + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Angle lines
        if SHOW_ANGLE_LINES:
            em = ins['entry_mask_coords']
            et = ins['entry_tip']
            if et and em is not None and len(em) > 0:
                sp, ep = _angle_line_pts(em, et)
                if sp and ep:
                    # Magenta trajectory line
                    cv2.line(frame, sp, ep, (255, 0, 255), 3)
                    # Yellow horizontal reference line
                    lw  = sp if sp[1] >= ep[1] else ep
                    ll2 = math.hypot(ep[0] - sp[0], ep[1] - sp[1])
                    hl2 = ll2 / 2
                    cv2.line(frame,
                             (int(lw[0] - hl2), lw[1]),
                             (int(lw[0] + hl2), lw[1]),
                             (0, 255, 255), 3)
                    # Arc angle computed from line direction (for arc ONLY —
                    # does NOT touch ins['entry_angle'] which was captured at
                    # the actual entry moment)
                    dx, dy = ep[0] - sp[0], ep[1] - sp[1]
                    a_deg  = math.degrees(math.atan2(dy, dx))
                    if a_deg < 0:
                        a_deg += 360
                    arc_a = a_deg % 180
                    cv2.ellipse(frame, lw, (40, 40), 0, 0,
                                -int(arc_a), (0, 255, 255), 2)

            # Entry angle text (original captured value, not arc_a)
            ang   = ins.get('entry_angle')
            label = (f"Incision #{ins['incision_idx']+1} Entry: {ang:.1f}"
                     if ang is not None
                     else f"Incision #{ins['incision_idx']+1} Entry: N/A")
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            cv2.rectangle(frame, (10, y_off - 35),
                          (20 + ts[0], y_off + 10), (0, 0, 0), -1)
            cv2.putText(frame, label, (15, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            y_off += 60

    # ── Resize and save ─────────────────────────────────────────────────
    h, w = frame.shape[:2]
    if w > 1280:
        frame = cv2.resize(frame, (1280, int(h * 1280 / w)),
                           interpolation=cv2.INTER_AREA)

    base      = os.path.splitext(os.path.basename(video_path))[0]
    out_fname = f"{base}_Suture_{suture_num}_entry_exit.jpg"
    cv2.imwrite(os.path.join(output_folder, out_fname), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    entry_angle = confirmed_insertions[0]['entry_angle']
    entry_tip   = confirmed_insertions[0]['entry_tip']
    exit_point  = confirmed_insertions[0]['locked_blue_dot']

    print(f"    ✅ Saved: {out_fname}  "
          f"entry={entry_tip}  exit={exit_point}")

    return {
        'suture_num':     suture_num,
        'success':        True,
        'output_file':    out_fname,
        'entry_angle':    entry_angle,
        'entry_tip':      entry_tip,
        'exit_point':     exit_point,
        'num_insertions': len(confirmed_insertions),
    }


# ── Batch entry point called from analyzer.py ──────────────────────────────

def analyze_entry_exit_integrated(video_path, output_folder, suture_times, model):
    """
    Run entry/exit analysis for every suture.
    Performs a single global incision scan of the video start so that
    sutures 2-N can reuse the incision locations even if the wound is
    no longer clearly visible at the start of those segments.

    Args:
        video_path:    Front-view video path.
        output_folder: Where to save JPGs.
        suture_times:  List of (start, end, duration) tuples.
        model:         Already-loaded YOLO model.

    Returns:
        List of result dicts (one per suture).
    """
    print("\n" + "=" * 60)
    print("NEEDLE ENTRY / EXIT DETECTION")
    print("=" * 60)
    print(f"Video:   {os.path.basename(video_path)}")
    print(f"Sutures: {len(suture_times)}")
    print("=" * 60)

    # ── Global incision scan (video start, done once for all sutures) ──────
    # The wound location is fixed throughout the procedure, so incision boxes
    # found early in the video apply to every suture segment.
    global_scan_end = INCISION_DETECTION_FRAMES * 2   # 300 frames ≈ 10 s at 30 fps
    print(f"Global incision scan: first {global_scan_end} frames...")
    global_boxes = _scan_for_incisions(video_path, model, 0, global_scan_end)
    print(f"  → {len(global_boxes)} incision box(es) found globally")

    results = []
    for idx, (sut_start, sut_end, _) in enumerate(suture_times, 1):
        result = analyze_entry_exit_for_suture(
            video_path, output_folder, idx, sut_start, sut_end, model,
            global_incision_boxes=global_boxes
        )
        results.append(result)

    ok = sum(1 for r in results if r.get('success') and r.get('output_file'))
    print(f"\nEntry/Exit complete — {ok}/{len(results)} sutures with confirmed insertion")
    print("=" * 60 + "\n")
    return results

from flask import Flask, render_template, request, redirect, url_for, send_file, flash, Response, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from analyzer import analyze_video
import zipfile
import io
import threading
import time
import queue
import re
import subprocess
import shutil


def is_h264(file_path):
    """Check if video is already H.264 codec - skip re-encoding if so."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1",
             file_path],
            capture_output=True, text=True
        )
        return result.stdout.strip().lower() == "h264"
    except Exception:
        return False


def remux_video_for_web(input_path):
    """Re-encode to H.264 + faststart. Skips if already H.264."""
    if is_h264(input_path):
        print(f"[REMUX] Already H.264, skipping: {os.path.basename(input_path)}")
        return False  # False = nothing was done (not an error)

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
            shutil.move(tmp_path, input_path)
            print(f"[REMUX] Re-encoded to H.264: {os.path.basename(input_path)}")
            return True
        else:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"[REMUX] Failed: {os.path.basename(input_path)}")
            return False
    except Exception as e:
        print(f"[REMUX] Error: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def auto_remux_all_existing_videos(results_folder):
    """On startup, remux any hand_movements.mp4 that needs fixing."""
    count = 0
    if not os.path.exists(results_folder):
        return
    for root, dirs, files in os.walk(results_folder):
        for f in files:
            if f.endswith("_hand_movements.mp4"):
                full_path = os.path.join(root, f)
                if remux_video_for_web(full_path):
                    count += 1
    if count:
        print(f"[REMUX] Fixed {count} existing video(s) for web playback.")
    else:
        print("[REMUX] All existing videos already web-ready.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Auto-fix any existing videos for browser playback on startup
auto_remux_all_existing_videos(RESULTS_FOLDER)

# Global dictionary to track processing status
processing_status = {}
status_queues = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_processing_status(folder_name, status, progress=0, message=""):
    """Update the processing status for a video"""
    processing_status[folder_name] = {
        'status': status,
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }
    
    if folder_name in status_queues:
        try:
            status_queues[folder_name].put({
                'status': status,
                'progress': progress,
                'message': message
            })
        except:
            pass

def process_video_thread(video_path, result_folder, folder_name, birdseye_path=None):
    """Process video in background thread with status updates"""
    print(f"[DEBUG] process_video_thread STARTED for {folder_name}")
    print(f"[DEBUG] Video path: {video_path}")
    print(f"[DEBUG] Result folder: {result_folder}")
    if birdseye_path:
        print(f"[DEBUG] Bird's eye video: {birdseye_path}")
    
    try:
        update_processing_status(folder_name, 'processing', 10, 'Starting video analysis...')
        print(f"[DEBUG] Status updated to processing")

        from analyzer import analyze_video_with_progress
        
        def progress_callback(progress, message):
            print(f"[DEBUG] Progress: {progress}% - {message}")
            update_processing_status(folder_name, 'processing', progress, message)
        
        analyze_video_with_progress(
            video_path, 
            result_folder, 
            progress_callback,
            birdseye_video_path=birdseye_path
        )
        
        update_processing_status(folder_name, 'completed', 100, 'Processing complete!')
        print(f"[DEBUG] Processing COMPLETED for {folder_name}")
        
    except Exception as e:
        print(f"[DEBUG] ERROR in processing: {str(e)}")
        update_processing_status(folder_name, 'error', 0, f'Error: {str(e)}')
        if os.path.exists(result_folder):
            import shutil
            shutil.rmtree(result_folder)

@app.route('/')
def index():
    results = []
    if os.path.exists(RESULTS_FOLDER):
        for folder_name in os.listdir(RESULTS_FOLDER):
            folder_path = os.path.join(RESULTS_FOLDER, folder_name)
            if os.path.isdir(folder_path):
                creation_time = os.path.getctime(folder_path)
                creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                
                txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                
                status_info = processing_status.get(folder_name, {})
                is_processing = status_info.get('status') == 'processing'
                
                results.append({
                    'folder_name': folder_name,
                    'creation_date': creation_date,
                    'has_results': len(txt_files) > 0,
                    'is_processing': is_processing,
                    'status': status_info.get('status', 'completed'),
                    'progress': status_info.get('progress', 100),
                    'message': status_info.get('message', '')
                })
    
    results.sort(key=lambda x: x['creation_date'], reverse=True)
    
    return render_template('index.html', results=results)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['video']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        base_name = os.path.splitext(filename)[0]
        folder_name = base_name
        
        result_folder = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
        
        if os.path.exists(result_folder):
            txt_files = [f for f in os.listdir(result_folder) if f.endswith('.txt')]
            if txt_files:
                flash(f'Video "{filename}" has already been processed! Showing existing results.', 'warning')
                return redirect(url_for('view_results', folder_name=folder_name))
        
        os.makedirs(result_folder, exist_ok=True)
        
        video_path = os.path.join(result_folder, filename)
        file.save(video_path)
        
        birdseye_path = None
        if 'birdseye_video' in request.files:
            birdseye_file = request.files['birdseye_video']
            if birdseye_file.filename != '' and allowed_file(birdseye_file.filename):
                birdseye_filename = secure_filename(birdseye_file.filename)
                birdseye_path = os.path.join(result_folder, birdseye_filename)
                birdseye_file.save(birdseye_path)
                print(f"[DEBUG] Bird's eye video uploaded: {birdseye_filename}")
        
        update_processing_status(folder_name, 'processing', 5, 'Uploading video...')
        
        print(f"[DEBUG] Starting background thread for: {folder_name}")
        thread = threading.Thread(
            target=process_video_thread,
            args=(video_path, result_folder, folder_name, birdseye_path)
        )
        thread.daemon = True
        thread.start()
        print(f"[DEBUG] Thread started: {thread.is_alive()}")

        if birdseye_path:
            flash(f'Videos uploaded! Processing started for: {filename} (with bird\'s eye view)', 'success')
        else:
            flash(f'Video uploaded! Processing started for: {filename}', 'success')
        return redirect(url_for('index'))
    
    flash('Invalid file type. Allowed: mp4, avi, mov, mkv', 'error')
    return redirect(url_for('index'))

@app.route('/status/<folder_name>')
def get_status(folder_name):
    status = processing_status.get(folder_name, {
        'status': 'unknown',
        'progress': 0,
        'message': 'No status available'
    })
    return jsonify(status)

@app.route('/stream/<folder_name>')
def stream_status(folder_name):
    def generate():
        q = queue.Queue()
        status_queues[folder_name] = q
        
        try:
            while True:
                try:
                    status = q.get(timeout=1)
                    yield f"data: {jsonify(status).get_data(as_text=True)}\n\n"
                    
                    if status.get('status') in ['completed', 'error']:
                        break
                except queue.Empty:
                    current_status = processing_status.get(folder_name, {})
                    if current_status.get('status') in ['completed', 'error']:
                        break
                    continue
        finally:
            if folder_name in status_queues:
                del status_queues[folder_name]
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/results/<folder_name>')
def view_results(folder_name):
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    
    if not os.path.exists(folder_path):
        flash('Results not found', 'error')
        return redirect(url_for('index'))
    
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_content = None
    
    if txt_files:
        txt_path = os.path.join(folder_path, txt_files[0])
        with open(txt_path, 'r') as f:
            txt_content = f.read()
    
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_file = video_files[0] if video_files else None

    available_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]
    
    suture_data = []
    
    if txt_content:
        lines = txt_content.split('\n')
        
        for i, line in enumerate(lines):
            match = re.match(r'Suture (\d+):\s*([\d.]+)s\s*-\s*([\d.]+)s\s*\(Duration:\s*([\d.]+)s\)', line)
            if match:
                suture_num = int(match.group(1))
                start_time = match.group(2)
                end_time = match.group(3)
                duration = match.group(4)
                
                suture_info = {
                    'suture_num': suture_num,
                    'time_range': f"{start_time}s - {end_time}s",
                    'duration': f"{duration}s",
                    'holding_detected': False,
                    'holding_pct': None,
                    'holding_image': None,
                    'insertion_detected': False,
                    'insertion_angle': None,
                    'insertion_image': None
                }
                
                for j in range(i+1, min(i+10, len(lines))):
                    check_line = lines[j]
                    
                    if 'Actual Holding:' in check_line:
                        holding_match = re.search(r'([\d.]+)%', check_line)
                        if holding_match:
                            suture_info['holding_pct'] = holding_match.group(1)
                            suture_info['holding_detected'] = True
                    
                    if 'Image:' in check_line and '33percent' in check_line:
                        img_match = re.search(r'Image:\s*(.+?_Suture_\d+_33percent\.jpg)', check_line)
                        if img_match:
                            suture_info['holding_image'] = img_match.group(1).strip()
                    
                    if re.match(r'Suture \d+:', check_line):
                        break
                
                suture_data.append(suture_info)
        
        segment_section = txt_content.split("HOLDING SEGMENTS:")
        if len(segment_section) > 1:
            segment_lines = segment_section[1].split('\n')
            
            for i, line in enumerate(segment_lines):
                seg_match = re.match(r'Segment (\d+):', line)
                if seg_match:
                    segment_num = int(seg_match.group(1))
                    
                    insertion_angle = None
                    insertion_image = None
                    insertion_detected = False
                    
                    for j in range(i+1, min(i+10, len(segment_lines))):
                        check_line = segment_lines[j]
                        
                        if 'Insertion Angle:' in check_line:
                            if 'Not Detected' not in check_line:
                                angle_match = re.search(r'([\d.]+)°', check_line)
                                if angle_match:
                                    insertion_angle = angle_match.group(1)
                                    insertion_detected = True
                        
                        if 'Image:' in check_line:
                            img_match = re.search(r'Image:\s*(.+?\.jpg)', check_line)
                            if img_match and 'insertion' in check_line.lower():
                                insertion_image = img_match.group(1).strip()
                                if insertion_image in available_images:
                                    insertion_detected = True
                        
                        if re.match(r'Segment \d+:', check_line):
                            break
                    
                    if segment_num <= len(suture_data):
                        suture_data[segment_num - 1]['insertion_detected'] = insertion_detected
                        suture_data[segment_num - 1]['insertion_angle'] = insertion_angle
                        suture_data[segment_num - 1]['insertion_image'] = insertion_image
    
    rotation_graphs = [f for f in os.listdir(folder_path) if f.endswith('_rotation.png')]
    rotation_graphs.sort()
    
    # Get knot analysis image
    knot_image = None
    knot_files = [f for f in os.listdir(folder_path) if f.endswith('_knot_analysis.jpg')]
    if knot_files:
        knot_image = knot_files[0]
    
    # Get expert hand movement graphs
    expert_movement_graphs = [f for f in os.listdir(folder_path) if f.endswith('_hand_movements.png')]
    expert_movement_graphs.sort()

    expert_movement_videos = [f for f in os.listdir(folder_path) if f.endswith('_hand_movements.mp4')]
    expert_movement_videos.sort()
    
    return render_template('results.html', 
                         folder_name=folder_name, 
                         txt_content=txt_content,
                         video_file=video_file,
                         suture_data=suture_data,
                         rotation_graphs=rotation_graphs,
                         knot_image=knot_image,
                         expert_movement_graphs=expert_movement_graphs,
                         expert_movement_videos=expert_movement_videos)

@app.route('/download/<folder_name>')
def download_results(folder_name):
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    
    if not os.path.exists(folder_path):
        flash('Results not found', 'error')
        return redirect(url_for('index'))
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join(folder_name, file)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{folder_name}_results.zip'
    )

@app.route('/download_txt/<folder_name>')
def download_txt(folder_name):
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    
    if not os.path.exists(folder_path):
        flash('Results not found', 'error')
        return redirect(url_for('index'))
    
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if txt_files:
        txt_path = os.path.join(folder_path, txt_files[0])
        return send_file(txt_path, as_attachment=True)
    
    flash('No text file found', 'error')
    return redirect(url_for('view_results', folder_name=folder_name))

@app.route('/suture_image/<folder_name>/<filename>')
def serve_suture_image(folder_name, filename):
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    return send_file(os.path.join(folder_path, filename), mimetype='image/jpeg')

@app.route('/rotation_graph/<folder_name>/<filename>')
def serve_rotation_graph(folder_name, filename):
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    return send_file(os.path.join(folder_path, filename), mimetype='image/png')

@app.route('/delete/<folder_name>', methods=['POST'])
def delete_results(folder_name):
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    
    if os.path.exists(folder_path):
        import shutil
        shutil.rmtree(folder_path)
        
        if folder_name in processing_status:
            del processing_status[folder_name]
        
        flash(f'Results deleted: {folder_name}', 'success')
    else:
        flash('Results not found', 'error')
    
    return redirect(url_for('index'))
@app.route('/knot_image/<folder_name>/<filename>')
def serve_knot_image(folder_name, filename):
    """Serve knot analysis image"""
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    return send_file(os.path.join(folder_path, filename), mimetype='image/jpeg')

@app.route('/expert_movement_graph/<folder_name>/<filename>')
def serve_expert_movement_graph(folder_name, filename):
    """Serve expert hand movement graph"""
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    return send_file(os.path.join(folder_path, filename), mimetype='image/png')
    
@app.route('/expert_movement_video/<folder_name>/<filename>')
def serve_expert_movement_video(folder_name, filename):
    """Stream video with byte-range support so browsers can seek and play."""
    folder_path = os.path.join(app.config['RESULTS_FOLDER'], folder_name)
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        return Response("File not found", status=404)

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)

    no_cache_headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
    }

    if range_header:
        byte_range = range_header.replace('bytes=', '').strip()
        parts = byte_range.split('-')
        start = int(parts[0]) if parts[0] else 0
        end   = int(parts[1]) if parts[1] else file_size - 1
        end   = min(end, file_size - 1)
        length = end - start + 1

        def generate_chunk():
            with open(file_path, 'rb') as f:
                f.seek(start)
                remaining = length
                chunk_size = 1024 * 256
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            'Content-Range':  f'bytes {start}-{end}/{file_size}',
            'Accept-Ranges':  'bytes',
            'Content-Length': str(length),
            'Content-Type':   'video/mp4',
            **no_cache_headers,
        }
        return Response(generate_chunk(), status=206, headers=headers)

    headers = {
        'Accept-Ranges':  'bytes',
        'Content-Length': str(file_size),
        'Content-Type':   'video/mp4',
        **no_cache_headers,
    }
    def generate_full():
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(1024 * 256)
                if not data:
                    break
                yield data

    return Response(generate_full(), status=200, headers=headers)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

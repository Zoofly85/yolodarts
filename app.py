from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import cali
import logging
import math

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize webcams
caps = []
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        caps.append(cap)
        logger.info(f"Camera {i} initialized successfully")
    else:
        logger.warning(f"Failed to open camera {i}")
        caps.append(None)

# Initialize YOLO models
dart_model = YOLO("darttipbox1.1.pt")
calibration_model = YOLO("best (20).pt")
logger.info("YOLO models loaded successfully")

# Initialize calibration data
calibration_data = [
    {'transform_matrix': None, 'bull_center': None, 'current_scores': [], 'click_score': None},
    {'transform_matrix': None, 'bull_center': None, 'current_scores': [], 'click_score': None},
    {'transform_matrix': None, 'bull_center': None, 'current_scores': [], 'click_score': None}
]

# Global variables for dart tracking
global_tracks = {}
current_dart_id = 0
max_disappeared = 10
max_darts = 3
confidence_threshold = 0.4  # Increased confidence threshold

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0
    
    return intersection / union

def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def process_frame(frame, camera_index):
    global global_tracks, current_dart_id
    
    results = dart_model(frame, verbose=False)
    annotated_frame = results[0].plot()

    detected_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = box.conf.item()
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detected_boxes.append([x1, y1, x2, y2])
                logger.info(f"Camera {camera_index}: Detected dart with confidence {confidence:.2f} at location: ({(x1+x2)/2:.2f}, {(y1+y2)/2:.2f})")
            else:
                logger.info(f"Camera {camera_index}: Ignored detection with low confidence {confidence:.2f}")

    logger.info(f"Camera {camera_index}: Detected {len(detected_boxes)} darts")

    # Match detections to global tracks
    unmatched_detections = list(range(len(detected_boxes)))
    matched_tracks = set()
    for track_id in list(global_tracks.keys()):
        if camera_index in global_tracks[track_id]['cameras']:
            if len(unmatched_detections) > 0:
                ious = [calculate_iou(global_tracks[track_id]['cameras'][camera_index]['box'], detected_boxes[i]) for i in unmatched_detections]
                if ious and max(ious) > 0.1:
                    best_match = unmatched_detections[np.argmax(ious)]
                    unmatched_detections.remove(best_match)
                    global_tracks[track_id]['disappeared'] = 0
                    global_tracks[track_id]['cameras'][camera_index]['box'] = detected_boxes[best_match]
                    matched_tracks.add(track_id)
                else:
                    global_tracks[track_id]['disappeared'] += 1
            else:
                global_tracks[track_id]['disappeared'] += 1

    # Create new global tracks for unmatched detections
    for i in unmatched_detections:
        box = detected_boxes[i]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        new_center = (center_x, center_y)
        
        is_new_dart = True
        for track_id, track in global_tracks.items():
            if camera_index in track['cameras']:
                if calculate_distance(new_center, track['cameras'][camera_index]['center']) < 20:
                    is_new_dart = False
                    break
        
        if is_new_dart and len(global_tracks) < max_darts:
            while current_dart_id in global_tracks:
                current_dart_id = (current_dart_id + 1) % max_darts
            global_tracks[current_dart_id] = {
                'cameras': {},
                'disappeared': 0
            }
            global_tracks[current_dart_id]['cameras'][camera_index] = {
                'box': box,
                'center': new_center,
            }
            logger.info(f"New dart {current_dart_id} detected. Camera {camera_index}: x={center_x:.2f}, y={center_y:.2f}")
            current_dart_id = (current_dart_id + 1) % max_darts

    # Remove disappeared tracks
    for track_id in list(global_tracks.keys()):
        if global_tracks[track_id]['disappeared'] > max_disappeared:
            logger.info(f"Dart {track_id} removed")
            del global_tracks[track_id]

    # Draw boxes and dots for all active tracks
    for track_id, track in global_tracks.items():
        if camera_index in track['cameras']:
            x1, y1, x2, y2 = [int(coord) for coord in track['cameras'][camera_index]['box']]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_x, center_y = int(track['cameras'][camera_index]['center'][0]), int(track['cameras'][camera_index]['center'][1])
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, f"Dart {track_id} not visible", (10, 30 + track_id * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return annotated_frame

def gen_frames(camera_index, draw_scoring=False):
    if caps[camera_index] is None:
        logger.warning(f"Camera {camera_index} is not available")
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
    else:
        while True:
            success, frame = caps[camera_index].read()
            if not success:
                logger.warning(f"Failed to capture frame from camera {camera_index}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            else:
                try:
                    frame = process_frame(frame, camera_index)
                    if draw_scoring and calibration_data[camera_index]['transform_matrix'] is not None:
                        frame = cali.draw_dartboard(frame, 
                                                    calibration_data[camera_index]['bull_center'],
                                                    calibration_data[camera_index]['transform_matrix'])
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    logger.error(f"Error processing frame from camera {camera_index}: {str(e)}")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calibration')
def calibration():
    return render_template('calibration.html', camera_status=[cap is not None for cap in caps])

@app.route('/score')
def score():
    return render_template('score.html')

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    draw_scoring = request.args.get('draw_scoring', 'false').lower() == 'true'
    return Response(gen_frames(camera_id - 1, draw_scoring),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate/<int:camera_id>', methods=['POST'])
def calibrate_camera(camera_id):
    camera_index = camera_id - 1
    if caps[camera_index] is None:
        logger.warning(f"Camera {camera_id} is not available for calibration")
        return jsonify({'message': f'Camera {camera_id} is not available'})
    
    # Perform calibration
    success, frame = caps[camera_index].read()
    if success:
        results = calibration_model(frame, verbose=False)
        calibration_points, bull_center = cali.perform_calibration(results)
        
        logger.info(f"Calibration points for camera {camera_id}: {calibration_points}")
        logger.info(f"Bull center for camera {camera_id}: {bull_center}")
        
        if len(calibration_points) != 4 or bull_center is None:
            logger.warning(f"Calibration failed for camera {camera_id}: Insufficient calibration points or missing bull center")
            return jsonify({'message': f'Calibration failed for camera {camera_id}. Please try again.'})
        
        transform_matrix = cali.calculate_transform(calibration_points)
        
        if transform_matrix is not None:
            calibration_data[camera_index]['transform_matrix'] = transform_matrix
            calibration_data[camera_index]['bull_center'] = bull_center
            calibration_data[camera_index]['current_scores'] = []
            calibration_data[camera_index]['click_score'] = None
            logger.info(f"Calibration successful for camera {camera_id}")
            logger.info(f"Transform matrix: {transform_matrix}")
            logger.info(f"Bull center: {bull_center}")
            return jsonify({'message': f'Calibration successful for camera {camera_id}'})
        else:
            logger.warning(f"Calibration failed for camera {camera_id}: Invalid transform matrix")
            return jsonify({'message': f'Calibration failed for camera {camera_id}. Please try again.'})
    else:
        logger.error(f"Failed to capture frame from camera {camera_id} for calibration")
        return jsonify({'message': f'Failed to capture frame from camera {camera_id}'})

def calculate_score(dart_location, transform_matrix, bull_center):
    # Transform dart location
    dart_x, dart_y = dart_location
    transformed_point = np.dot(transform_matrix, np.array([dart_x, dart_y, 1]))
    transformed_x, transformed_y = transformed_point[:2] / transformed_point[2]

    # Calculate distance from bull's eye
    dx = transformed_x - bull_center[0]
    dy = transformed_y - bull_center[1]
    distance = math.sqrt(dx*dx + dy*dy)

    # Define scoring zones (you may need to adjust these values)
    if distance < 6.35:  # Double bull (12.7mm diameter)
        return 50
    elif distance < 15.9:  # Single bull (31.8mm diameter)
        return 25
    else:
        # Calculate angle
        angle = math.atan2(dy, dx) * 180 / math.pi
        if angle < 0:
            angle += 360

        # Define scoring segments
        segments = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
        segment_angle = 360 / len(segments)
        segment_index = int(angle / segment_angle)
        score = segments[segment_index]

        # Check if it's in the double or triple ring
        if 107 < distance < 115:  # Triple ring (assuming 8mm width)
            return score * 3
        elif 162 < distance < 170:  # Double ring (assuming 8mm width)
            return score * 2
        else:
            return score

@app.route('/get_score', methods=['GET'])
def get_score():
    scores = []
    locations = []
    
    for camera_index in range(3):
        if calibration_data[camera_index]['transform_matrix'] is None:
            logger.warning(f"Camera {camera_index + 1} not calibrated")
            scores.append(None)
            locations.append(None)
            continue
        
        camera_scores = []
        camera_locations = []
        
        for track_id, track in global_tracks.items():
            if camera_index in track['cameras']:
                dart_location = track['cameras'][camera_index]['center']
                score = calculate_score(dart_location, 
                                        calibration_data[camera_index]['transform_matrix'],
                                        calibration_data[camera_index]['bull_center'])
                camera_scores.append(score)
                camera_locations.append(dart_location)
                logger.info(f"Dart {track_id} detected by camera {camera_index + 1} at location: {dart_location}, Score: {score}")
            else:
                logger.info(f"Dart {track_id} not visible in camera {camera_index + 1}")
        
        if camera_scores:
            best_camera_score = max(camera_scores)
            best_camera_location = camera_locations[camera_scores.index(best_camera_score)]
            scores.append(best_camera_score)
            locations.append(best_camera_location)
            logger.info(f"Camera {camera_index + 1} best score: {best_camera_score}, Location: {best_camera_location}")
        else:
            scores.append(None)
            locations.append(None)
            logger.info(f"No dart detected by camera {camera_index + 1}")
    
    valid_scores = [score for score in scores if score is not None]
    if valid_scores:
        best_score = max(valid_scores)
        best_camera_index = scores.index(best_score)
        best_location = locations[best_camera_index]
        logger.info(f"Overall best score: {best_score}, Location: {best_location}, Camera: {best_camera_index + 1}")
        return jsonify({'score': best_score, 'location': best_location, 'camera_scores': scores})
    else:
        logger.info("No darts detected by any camera")
        return jsonify({'score': None, 'location': None, 'camera_scores': scores})

if __name__ == '__main__':
    app.run(debug=True)
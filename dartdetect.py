import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("C:\\Users\\eugen\\Downloads\\darttipbox1.1.pt")

# Open three webcams
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)

global_tracks = {}
current_dart_id = 0
max_disappeared = 10
MAX_DARTS_BEFORE_RESET = 3

def calculate_iou(box1, box2):
    # Calculate IoU between two boxes
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
    
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    detected_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detected_boxes.append([x1, y1, x2, y2])

    # Match detections to global tracks
    unmatched_detections = list(range(len(detected_boxes)))
    for track_id in list(global_tracks.keys()):
        if camera_index in global_tracks[track_id]['cameras']:
            if len(unmatched_detections) > 0:
                ious = [calculate_iou(global_tracks[track_id]['cameras'][camera_index]['box'], detected_boxes[i]) for i in unmatched_detections]
                if ious and max(ious) > 0.1:
                    best_match = unmatched_detections[np.argmax(ious)]
                    unmatched_detections.remove(best_match)
                    global_tracks[track_id]['disappeared'] = 0
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
        for track in global_tracks.values():
            if camera_index in track['cameras']:
                if calculate_distance(new_center, track['cameras'][camera_index]['center']) < 20:
                    is_new_dart = False
                    break
        
        if is_new_dart:
            if current_dart_id not in global_tracks:
                global_tracks[current_dart_id] = {
                    'cameras': {},
                    'disappeared': 0
                }
            global_tracks[current_dart_id]['cameras'][camera_index] = {
                'box': box,
                'center': new_center,
            }
            print(f"All Cameras, New dart {current_dart_id} detected. Camera {camera_index}: x={center_x:.2f}, y={center_y:.2f}")
            if len(global_tracks[current_dart_id]['cameras']) == 3:
                current_dart_id += 1

    # Remove disappeared tracks
    for track_id in list(global_tracks.keys()):
        if global_tracks[track_id]['disappeared'] > max_disappeared:
            print(f"All Cameras, Dart {track_id} removed")
            del global_tracks[track_id]

    # Reset counter if all darts are removed
    if len(global_tracks) == 0:
        current_dart_id = 0
        print("All darts removed. Resetting counter.")

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

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if not (ret1 and ret2 and ret3):
        break

    annotated_frame1 = process_frame(frame1, 0)
    annotated_frame2 = process_frame(frame2, 1)
    annotated_frame3 = process_frame(frame3, 2)

    # Resize frames to fit on screen
    scale = 0.5
    annotated_frame1 = cv2.resize(annotated_frame1, (0, 0), fx=scale, fy=scale)
    annotated_frame2 = cv2.resize(annotated_frame2, (0, 0), fx=scale, fy=scale)
    annotated_frame3 = cv2.resize(annotated_frame3, (0, 0), fx=scale, fy=scale)

    # Stack frames horizontally
    stacked_frames = np.hstack((annotated_frame1, annotated_frame2, annotated_frame3))

    cv2.imshow("YOLOv8 Inference - 3 Cameras", stacked_frames)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()
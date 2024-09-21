import cv2
import numpy as np
from ultralytics import YOLO
import math

# Define scoring zones as a global constant
SCORING_ZONES = {
    'DB': 6.35,
    'SB': 15.9,
    'Triple_Inner': 97,
    'Triple_Outer': 107,
    'Double_Inner': 160,
    'Double_Outer': 170
}

MODEL_WIDTH, MODEL_HEIGHT = 1280, 720

def perform_calibration(results):
    calibration_points = {}
    bull_center = None
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if class_name == 'bull':
                bull_center = center
            elif class_name.startswith('cal'):
                calibration_points[class_name] = center

    return calibration_points, bull_center

def calculate_transform(calibration_points):
    if 'cal1' in calibration_points and 'cal2' in calibration_points and 'cal3' in calibration_points and 'cal4' in calibration_points:
        std_points = np.array([
            [0, -SCORING_ZONES['Double_Outer']],   # Top (cal1)
            [SCORING_ZONES['Double_Outer'], 0],    # Right (cal2)
            [0, SCORING_ZONES['Double_Outer']],    # Bottom (cal3)
            [-SCORING_ZONES['Double_Outer'], 0]    # Left (cal4)
        ], dtype=np.float32)

        src_points = np.array([
            calibration_points['cal1'],
            calibration_points['cal2'],
            calibration_points['cal3'],
            calibration_points['cal4']
        ], dtype=np.float32)

        return cv2.getPerspectiveTransform(src_points, std_points)
    return None

def calculate_score(x, y, transform_matrix):
    point = np.array([[[x, y]]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, transform_matrix)[0][0]

    distance = np.linalg.norm(transformed_point)
    angle = math.atan2(transformed_point[1], transformed_point[0])
    
    angle = (angle + math.pi/2 + math.pi/10) % (2 * math.pi)

    sector_scores = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
    sector_index = int(angle / (2 * math.pi) * 20)
    base_score = sector_scores[sector_index]

    if distance <= SCORING_ZONES['DB']:
        return "DB"
    elif distance <= SCORING_ZONES['SB']:
        return "SB"
    elif SCORING_ZONES['Triple_Inner'] < distance <= SCORING_ZONES['Triple_Outer']:
        return f"T{base_score}"
    elif SCORING_ZONES['Double_Inner'] < distance <= SCORING_ZONES['Double_Outer']:
        return f"D{base_score}"
    elif distance <= SCORING_ZONES['Double_Outer']:
        return str(base_score)
    else:
        return "0"

def draw_dartboard(frame, bull_center, transform_matrix):
    if bull_center is None or transform_matrix is None:
        return frame

    radii = list(SCORING_ZONES.values())
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    
    for radius, color in zip(radii, colors):
        circle_points = []
        for angle in range(0, 360, 5):
            x = int(radius * math.cos(math.radians(angle)))
            y = int(radius * math.sin(math.radians(angle)))
            circle_points.append([x, y])
        
        circle_points = np.array(circle_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(circle_points, np.linalg.inv(transform_matrix))
        cv2.polylines(frame, [transformed_points.reshape(-1, 2).astype(int)], True, color, 2)

    for i in range(20):
        angle = i * (2 * math.pi / 20)
        x = int(SCORING_ZONES['Double_Outer'] * math.cos(angle))
        y = int(SCORING_ZONES['Double_Outer'] * math.sin(angle))
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, np.linalg.inv(transform_matrix))[0][0].astype(int)
        cv2.line(frame, bull_center, tuple(transformed_point), (128, 128, 128), 1)

    return frame

def process_frame(frame, model, calibration_data):
    frame_resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
    
    if calibration_data['transform_matrix'] is None:
        results = model(frame_resized, verbose=False)
        calibration_points, bull_center = perform_calibration(results)
        transform_matrix = calculate_transform(calibration_points)
        if transform_matrix is not None:
            calibration_data['transform_matrix'] = transform_matrix
            calibration_data['bull_center'] = bull_center
            print("Calibration complete")
    else:
        frame_resized = draw_dartboard(frame_resized, calibration_data['bull_center'], calibration_data['transform_matrix'])
        
        results = model(frame_resized, verbose=False)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls = int(box.cls[0])
                class_name = result.names[cls]
                if class_name == 'darttip':
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    score = calculate_score(center[0], center[1], calibration_data['transform_matrix'])
                    calibration_data['current_scores'].append(score)
                    cv2.circle(frame_resized, center, 5, (0, 0, 255), -1)
                    cv2.putText(frame_resized, f"Score: {score}", (center[0]+10, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display last 3 scores
    scores = calibration_data['current_scores'][-3:]
    score_text = ' '.join(map(str, scores))
    cv2.putText(frame_resized, f"Scores: {score_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display clicked score
    if calibration_data['click_score'] is not None:
        cv2.putText(frame_resized, f"Clicked Score: {calibration_data['click_score']}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display calibration status
    status = "Calibrated" if calibration_data['transform_matrix'] is not None else "Not Calibrated"
    cv2.putText(frame_resized, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame_resized

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_data = param
        if calibration_data['transform_matrix'] is not None:
            score = calculate_score(x, y, calibration_data['transform_matrix'])
            print(f"Clicked position score: {score}")
            calibration_data['click_score'] = score

def main():
    model = YOLO("C:\\Users\\eugen\\Downloads\\best (20).pt")
    
    # Initialize three webcams
    caps = [cv2.VideoCapture(i) for i in range(3)]
    
    windows = ["Camera 1", "Camera 2", "Camera 3"]
    for window in windows:
        cv2.namedWindow(window)

    calibration_data = [
        {'transform_matrix': None, 'bull_center': None, 'current_scores': [], 'click_score': None},
        {'transform_matrix': None, 'bull_center': None, 'current_scores': [], 'click_score': None},
        {'transform_matrix': None, 'bull_center': None, 'current_scores': [], 'click_score': None}
    ]

    # Set mouse callback for each window
    for i, window in enumerate(windows):
        cv2.setMouseCallback(window, mouse_callback, calibration_data[i])

    while True:
        frames = []
        for cap in caps:
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                continue
            frames.append(frame)

        processed_frames = []
        for i, frame in enumerate(frames):
            processed_frame = process_frame(frame, model, calibration_data[i])
            processed_frames.append(processed_frame)

        # Display frames
        for i, frame in enumerate(processed_frames):
            cv2.imshow(windows[i], frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in [ord("1"), ord("2"), ord("3")]:
            camera_index = int(chr(key)) - 1
            print(f"Recalibrating Camera {camera_index + 1}")
            calibration_data[camera_index]['transform_matrix'] = None
            calibration_data[camera_index]['bull_center'] = None
            calibration_data[camera_index]['current_scores'] = []
            calibration_data[camera_index]['click_score'] = None
        elif key == ord("r"):
            print("Recalibrating all cameras...")
            for data in calibration_data:
                data['transform_matrix'] = None
                data['bull_center'] = None
                data['current_scores'] = []
                data['click_score'] = None

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
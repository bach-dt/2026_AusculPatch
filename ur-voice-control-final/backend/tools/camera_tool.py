import cv2
import numpy as np
from ultralytics import YOLO

from langchain.agents import tool

# Marker ID positions layout
# 1 -------- 0
# |          |
# |          |
# 2 -------- 3

real_world_pts = np.array([
    [-317, 109],       # ID 1
    [-253, -635],      # ID 0
    [-907, -634],        # ID 3
    [-893, 55]        # ID 2
], dtype='float32')

@tool
def detect_table_objects(camera_port: int = 6) -> str:
    """
        Captures 10 frames from the camera, detects ArUco markers and objects using YOLOv8,
        then transforms object positions to real-world coordinates based on known marker positions.
        Use this to see what objects are in reach of the table and its position

        Returns:
            String of object class and real-world (x, y) coordinates.
            Returns an error message if detection fails.
    """
    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    expected_ids = {
        1: 'top_left',
        0: 'top_right',
        2: 'bottom_left',
        3: 'bottom_right'
    }
    corner_points = {}

    yolo = YOLO('yolov8n.onnx', task='detect')

    output_width, output_height = 800, 600
    detections = {}
    cap = cv2.VideoCapture(camera_port)

    for i in range(10): 
        ret, frame = cap.read()

        if not ret:
            continue

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is None or len(ids) < 4:
            continue

        ids = ids.flatten()
        found_ids = set(ids)

        if not all(k in found_ids for k in expected_ids):
            continue

        for i, id_ in enumerate(ids):
            if id_ in expected_ids:
                marker_corners = corners[i][0]
                center = marker_corners.mean(axis=0)
                corner_points[expected_ids[id_]] = center

        if len(corner_points) != 4:
            continue

        pts_src = np.array([
            corner_points['top_left'],
            corner_points['top_right'],
            corner_points['bottom_right'],
            corner_points['bottom_left']
        ], dtype='float32')

        pts_dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype='float32')

        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, matrix, (output_width, output_height))
        H_warp_to_real = cv2.getPerspectiveTransform(pts_dst, real_world_pts)

        results = yolo.track(warped, stream=True, verbose=False)

        for result in results:
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    pt = np.array([[[center_x, center_y]]], dtype='float32')
                    real_pt = cv2.perspectiveTransform(pt, H_warp_to_real)
                    real_x, real_y = real_pt[0][0]

                    class_name = result.names[int(box.cls[0])]
                    detections[class_name] = (real_x, real_y)
    cap.release()

    res = ""
    for item in detections:
        res += f'{item} at: {detections[item][0]:2f}mm, {detections[item][1]:2f}mm' 

    return res

@tool
def detect_table_markers(camera_port: int = 6) -> str:
    """
        Captures 10 frames from the camera, detects ArUco markers  hen transforms object 
        positions to real-world coordinates based on known marker positions.
        Use this to see what objects are in reach of the table and its position

        Returns:
            Aruco marker positions
            Returns an error message if detection fails.
    """
    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    expected_ids = {
        1: 'top_left',
        0: 'top_right',
        2: 'bottom_left',
        3: 'bottom_right'
    }
    corner_points = {}

    detection = ""

    output_width, output_height = 800, 600
    cap = cv2.VideoCapture(camera_port)

    for i in range(10): 
        ret, frame = cap.read()

        if not ret:
            continue

        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is None or len(ids) < 4:
            continue

        ids = ids.flatten()
        found_ids = set(ids)

        if not all(k in found_ids for k in expected_ids):
            continue

        for i, id_ in enumerate(ids):
            if id_ in expected_ids:
                marker_corners = corners[i][0]
                center = marker_corners.mean(axis=0)
                corner_points[expected_ids[id_]] = center

        if len(corner_points) != 4:
            continue

        pts_src = np.array([
            corner_points['top_left'],
            corner_points['top_right'],
            corner_points['bottom_right'],
            corner_points['bottom_left']
        ], dtype='float32')

        pts_dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype='float32')

        H_warp_to_real = cv2.getPerspectiveTransform(pts_src, real_world_pts)

        for i in [i for i in found_ids if i not in [0,1,2,3]]:
            pt = np.array([[[found_ids[i][0], found_ids[i][1]]]], dtype='float32')
            real_pt = cv2.perspectiveTransform(pt, H_warp_to_real)
            real_x, real_y = real_pt[0][0]

            detection += f'Aruco Marker {i} found at {real_x:.2f}mm, {real_y:.2f}mm. '

    cap.release()
    return detection 


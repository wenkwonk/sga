import os
import sys
import cv2
import json
import numpy as np
from ultralytics import YOLO

#pathing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.computer_vision.homography import CourtHomographyTransformer

def main(video_name=None):
    #fallback to default if no filename is supplied by the caller
    if video_name is None:
        video_name = "foul_sample_1.mp4"

    #pathing
    VIDEO_PATH = os.path.join(project_root, "data", "raw", video_name)
    PAINT_MODEL_PATH = os.path.join(project_root, "runs", "pose", "paint_detector", "weights", "best.pt")
    
    video_base_name = os.path.splitext(video_name)[0]
    json_filename = f"{video_base_name}_dynamic.json"
    JSON_OUTPUT_PATH = os.path.join(project_root, "data", "processed", "court_out", json_filename)
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(VIDEO_PATH) or not os.path.exists(PAINT_MODEL_PATH):
        print(f"Error: Missing assets! Verify paths:\nVideo: {VIDEO_PATH}\nModel: {PAINT_MODEL_PATH}")
        return False

    #init ai models
    player_model = YOLO("yolov8m-pose.pt")
    paint_model = YOLO(PAINT_MODEL_PATH)
    transformer = CourtHomographyTransformer()

    cap = cv2.VideoCapture(VIDEO_PATH)
    json_payload = []
    frame_idx = 0

    print("Running Headless Pipeline: Extracting coordinates, boxes, and skeletons...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        #model inference
        paint_res = paint_model.predict(source=frame, conf=0.25, device="mps", verbose=False)[0]
        player_res = player_model.track(source=frame, conf=0.4, classes=[0], persist=True, device="mps", verbose=False)[0]

        H = None
        detected_side = "unknown"
        saved_paint_corners = []

        #resolve court points
        if paint_res.keypoints is not None and len(paint_res.keypoints.xy) > 0:
            keypoints_list = paint_res.keypoints.xy[0].cpu().tolist()
            
            if len(keypoints_list) >= 5:
                scr_far_baseline = keypoints_list[0]
                scr_close_baseline = keypoints_list[1]
                scr_far_foul = keypoints_list[2]
                scr_close_foul = keypoints_list[3]
                top_of_key = keypoints_list[4]
                
                #save pixels for visualizer overlay
                saved_paint_corners = [[int(pt[0]), int(pt[1])] for pt in keypoints_list[:5]]
                
                foul_center_x = (scr_far_foul[0] + scr_close_foul[0]) / 2
                detected_side = "left" if top_of_key[0] > foul_center_x else "right"
                
                ordered_corners = [scr_far_baseline, scr_close_baseline, scr_far_foul, scr_close_foul]
                
                if all(pt[0] > 0 and pt[1] > 0 for pt in ordered_corners):
                    H = transformer.compute_matrix(ordered_corners, side=detected_side)

        #process players
        if player_res.boxes is not None and player_res.keypoints is not None:
            track_ids = player_res.boxes.id.int().cpu().tolist() if player_res.boxes.id is not None else [0] * len(player_res.boxes)
            keypoints_tensor = player_res.keypoints.xy.cpu().tolist()
            boxes_tensor = player_res.boxes.xyxy.cpu().tolist()

            for idx, player_id in enumerate(track_ids):
                kp = keypoints_tensor[idx]
                box = boxes_tensor[idx]
                
                foot_x = (kp[15][0] + kp[16][0]) / 2 if len(kp) > 16 else (box[0] + box[2]) / 2
                foot_y = (kp[15][1] + kp[16][1]) / 2 if len(kp) > 16 else box[3]

                clean_coords = [0.0, 0.0]
                if H is not None:
                    court_feet = transformer.pixel_to_feet([foot_x, foot_y], H)
                    if court_feet and transformer.is_within_bounds(court_feet):
                        while isinstance(court_feet, list) and len(court_feet) > 0 and isinstance(court_feet[0], list):
                            court_feet = court_feet[0]
                        clean_coords = [float(court_feet[0]), float(court_feet[1])]

                #format strictly to raw ints
                int_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                int_skeleton = [[int(pt[0]), int(pt[1])] for pt in kp]

                #save flat row to ledger
                json_payload.append({
                    "frame": frame_idx,
                    "player_id": player_id,
                    "court_coordinates_feet": clean_coords,
                    "court_side": detected_side,
                    "bbox": int_box,
                    "skeleton": int_skeleton,
                    "paint_corners": saved_paint_corners
                })

        frame_idx += 1

    cap.release()
    
    #write to file
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(json_payload, f, indent=4)
        
    print(f"Process Complete! Native model data written to: {JSON_OUTPUT_PATH}")
    return True

if __name__ == "__main__":
    main()
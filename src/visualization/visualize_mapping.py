import os
import sys
import cv2
import json
import numpy as np

#pathing
project_root = "/Users/USERNAME/Desktop/SGA"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#auto script call
from src.computer_vision.feature_extraction import main as run_headless_tracker

def generate_native_court():
    #court blueprint setup
    canvas_W = 940
    canvas_H = 500
    court = np.zeros((canvas_H, canvas_W, 3), dtype=np.uint8)
    court[:] = (42, 34, 30)

    line_color = (255, 255, 255)
    thick = 2

    #outer lines and midcourt
    cv2.rectangle(court, (0, 0), (canvas_W, canvas_H), line_color, thick)
    mid_x = int(canvas_W / 2)
    cv2.line(court, (mid_x, 0), (mid_x, canvas_H), line_color, thick)
    
    #center circle
    center_pt = (mid_x, int(canvas_H / 2))
    cv2.circle(court, center_pt, 60, line_color, thick)

    #left lane
    cv2.rectangle(court, (0, 250 - 80), (190, 250 + 80), line_color, thick)
    cv2.circle(court, (190, 250), 60, line_color, thick)
    
    #right lane
    cv2.rectangle(court, (canvas_W - 190, 250 - 80), (canvas_W, 250 + 80), line_color, thick)
    cv2.circle(court, (canvas_W - 190, 250), 60, line_color, thick)

    #hoops and rims
    cv2.line(court, (40, 250 - 30), (40, 250 + 30), line_color, 3)
    cv2.circle(court, (47, 250), 7, (0, 140, 255), 2)
    cv2.line(court, (canvas_W - 40, 250 - 30), (canvas_W - 40, 250 + 30), line_color, 3)
    cv2.circle(court, (canvas_W - 47, 250), 7, (0, 140, 255), 2)

    #arcs and lines
    cv2.line(court, (0, 30), (140, 30), line_color, thick)
    cv2.line(court, (0, canvas_H - 30), (140, canvas_H - 30), line_color, thick)
    cv2.ellipse(court, (47, 250), (238, 238), 0, -68, 68, line_color, thick)
    cv2.line(court, (canvas_W, 30), (canvas_W - 140, 30), line_color, thick)
    cv2.line(court, (canvas_W, canvas_H - 30), (canvas_W - 140, canvas_H - 30), line_color, thick)
    cv2.ellipse(court, (canvas_W - 47, 250), (238, 238), 0, 112, 248, line_color, thick)

    return court

#skeleton joints definition
SKELETON_EDGES = [
    (16, 14), (14, 12), (17, 15), (15, 13), (13, 11), (12, 11),
    (12, 6), (11, 5), (6, 5), (6, 8), (8, 10), (5, 7), (7, 9)
]

def draw_skeleton_sticks(frame, kp_list, player_id):
    #render skeleton bones and nodes
    color = ((player_id * 40) % 255, (player_id * 85) % 255, 235)
    if len(kp_list) == 0:
        return
        
    for start, end in SKELETON_EDGES:
        if start < len(kp_list) and end < len(kp_list):
            p1 = (int(kp_list[start][0]), int(kp_list[start][1]))
            p2 = (int(kp_list[end][0]), int(kp_list[end][1]))
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
                
    for pt in kp_list:
        px, py = int(pt[0]), int(pt[1])
        if px > 0 and py > 0:
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)

#main runtime variables
cap = None
court_blueprint_master = None
tracking_database = None
window_name = "SGA Spatial Audit Engine"
current_frame_idx = 0
is_playing = False  

def get_dashboard_image(val):
    global cap, court_blueprint_master, tracking_database, is_playing
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, video_frame = cap.read()
    if not ret or video_frame is None:
        return None

    overlay_video_frame = video_frame.copy()
    court_canvas = court_blueprint_master.copy()
    canvas_W, canvas_H = 940, 500

    #filter database rows for current frame
    current_frame_players = [row for row in tracking_database if row.get("frame") == val]
    detected_side = current_frame_players[0].get("court_side", "UNKNOWN") if len(current_frame_players) > 0 else "UNKNOWN"

    #draw paint lane boundaries on video side
    if len(current_frame_players) > 0:
        paint_corners = current_frame_players[0].get("paint_corners", [])
        if paint_corners and len(paint_corners) >= 5:
            #map coordinates safely
            p0 = (paint_corners[0][0], paint_corners[0][1])
            p1 = (paint_corners[1][0], paint_corners[1][1])
            p2 = (paint_corners[2][0], paint_corners[2][1])
            p3 = (paint_corners[3][0], paint_corners[3][1])
            p4 = (paint_corners[4][0], paint_corners[4][1])
            
            #draw rectangle lane lines
            cv2.line(overlay_video_frame, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(overlay_video_frame, p1, p3, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(overlay_video_frame, p3, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(overlay_video_frame, p2, p0, (0, 255, 255), 2, cv2.LINE_AA)
            
            #draw top of key triangle lines
            cv2.line(overlay_video_frame, p4, p2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(overlay_video_frame, p4, p3, (0, 255, 255), 2, cv2.LINE_AA)
            
            #draw corner track nodes
            node_colors = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0)]
            for idx, pt in enumerate(paint_corners):
                cv2.circle(overlay_video_frame, (pt[0], pt[1]), 5, node_colors[idx], -1)

    #process frame players
    for player in current_frame_players:
        player_id = player["player_id"]
        court_feet = player["court_coordinates_feet"]
        box = player.get("bbox")
        kp = player.get("skeleton")
        
        #overlay boxes and lines on video frame
        if box is not None:
            cv2.rectangle(overlay_video_frame, (box[0], box[1]), (box[2], box[3]), (235, 150, 4), 1)
        if kp is not None:
            draw_skeleton_sticks(overlay_video_frame, kp, player_id)

        #map to 2d court canvas
        if court_feet and court_feet != [0.0, 0.0]:
            canvas_X = int(court_feet[0] * 10)
            canvas_Y = int(court_feet[1] * 10)

            if 0 <= canvas_X < canvas_W and 0 <= canvas_Y < canvas_H:
                color = ((player_id * 40) % 255, (player_id * 85) % 255, 235)
                cv2.circle(court_canvas, (canvas_X, canvas_Y), 7, color, -1)
                cv2.circle(court_canvas, (canvas_X, canvas_Y), 8, (255, 255, 255), 1)
                cv2.putText(court_canvas, f"P{player_id}", (canvas_X + 12, canvas_Y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    #rescale into presentation dashboard layout
    video_resized = cv2.resize(overlay_video_frame, (640, 360))
    court_resized = cv2.resize(court_canvas, (640, 340))
    
    padded_court = np.zeros((360, 640, 3), dtype=np.uint8)
    padded_court[:] = (30, 34, 42)
    padded_court[10:350, :] = court_resized
    
    side_by_side_dashboard = np.hstack((video_resized, padded_court))
    
    status_text = "PLAYING" if is_playing else "PAUSED"
    status_color = (0, 255, 0) if is_playing else (0, 165, 255)
    
    cv2.rectangle(side_by_side_dashboard, (10, 10), (450, 45), (15, 18, 22), -1)
    cv2.putText(side_by_side_dashboard, f"SGA ENGINE | Frame: {val} | Side: {detected_side.upper()}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(side_by_side_dashboard, status_text, (340, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)

    return side_by_side_dashboard

def on_trackbar_change(val):
    global current_frame_idx, is_playing
    current_frame_idx = val
    if not is_playing:
        img = get_dashboard_image(current_frame_idx)
        if img is not None:
            cv2.imshow(window_name, img)

def main():
    global cap, court_blueprint_master, tracking_database, window_name, current_frame_idx, is_playing
    
    #video files config
    VIDEO_PATH = os.path.join(project_root, "data", "raw", "foul_sample_1.mp4")
    JSON_OUTPUT_PATH = os.path.join(project_root, "data", "processed", "court_out", "foul_sample_1_dynamic.json")

    #verify database matrix presence
    if not os.path.exists(JSON_OUTPUT_PATH):
        print("Tracking database file not found! Automatically calling headless AI tracker script...")
        success = run_headless_tracker()
        if not success or not os.path.exists(JSON_OUTPUT_PATH):
            print("Error: Headless tracking pipeline failed to generate file matrix.")
            return
            
    print("Tracking database located! Reading data matrices...")
    with open(JSON_OUTPUT_PATH, "r") as f:
        tracking_database = json.load(f)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    court_blueprint_master = generate_native_court()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Timeline", window_name, 0, total_frames - 1, on_trackbar_change)

    #timeline event loop
    while True:
        img = get_dashboard_image(current_frame_idx)
        if img is not None:
            cv2.imshow(window_name, img)
            
        if is_playing:
            cv2.setTrackbarPos("Timeline", window_name, current_frame_idx)

        delay = 1 if is_playing else 100
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):  
            is_playing = not is_playing  
            
        if is_playing:
            current_frame_idx += 1
            if current_frame_idx >= total_frames or current_frame_idx >= len(tracking_database):
                current_frame_idx = 0  

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
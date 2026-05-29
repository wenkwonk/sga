import numpy as np
import cv2

class CourtHomographyTransformer:
    def __init__(self):
        #court dimension setups
        self.paint_width_top = 17.0
        self.paint_width_bottom = 33.0
        self.paint_depth = 19.0
        self.total_court_length = 94.0
        self.total_court_width = 50.0

    def compute_matrix(self, screen_points, side="left"):
        #compute homography matrix from pixels to feet
        src_pts = np.array(screen_points, dtype=np.float32)
        
        if side == "left":
            #left side paint lane targets
            dst_pts = np.array([
                [0.0, self.paint_width_top],
                [0.0, self.paint_width_bottom],
                [self.paint_depth, self.paint_width_top],
                [self.paint_depth, self.paint_width_bottom]
            ], dtype=np.float32)
        else:
            #right side paint lane targets
            dst_pts = np.array([
                [self.total_court_length, self.paint_width_top],
                [self.total_court_length, self.paint_width_bottom],
                [self.total_court_length - self.paint_depth, self.paint_width_top],
                [self.total_court_length - self.paint_depth, self.paint_width_bottom]
            ], dtype=np.float32)
            
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def pixel_to_feet(self, pixel_coord, H):
        #warp pixel point to real world feet coordinates
        if H is None:
            return None
            
        pt = np.array([pixel_coord[0], pixel_coord[1], 1.0], dtype=np.float32).reshape(3, 1)
        transformed_pt = np.dot(H, pt)
        
        w = transformed_pt[2, 0]
        if abs(w) > 1e-5:
            x_feet = transformed_pt[0, 0] / w
            y_feet = transformed_pt[1, 0] / w
            return [x_feet, y_feet]
            
        return None

    def is_within_bounds(self, court_feet):
        #check if coordinates fall inside arena boundary buffer
        if court_feet is None:
            return False
            
        while isinstance(court_feet, list) and len(court_feet) > 0 and isinstance(court_feet[0], list):
            court_feet = court_feet[0]
            
        x, y = court_feet[0], court_feet[1]
        
        return (-5.0 <= x <= self.total_court_length + 5.0) and (-5.0 <= y <= self.total_court_width + 5.0)
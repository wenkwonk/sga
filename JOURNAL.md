# Development Journal

## May 27, 2026 - Repository Setup
* **[AD]** Created Repository and setup basic architecture
* **[AD&DK]** Researched general tech stack and ideas

## May 28, 2026 - YOLO Pose Tracking & Pipeline Setup
* **[AD]** Built a custom pipeline in `tracker.py` using YOLO11m-pose.pt to track player skeletons. Results from tracking are dumped off into a .json
* **[AD]** **Challenge:** Players blip in and out of detection when overlapping. Sidelines are crowded and introduce noise.
* **[AD]** Collected and annotated 60 data points (worst 2 hours of my life).
* **[AD]** **Architectural Decision:** Decided to anchor player positions based on the known dimensions of the key/paint, as outer court boundaries are too noisy. Trained a custom YOLO Pose model (48 train / 12 val split) on still broadcast frames. Labeled 5 structural paint points:
  * `0`: Far Baseline Corner
  * `1`: Close Baseline Corner
  * `2`: Far Foul Line Corner
  * `3`: Close Foul Line Corner
  * `4`: Top of the Key (Apex)
  * *Note:* Court orientation (left vs right side) is programmatically detected by checking the spatial position of point `4` relative to the remaining anchors.
* **[AD]** **Challenge:** Court left side paint tracking works fine, but right side point 1 seems to be completely wrong. It appears near the foul line instead of being close baseline.
* **[AD]** Successfully translated player coordinates onto a 2D plane using math in `homography.py`.
* **[AD]** Created a visualizer engine `visualize_mapping.py` to help visualize the results so far and help debug.
* **[AD]** **To-Do:** Fix right side paint-detection bug.
* **[AD]** **To-Do:** Tune player detection.
* **[DK]** Updated README.md

## May 29, 2026 - YOLO Pose Tracking & Pipeline Tuning
* **[AD]** Collected and annotated another 60 data points.
* **[AD]** Corrected right side paint detection bug from May 28. This was achieved by the collection of new data bringing our dataset to 120 data points (96 train 24 val) and retraining the custom paint detector
* **[AD]** Renamed `tracker.py` to `feature_extraction.py`
* **[AD]** Fully integrated `feature_extraction.py` into `visualize_mapping.py` so that `visualize_mapping.py` calls `feature_extraction.py` automatically when a .json is not already detected.
* **[AD]** Cleaned up `visualize_mapping.py` UI.
* **[AD]** Cleaned up repository by removing dead files.
* **[AD]** **To-Do:** Tune player detection. Players still blip in and out of detection when overlapping. Sidelines are still crowded and introduce noise. Might be time to start using Bytetrack.
* **[AD]** **To-Do:** Maybe add contracts?

## May 29, 2026 - Pipeline Cleaning
* **[AD]** Un-hardcoded video file pathing in `feature_extraction.py` and `visualize_mapping.py`.
* **[AD]** Un-hardcoded .json file naming from `feature_extraction.py` to make handoff more dynamic.

## June 3, 2026 - Player Tracking Model Change & Claude
* **[AD]** Changed player tracking model to botsort to improve player ID.
* **[AD]** **Challenge:** Tracking is still weak. Post-processing is likely the best way to move forward.
* **[AD]** Set up a CLAUDE.md based on Karpathy's template.
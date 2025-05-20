from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


# [Previous imports remain the same]

def main():
    # Read Video
    video_frames = read_video('/content/drive/MyDrive/Playground/theeye_cv/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('/content/drive/MyDrive/Playground/theeye_cv/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                     read_from_stub=True,
                                     stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Export speed and distance data to CSV
    speed_and_distance_estimator.export_to_csv(tracks)  # This is the correct way to call it

    # [Rest of your existing code remains the same]

if __name__ == '__main__':
    main()

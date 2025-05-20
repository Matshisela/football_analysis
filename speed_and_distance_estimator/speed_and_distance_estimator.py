import cv2
import sys 
import pandas as pd
import os
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
        self.total_distance = {}  # Track cumulative distance per player
        self.speed_data = {}     # Track all speed measurements per player
    
    def add_speed_and_distance_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Initialize storage if needed
                    if object not in self.total_distance:
                        self.total_distance[object] = {}
                        self.speed_data[object] = {}
                    
                    if track_id not in self.total_distance[object]:
                        self.total_distance[object][track_id] = 0
                        self.speed_data[object][track_id] = []
                    
                    # Update cumulative distance
                    self.total_distance[object][track_id] += distance_covered
                    self.speed_data[object][track_id].append(speed_km_per_hour)

                    # Update tracks with current values
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = self.total_distance[object][track_id]
    
    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        if speed is None or distance is None:
                            continue
                        
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            output_frames.append(frame)
        
        return output_frames
    
    def export_to_csv(self, tracks, output_folder='speed_distance_results'):
        """Export speed and distance statistics to CSV file"""
        os.makedirs(output_folder, exist_ok=True)
        
        data = []
        for object_type in self.total_distance.keys():
            for player_id in self.total_distance[object_type].keys():
                if player_id in self.speed_data[object_type]:
                    avg_speed = sum(self.speed_data[object_type][player_id]) / len(self.speed_data[object_type][player_id])
                    max_speed = max(self.speed_data[object_type][player_id])
                    total_dist = self.total_distance[object_type][player_id]
                    
                    # Try to get team information if available
                    team = "unknown"
                    for frame in tracks.get('players', []):
                        if player_id in frame and 'team' in frame[player_id]:
                            team = frame[player_id]['team']
                            break
                    
                    data.append({
                        'player_id': player_id,
                        #'object_type': object_type,
                        #'team': team,
                        'average_speed_kmh': avg_speed,
                        'max_speed_kmh': max_speed,
                        'total_distance_m': total_dist,
                        'num_speed_samples': len(self.speed_data[object_type][player_id])
                    })
        
        if data:
            df = pd.DataFrame(data)
            output_path = os.path.join(output_folder, 'speed_distance_stats.csv')
            df.to_csv(output_path, index=False)
            print(f"Successfully exported speed and distance data to {output_path}")
            return output_path
        else:
            print("No speed and distance data available for export")
            return None

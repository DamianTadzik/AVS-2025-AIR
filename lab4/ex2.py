import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_file_to_list(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        timestamps = []
        x_coords = []
        y_coords = []
        events = []

        for line in lines:
            spl = line.split()
            if len(spl) < 4:
                continue  # Skip malformed lines
            
            timestamp = float(spl[0])
            x = int(spl[1])
            y = int(spl[2])
            event = int(spl[3])

            timestamps.append(timestamp)
            x_coords.append(x)
            y_coords.append(y)
            events.append(event)

        return timestamps, x_coords, y_coords, events

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], [], []

def visualize_event_data(timestamps, x_coords, y_coords, events, title="3D Event Data Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Split data into positive and negative event lists
    pos_x, pos_y, pos_t = [], [], []
    neg_x, neg_y, neg_t = [], [], []
    
    for t, x, y, e in zip(timestamps, x_coords, y_coords, events):
        if e > 0:
            pos_x.append(x)
            pos_y.append(y)
            pos_t.append(t)
        else:
            neg_x.append(x)
            neg_y.append(y)
            neg_t.append(t)
    
    # Plot positive events in blue and negative events in red
    ax.scatter(pos_x, pos_y, pos_t, c='blue', label='Positive Events')
    ax.scatter(neg_x, neg_y, neg_t, c='red', label='Negative Events')
    
    # Labels and legend
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Timestamp")
    ax.set_title(title)
    ax.legend()
    ax.view_init(elev=20, azim=135)  # Adjust the angle for better visibility
    
    plt.show()
file_path = 'DAVIS/events.txt'
timestamps, x_coords, y_coords, events = read_file_to_list(file_path)

print(f"{len(timestamps)=}, {len(x_coords)=}, {len(y_coords)=}, {len(events)=}")

# 1.1: Visualize first 8000 events
visualize_event_data(timestamps[:8000], x_coords[:8000], y_coords[:8000], events[:8000], title="First 8000 Events")

# 1.2: Visualize events between timestamps 0.5 and 1
filtered_indices = [i for i, t in enumerate(timestamps) if 0.5 <= t <= 1]
filtered_timestamps = [timestamps[i] for i in filtered_indices]
filtered_x_coords = [x_coords[i] for i in filtered_indices]
filtered_y_coords = [y_coords[i] for i in filtered_indices]
filtered_events = [events[i] for i in filtered_indices]
visualize_event_data(filtered_timestamps, filtered_x_coords, filtered_y_coords, filtered_events, title="Events between 0.5 and 1s")

# 1. How long is the sequence used during exercise 1.1 (in seconds)?
#    - 1s lol
# 2. What’s the resolution of event timestamps?
#    - Hard to tell what the resolution is, the average can be calculated, the max also could be calculated - simple maths
# 3. What does the time difference between consecutive events depend on?
#    - It depends on the event sensor's sampling rate and the motion characteristics of objects in the scene.
# 4. What does positive/negative event polarity mean?
#    - Positive events correspond to brightness increases, while negative events correspond to brightness decreases.
# 5. What is the direction of movement of objects in exercise 1.2?
#    - The scatter plot shows that they are moving along the x-axis, towards positive values of it

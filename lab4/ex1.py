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

            if timestamp > 1:
                break

        return timestamps, x_coords, y_coords, events

    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], [], []

# Example usage
file_path = 'DAVIS/events.txt'
timestamps, x_coords, y_coords, events = read_file_to_list(file_path)
print(f"Number of elements: {len(timestamps)=}, {len(x_coords)=}, {len(y_coords)=}, {len(events)=}")
print(f"First and last: {timestamps[0]=}, {timestamps[-1]=}")
print(f"Coords: {max(x_coords)=}, {min(x_coords)=}, {max(y_coords)=}, {min(y_coords)=}")
print(f"Event sums: {sum([1 for x in events if x == 1])=}, {sum([-1 for x in events if x == 0])=}")
print(f"Event total sum: {(sum([1 for x in events if x == 1]) + sum([-1 for x in events if x == 0]))=}")

def visualize_event_data(timestamps, x_coords, y_coords, events):
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
    ax.set_title("3D Event Data Visualization")
    ax.legend()
    
    plt.show()

visualize_event_data(timestamps, x_coords, y_coords, events)

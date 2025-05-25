import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

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
            if timestamp < 1:
                continue
            x = int(spl[1])
            y = int(spl[2])
            event = int(spl[3])
            timestamps.append(timestamp)
            x_coords.append(x)
            y_coords.append(y)
            events.append(1 if event > 0 else -1)
            if timestamp > 2:
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
print(f"Event sums: {sum([1 for x in events if x == 1])=}, {sum([-1 for x in events if x == -1])=}")
print(f"Event total sum: {sum(events)=}")

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

# visualize_event_data(timestamps, x_coords, y_coords, events)

def event_frame(x_coords, y_coords, events, image_shape):
    image_representation = 127 * np.ones(image_shape, dtype=np.uint8)
    
    for x, y, event in zip(x_coords, y_coords, events):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:  # Ensure within bounds
            image_representation[y, x] = 255 if event > 0 else 0
    
    return image_representation

tau = 0.01 # 10 ms

t0 = None
start_id = None
end_id = None

for id, timestamp in enumerate(timestamps):
    if t0 is None:
        # Initialize t0 with the first timestamp
        t0 = timestamp
        start_id = id  # Save the first id
    else:
        if timestamp > t0 + tau:
            # Save the last id when time difference exceeds tau
            end_id = id

            # Slice the data using the saved indices (start_id to end_id)
            tmp_x_coords = x_coords[start_id:end_id]
            tmp_y_coords = y_coords[start_id:end_id]
            tmp_events = events[start_id:end_id]

            I = event_frame(tmp_x_coords, tmp_y_coords, tmp_events, (180, 240))
            cv2.imshow("I", I.astype('uint8'))
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
            # Reset t0 and start_id for the next slice
            t0 = timestamp
            start_id = id

# Tau changes the "trace of the objects"

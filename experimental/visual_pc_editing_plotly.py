import plotly.graph_objects as go
import numpy as np

# Generate random point clouds (replace with your own data)
point_cloud1 = np.random.rand(100, 3)  # Replace with your first point cloud data
point_cloud2 = np.random.rand(100, 3)  # Replace with your second point cloud data

# Create a 3D scatter plot for point_cloud2 (blue points)
fig = go.Figure(data=[go.Scatter3d(x=point_cloud2[:, 0], y=point_cloud2[:, 1], z=point_cloud2[:, 2], mode='markers', marker=dict(size=5, color='blue'))])

# Define variables to track the selection area
selection_start = None
selection_end = None

# Callback function for drawing the selection area
def draw_selection(event, points, selected):
    global selection_start, selection_end
    x, y, _ = points

    if event == 'click':
        if selection_start is None:
            selection_start = (x, y)
            selected.points = [go.Scatter3d(x=[x], y=[y], z=[0], mode='markers', marker=dict(size=5, color='red'))]
        else:
            selection_end = (x, y)
            selected.points = []
            selected_points = []

            for i, point in enumerate(point_cloud2):
                px, py, _ = point
                if selection_start[0] <= px <= selection_end[0] and selection_start[1] <= py <= selection_end[1]:
                    selected_points.append(i)

            # Highlight the selected points in green
            selected.points = go.Scatter3d(x=point_cloud2[selected_points, 0], y=point_cloud2[selected_points, 1], z=point_cloud2[selected_points, 2], mode='markers', marker=dict(size=5, color='green'))

# Convert fig.data to a list and add the new scatter trace
new_data = list(fig.data)
new_data.append(go.Scatter3d(x=point_cloud1[:, 0], y=point_cloud1[:, 1], z=point_cloud1[:, 2], mode='markers', marker=dict(size=5, color='red')))

# Assign the updated data list back to fig.data
fig.data = new_data
# Set the initial state of the Plotly scatter plot
selected_points = []
fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='markers', marker=dict(size=5, color='green')))

# Register the callback function for the scatter plot
fig.data[0].on_click(draw_selection)

# Show the interactive Plotly scatter plot
fig.show()

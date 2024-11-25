import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.cm import get_cmap
from matplotlib.markers import MarkerStyle
import ipywidgets as widgets
from IPython.display import display, FileLink
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import datetime

# File upload widget
file_upload = widgets.FileUpload(
    accept='.csv',  # Accept only CSV files
    multiple=False  # Allow only a single file upload
)
upload_button = widgets.Button(description="Process File")
download_button = widgets.Button(description="Download Image")
output = widgets.Output()

# Configuration panel widgets
eps_slider = widgets.FloatSlider(value=0.4, min=0.1, max=2.0, step=0.05, description="DBSCAN eps:")
min_samples_slider = widgets.IntSlider(value=5, min=1, max=20, step=1, description="Min Samples:")
marker_size_slider = widgets.IntSlider(value=200, min=50, max=500, step=10, description="Marker Size:")
color_map_dropdown = widgets.Dropdown(options=['tab10', 'viridis', 'plasma', 'inferno', 'coolwarm'],
                                      value='tab10', description="Color Map:")
save_path_text = widgets.Text(value='polar_plot_with_dbscan_clusters.png', description="Save Path:")

# Add dropdowns for selecting columns
timestamp_column_dropdown = widgets.Dropdown(description="Timestamp Column", options=[])
event_type_column_dropdown = widgets.Dropdown(description="Event Type Column", options=[])

# Display configuration panel and widgets
config_panel = widgets.VBox([
    eps_slider,
    min_samples_slider,
    marker_size_slider,
    color_map_dropdown,
    timestamp_column_dropdown,
    event_type_column_dropdown,
    save_path_text
])

# Display widgets
display(widgets.VBox([file_upload, upload_button, config_panel, download_button, output]))

# Placeholder for the file name of the saved plot
saved_plot_path = "polar_plot_with_dbscan_clusters.png"

def process_file(change):
    global saved_plot_path
    with output:
        output.clear_output()  # Clear previous output
        if not file_upload.value:
            print("Please upload a file first.")
            return
        
        file_content = file_upload.value[0]['content']
        
        # Create a temporary file to read the CSV
        with open('temp.csv', 'wb') as f:
            f.write(file_content)

        # Read the CSV file into a DataFrame
        data = pd.read_csv('temp.csv')

        # Remove the temporary file
        os.remove('temp.csv')

        print("File uploaded and processed successfully!")

        # Update dropdown options based on columns in the CSV
        timestamp_column_dropdown.options = data.columns.tolist()
        event_type_column_dropdown.options = data.columns.tolist()

        # Ensure the user selects the correct columns
        if not timestamp_column_dropdown.value or not event_type_column_dropdown.value:
            print("Please select both the Timestamp and Event Type columns.")
            return

        # Extract the selected columns
        Timestamps = data[timestamp_column_dropdown.value]
        EventTypes = data[event_type_column_dropdown.value]

        # Generate unique markers and colors for each Event type
        unique_Event_types = EventTypes.unique()
        cmap = plt.colormaps[color_map_dropdown.value]  # Using selected color map
        markers = ['o', 's', '^', 'D', 'P', '*', 'X']  # Predefined markers
        Event_styles = {Event: (markers[i % len(markers)], cmap(i / len(unique_Event_types))) for i, Event in enumerate(unique_Event_types)}

        # Prepare data for clustering (polar coordinates)
        polar_coordinates = []
        timestamps_parsed = []

        for t in range(len(Timestamps)):
            timestamp = Timestamps[t]
            try:
                # Try parsing the timestamp
                Date = timestamp.split(' ')[0]
                Time = timestamp.split(' ')[1]
                Day = int(Date.split('-')[-1])
                Hour = int(Time.split(':')[0]) + (int(Time.split(':')[1]) / 60)

                # Convert to polar coordinates (angle in radians and distance for day)
                angle = np.pi / 12 * Hour  # Hour to angle (radians)
                radial_distance = Day  # Day as the radial distance

                polar_coordinates.append([angle, radial_distance])
                timestamps_parsed.append(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))

            except ValueError as e:
                # Handle the invalid timestamp (e.g., skip or log)
                print(f"Skipping invalid timestamp: {timestamp} ({e})")
                continue

        if len(timestamps_parsed) == 0:
            print("No valid timestamps found. Please check your data.")
            return

        polar_coordinates = np.array(polar_coordinates)

        # Scale the data to improve clustering
        scaler = StandardScaler()
        scaled_polar_coordinates = scaler.fit_transform(polar_coordinates)

        # Apply DBSCAN clustering with fine-tuned parameters
        dbscan = DBSCAN(eps=eps_slider.value, min_samples=min_samples_slider.value)
        labels = dbscan.fit_predict(scaled_polar_coordinates)
        unique_labels = set(labels)

        print(f"Identified clusters: {unique_labels}")

        # Set up the polar plot
        fig = plt.figure(figsize=(30, 30))
        ax = fig.add_subplot(111, projection='polar')

        # Define the angles of the bold separators at 0, 90, 180, and 270 degrees
        separator_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

        # Set the angles and labels for the radial grid lines
        ax.set_xticks(np.arange(1, 25) * np.pi / 12)
        ax.set_xticklabels([str(number) for number in range(1, 25)])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Add bold separators at every 90 degrees and label each one (1 to 31)
        for angle in separator_angles:
            ax.axvline(angle, color='black', linewidth=2, linestyle='')
            # Label the separator line with 1-31 for each
            for day in range(1, 32):
                ax.text(angle, day, str(day), horizontalalignment='center', verticalalignment='center', fontsize=10)

        # Add points with unique markers and colors
        for t in range(len(Timestamps)):
            # Extract date, time, and Event type
            timestamp = Timestamps[t]
            Event_type = EventTypes[t]
            marker, color = Event_styles[Event_type]

            Date = timestamp.split(' ')[0]
            Time = timestamp.split(' ')[1]
            Day = int(Date.split('-')[-1])
            Hour = int(Time.split(':')[0]) + (int(Time.split(':')[1]) / 60)

            # Plot the point with specific marker and color
            ax.scatter(np.pi / 12 * Hour, Day, color=color, s=marker_size_slider.value, marker=MarkerStyle(marker), label=Event_type if t == 0 else "")

        # Plot red circles around clusters identified by DBSCAN
        for label in unique_labels:
            if label != -1:  # -1 is for noise points, not belonging to any cluster
                # Get the points belonging to this cluster
                cluster_points = polar_coordinates[labels == label]

                # Calculate the center of the cluster (mean position)
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_angle, cluster_radius = cluster_center

                # Plot the center of the cluster
                ax.plot(cluster_angle, cluster_radius, 'ro', markersize=10)  # Red point for cluster center
                # Draw a red circle around the cluster
                circle = plt.Circle((cluster_angle, cluster_radius), 1, color='red', fill=False, linewidth=3)  # Create a circle
                ax.add_patch(circle)

        # Add a legend
        handles = [plt.Line2D([0], [0], marker=MarkerStyle(Event_styles[Event][0]), color='w', 
                              markerfacecolor=Event_styles[Event][1], markersize=12) 
                   for Event in unique_Event_types]
        ax.legend(handles, unique_Event_types, title="Event Types", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 32)  # Set radial limit to 32 to cover 1-31 days

        # Explicitly set the radial grid lines (rings)
        ax.set_yticks(np.arange(1, 32))  # Show rings from 1 to 31
        ax.set_yticklabels([str(i) for i in range(1, 32)])  # Label the rings with day numbers

        # Ensure all grid lines are visible even if there are no points
        ax.grid(True, which='both', linestyle='-', linewidth=1.5, color='gray')  # Make grid lines visible

        # Save the plot to a file
        saved_plot_path = save_path_text.value
        fig.savefig(saved_plot_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)  # Close the figure to avoid display issues
        print(f"Plot saved as {saved_plot_path}")

def download_plot(change):
    with output:
        if not os.path.exists(saved_plot_path):
            print("No plot to download. Please generate the plot first.")
            return
        
        # Provide a download link
        display(FileLink(saved_plot_path, result_html_prefix="Click here to download the plot: "))

# Attach event handlers to the buttons
upload_button.on_click(process_file)
download_button.on_click(download_plot)

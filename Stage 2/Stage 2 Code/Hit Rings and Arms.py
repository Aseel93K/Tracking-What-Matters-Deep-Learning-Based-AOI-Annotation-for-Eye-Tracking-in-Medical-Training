from shapely.geometry import Point, Polygon
import json
import pandas as pd

# Your Excel file path
excel_file_path = r'C:\Users\aseel\OneDrive - The Pennsylvania State University\Desktop\Laparoscopic data\E1 B validation\E1 B.xlsx'

# Read fixation data from Excel
fixation_data = pd.read_excel(excel_file_path)

# Your JSON data with multiple sets of points
with open(
        r'C:\Users\aseel\OneDrive - The Pennsylvania State University\Desktop\Laparoscopic data\E1 B validation\result E1 B.json',
        'r') as file:
    json_data = json.load(file)

# List to store fixation counts
fixation_counts = []

# Iterate through images and data
for image in json_data:
    image_name = image["image_name"]
    # Initialize counts for the current image
    image_counts = {'ImageName': image_name, 'ArmHit': 0, 'RingHit': 0}

    for item in image["data"]:
        class_label = item["class_label"]
        for polygon_points in item["points"]:
            # Check if there are at least 3 coordinates in the "points" data
            if len(polygon_points) >= 3:
                # Create a Shapely Polygon from the points
                polygon = Polygon(polygon_points)

                # Count pegs and graspers for each fixation
                for _, fixation_info in fixation_data.iterrows():
                    fixation_location = Point(fixation_info['FixationPointX'], fixation_info['FixationPointY'])
                    frame_start = fixation_info['FrameStart']
                    frame_end = fixation_info['FrameEnd']

                    # Check if the current frame index is within the specified range
                    if frame_start <= int(image_name.split()[-1].split(".")[0]) <= frame_end:
                        # Check if fixation location is within the polygon
                        if fixation_location.within(polygon):
                            # Increment the corresponding count
                            if class_label == 'Arms':
                                image_counts['ArmHit'] += 1
                            elif class_label == 'Rings':
                                image_counts['RingHit'] += 1

    # Append counts for the current image to the list
    fixation_counts.append(image_counts)

# Create a DataFrame from the fixation counts list
fixation_counts_df = pd.DataFrame(fixation_counts)

# Save the counts to an Excel file
fixation_counts_df.to_excel('Excel.xlsx', index=False)

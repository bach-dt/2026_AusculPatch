import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import requests
import os
from urllib.parse import urlparse
import tempfile

import rtde_control

from langchain.agents import tool

# ur_ip = "192.168.56.101" # for simulator 
ur_ip = "192.168.0.100" # for physical robot

# Google Custom Search API Configuration - Replace with your own
GOOGLE_API_KEY = ""
GOOGLE_SEARCH_ENGINE_ID = ""


def _load_image(image_path):
    """Load and preprocess image for edge detection."""
    img_pil = Image.open(image_path)
    
    # Convert to RGB
    if img_pil.mode == 'RGBA':
        background = Image.new('RGB', img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[-1])
        img_pil = background
    elif img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    # Convert to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img, gray


def _search_image(query):
    """Search and download image using Google API."""
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE" or GOOGLE_SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE":
        raise Exception("Please set your Google API credentials at the top of the file")
    
    # Search for image
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_SEARCH_ENGINE_ID,
        'q': query,
        'searchType': 'image',
        'num': 1,
        'safe': 'high',
        'imgType': 'photo'
    }
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if 'items' not in data or len(data['items']) == 0:
        raise Exception("No images found for the query")
    
    # Download image
    image_url = data['items'][0]['link']
    img_response = requests.get(image_url, timeout=15)
    img_response.raise_for_status()
    
    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    file_ext = os.path.splitext(urlparse(image_url).path)[-1] or '.jpg'
    temp_path = os.path.join(temp_dir, f"search_image_{hash(query) % 10000}{file_ext}")
    
    with open(temp_path, 'wb') as f:
        f.write(img_response.content)
    
    return temp_path


def _transform_image(image, scale, rotation_degrees):
    """Apply scale and rotation to image."""
    if scale == 1.0 and rotation_degrees == 0:
        return image
    
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, -rotation_degrees, scale)
    
    # Calculate new dimensions
    cos_val = abs(matrix[0, 0])
    sin_val = abs(matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    # Adjust translation
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    
    return cv2.warpAffine(image, matrix, (new_width, new_height), 
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _extract_paths(gray_image, edge_low, edge_high, simplify_tolerance, min_area):
    """Extract and optimize robot paths from image."""
    # Edge detection
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, edge_low, edge_high)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert to paths
    paths = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            simplified = cv2.approxPolyDP(contour, simplify_tolerance, True)
            path = [(int(point[0][0]), int(point[0][1])) for point in simplified]
            if len(path) > 2:
                paths.append(path)
    
    # Optimize path order to minimize travel distance
    if len(paths) <= 1:
        return paths
    
    optimized = [paths[0]]
    remaining = paths[1:]
    
    while remaining:
        current_end = optimized[-1][-1]
        min_dist = float('inf')
        closest_idx = 0
        reverse = False
        
        for i, path in enumerate(remaining):
            dist_start = np.sqrt((current_end[0] - path[0][0])**2 + (current_end[1] - path[0][1])**2)
            dist_end = np.sqrt((current_end[0] - path[-1][0])**2 + (current_end[1] - path[-1][1])**2)
            
            if dist_start < min_dist:
                min_dist, closest_idx, reverse = dist_start, i, False
            if dist_end < min_dist:
                min_dist, closest_idx, reverse = dist_end, i, True
        
        closest_path = remaining.pop(closest_idx)
        optimized.append(closest_path[::-1] if reverse else closest_path)
    
    return optimized


def _plot_3d_path(continuous_path):
    """Plot robot path in 3D."""
    if not continuous_path:
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    coords = np.array(continuous_path)
    x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
    
    z_drawing = min(z_coords)
    z_travel = max(z_coords)
    
    # Plot path segments
    for i in range(len(continuous_path) - 1):
        x1, y1, z1 = continuous_path[i]
        x2, y2, z2 = continuous_path[i + 1]
        
        if z1 == z_drawing and z2 == z_drawing:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=2, alpha=1.0)
        else:
            ax.plot([x1, x2], [y1, y2], [z1, z2], 'r-', linewidth=1, alpha=0.7)
    
    # Mark start/end points
    ax.scatter(*continuous_path[0], color='green', s=150, marker='o', edgecolor='black')
    ax.scatter(*continuous_path[-1], color='orange', s=150, marker='s', edgecolor='black')
    
    # Mark pen transitions
    for i in range(1, len(continuous_path) - 1):
        x, y, z = continuous_path[i]
        prev_z, next_z = continuous_path[i-1][2], continuous_path[i+1][2]
        
        if prev_z == z_drawing and z == z_travel:  # Pen lift
            ax.scatter(x, y, z, color='purple', s=80, marker='^', alpha=0.8, edgecolor='black')
        elif prev_z == z_travel and z == z_drawing:  # Pen down
            ax.scatter(x, y, z, color='purple', s=80, marker='v', alpha=0.8, edgecolor='black')
    
    # Styling
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Robot Path\n(Blue: Drawing, Red: Travel)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Drawing'),
        Line2D([0], [0], color='red', linewidth=1, label='Travel'),
        Line2D([0], [0], marker='o', color='green', linewidth=0, markersize=8, label='Start'),
        Line2D([0], [0], marker='s', color='orange', linewidth=0, markersize=8, label='End'),
        Line2D([0], [0], marker='^', color='purple', linewidth=0, markersize=6, label='Pen up'),
        Line2D([0], [0], marker='v', color='purple', linewidth=0, markersize=6, label='Pen down')
    ]
    ax.legend(handles=legend_elements)
    
    # Stats
    total_points = len(continuous_path)
    drawing_points = sum(1 for _, _, z in continuous_path if z == z_drawing)
    stats = f"Points: {total_points} | Drawing: {drawing_points} | Travel: {total_points - drawing_points}"
    ax.text2D(0.02, 0.98, stats, transform=ax.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def get_robot_paths_from_image(image_path, scale=1.0, rotation_degrees=0, show_plot=False, 
                              z_height=0, edge_threshold_low=50, edge_threshold_high=150, 
                              simplify_tolerance=2.0, min_contour_area=100, lift_height=50):
    
    temp_file = None

    print("HERE,", image_path)
    
    try:
        # Get image (local file or search)
        if os.path.exists(image_path + ".jpg"):
            final_path = image_path + ".jpg"
        else:
            final_path = _search_image(image_path)
            temp_file = final_path

        print(final_path)
        
        # Load and process image
        img, gray = _load_image(final_path)
        
        # Apply transformations
        if scale != 1.0 or rotation_degrees != 0:
            img = _transform_image(img, scale, rotation_degrees)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract optimized paths
        paths = _extract_paths(gray, edge_threshold_low, edge_threshold_high, 
                              simplify_tolerance, min_contour_area)
        
        if not paths:
            return []
        
        # Convert to continuous path with Z-axis control
        continuous_path = []
        pen_up_height = z_height + lift_height
        
        for i, path in enumerate(paths):
            if i > 0:
                # Lift pen, move to new start, lower pen
                last_x, last_y = continuous_path[-1][:2]
                start_x, start_y = path[0]
                continuous_path.extend([
                    (last_x, last_y, pen_up_height),
                    (start_x, start_y, pen_up_height),
                    (start_x, start_y, z_height)
                ])
            else:
                # First path - start with pen down
                continuous_path.append((path[0][0], path[0][1], z_height))
            
            # Add remaining points in path
            for x, y in path[1:]:
                continuous_path.append((x, y, z_height))
        
        # Show plot if requested
        if show_plot:
            _plot_3d_path(continuous_path)
        
        return continuous_path
        
    except Exception as e:
        raise Exception(f"Error processing image: {e}")
        
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

@tool
def draw_image(image_name: str, start_x: float = -0.875, start_y: float = -0.174, 
               z_height: float = 100, plot: bool = False, scale: float = 1.0, 
               rotation_degrees: int = 180): 
    """
    Draws out the outline of a given object, actuating the robot arm along paths 
    with Z-axis control for pen lifting.
    
    Parameters:
    - image_name: Object name to draw, i.e. "Apple", "Orange"
    - start_x: Starting x position of the drawing
    - start_y: Starting y position of the drawing
    - z_height: Z coordinate when drawing (pen down), default = 0.05
    - scale: Scale factor (1.0 = original size)
    - rotation_degrees: Rotation angle in degrees (positive = clockwise)
    - plot: Whether to display 3D visualization
    
    Returns:
        Success message or error message
    """
    
    # image_name += " outline drawing"
    
    conversion_factor = 0.001 / 2
    velocity = 0.1
    acceleration = 0.3
    blend = 0.001

    tcp_rotation = [-0.001, 3.12, 0.04]
    movement_info = [velocity, acceleration, blend]
    
    print("Generating Path")
    try: 
        path = get_robot_paths_from_image(image_name, z_height=z_height, show_plot=plot, scale=1.0, rotation_degrees=rotation_degrees)
    except Exception as e: 
        print("Error", e)

    print("Path Generated")

    robot_path = []
    for p in path: 
        position = [p[0] * conversion_factor + start_x, p[1] * conversion_factor + start_y, p[2] * conversion_factor]
        robot_path.append(position + tcp_rotation + movement_info)


    try:
        rtde_c = rtde_control.RTDEControlInterface(ur_ip)
    except:
        return "Error connecting to UR Robot"
    
    rtde_c.moveL(robot_path)
    rtde_c.stopScript()

    return f'Drawing of {image_name} successfully drawn!'

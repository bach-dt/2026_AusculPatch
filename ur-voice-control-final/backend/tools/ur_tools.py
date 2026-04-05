import rtde_control
import rtde_receive
import serial
import time
import matplotlib.pyplot as plt

from HersheyFonts import HersheyFonts
from langchain.agents import tool

# ur_ip = "192.168.56.101" # for simulator 
ur_ip = "192.168.0.100" # for physical robot

@tool 
def move_tcp_direct(x: float, y: float, z: float) -> str:
  """
    Moves the robot TCP to the mentioned cartesian position,

    Args: 
      x (float): Target x-coordinate (meters).
      y (float): Target y-coordinate (meters).
      z (float): Target z-coordinate (meters).
    
    Returns: 
      Service response or error message.
  """

  try:
    rtde_c = rtde_control.RTDEControlInterface(ur_ip)
  except:
    return "Error connecting to UR Robot"
  
  rtde_c.moveL([x, y, z, -0.001, 3.12, 0.04], 0.5, 0.3)

  return f'Moved TCP to {x}, {y}, {z}' + '\n'

@tool 
def move_till_contact() -> str:
  """
    Moves the robot TCP until it comes into contact with an object or table surface
    
    Returns: 
      Service response or error message.
  """

  try:
    rtde_c = rtde_control.RTDEControlInterface(ur_ip)
  except:
    return "Error connecting to UR Robot"
  
  speed = [0, 0, -0.075, 0, 0, 0]
  rtde_c.moveUntilContact(speed)

  return "Moved TCP until contact\n"

@tool
def get_pose() -> str:
  """
    Retrieves the current end effector pose of the robot in Cartesian coordinates.

    Returns: 
      Current pose as a formatted string or an error message.
  """

  rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip) # check if connected 
  actual_tcp_pose = rtde_r.getActualTCPPose()

  pose_str = (
    f"Position: ({actual_tcp_pose[0]:.4f}, "
    f"{actual_tcp_pose[1]:.4f}, "
    f"{actual_tcp_pose[2]:.4f}), "
    f"Orientation: ({actual_tcp_pose[3]:.4f}, "
    f"{actual_tcp_pose[4]:.4f}, "
    f"{actual_tcp_pose[5]:.4f})"
  )

  return f'Current pose: {pose_str}' + '\n'

@tool
def get_joint_positions() -> str:
  """
    Retrieves the current joint positions of the UR5e robot.

    Returns: 
      Joint states as a formatted string or an error message.
  """

  rtde_r = rtde_receive.RTDEReceiveInterface(ur_ip)
  actual_q = rtde_r.getActualQ()

  pose_str = (
    f"Position: ({actual_q[0]:.4f}, "
    f"{actual_q[1]:.4f}, "
    f"{actual_q[2]:.4f}), "
    f"Orientation: ({actual_q[3]:.4f}, "
    f"{actual_q[4]:.4f}, "
    f"{actual_q[5]:.4f})"
  )

  return f'Current pose: {pose_str}' + '\n'

@tool 
def open_gripper() -> str: 
  """
    Opens the gripper attached to the robotics arm

    Returns: 
      Confirmation of gripper opening or an error message.
  """

  ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
  time.sleep(2)

  ser.write(b'1')

  ser.close()

  return "Gripper Opened\n"

@tool 
def close_gripper() -> str: 
  """
    Close the gripper attached to the robotics arm

    Returns: 
      Confirmation of gripper closing or an error message.
  """

  ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
  time.sleep(2)

  ser.write(b'0')

  ser.close()

  return "Gripper Closed\n"

@tool
def write_word(word: str, start_x: float = -0.144, start_y: float = -0.436, z_height: float = 0.05, plot:bool = False) -> str:
  """
    Moves the TCP to trace out a word given by the word parameter.
  
    Args: 
      word (string): Word for the robot to write out
      start_x (float): Starting x position for the text. Default is -0.144m
      start_y (float): Starting y position for the text. Default is -0.436m 
      plot (bool): Plot the path of the robot's TCP (for debugging). Default is False 
  
    Returns:
      Success message or Error message
  """
  font = HersheyFonts()
  font.load_default_font()
  font.normalize_rendering(50)

  velocity = 0.1
  acceleration = 0.3
  blend = 0.001
  scale = 0.001 # from mm 

  path = []
  for (x1, y1), (x2, y2) in font.lines_for_text(word):
    x1 = x1 * scale + start_x 
    x2 = x2 * scale + start_x 
    y1 = y1 * scale + start_y 
    y2 = y2 * scale + start_y 

    tcp_rotation = [-0.001, 3.12, 0.04]
    movement_info = [velocity, acceleration, blend]

    point_start = [x1, y1, z_height] + tcp_rotation + movement_info
    point_end = [x2, y2, z_height] + tcp_rotation + movement_info

    if len(path) > 0 and (path[-1][0], path[-1][1]) != (point_start[0], point_start[1]):
      point_space = [path[-1][0], path[-1][1], z_height + 0.1] + tcp_rotation + movement_info
      path.append(point_space)
    
    path.extend([point_start, point_end])

  try:
    rtde_c = rtde_control.RTDEControlInterface(ur_ip)
  except:
    return "Error connecting to UR Robot"
  
  rtde_c.moveL(path)
  rtde_c.stopScript()

  if plot:
    plot_path(path)

  return f'Word {word} successfully written down!'

def plot_path(path: list[float]):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  for i in range(len(path) - 1):
    ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], [path[i][2], path[i + 1][2]], colour='blue') 
  
  plt.grid(True)
  plt.axis("equal")
  plt.show()


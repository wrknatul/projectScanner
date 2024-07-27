import math
import cv2
import numpy as np

def equation_of_line(x1, y1, angle_of_slope):
    """
    Function that returns the equation of a line in the form y = kx + b, 
    given the coordinates of one point (x1, y1) and the angle of slope.
    
    Args:
        x1: The x-coordinate of the point on the line.
        y1: The y-coordinate of the point on the line.
        angle_of_slope: The angle of slope of the line in degrees.
    
    Returns:
        A tuple (k, b), where k is the slope and b is the y-intercept.
    """
    
    # Convert the angle from degrees to radians
    angle_of_slope_in_radians = math.radians(angle_of_slope)
    # Calculate the slope (k)
    k = math.tan(angle_of_slope_in_radians)
    
    # Calculate the y-intercept (b)
    b = y1 - k * x1
  
    return (k, b)

def compute_line_length_in_rectangle(line_points, x, y, width, height):
  """
  Computes the length of a line segment that is inside a rectangle.

  Args:
    line_points: A list of two tuples (x1, y1) and (x2, y2) representing the start and end points of the line.
    rectangle: A tuple (x, y, width, height) representing the coordinates and dimensions of the rectangle.

  Returns:
    The length of the line segment inside the rectangle.
  """


  # Get line segment coordinates
  x1, y1 = line_points[0]
  x2, y2 = line_points[1]
  A = y1 - y2
  B = x2 - x1
  C = -(A*x1 + B*y1)
  if np.abs(A*x + B*y + C)/np.sqrt(A*A + B*B) > width + height:
    return 0
  # Calculate intersection points with rectangle edges
  intersection_points = []
  # Check for intersections with left and right edges
  if B != 0:  # Avoid division by zero
    for edge_x in [x, x + width]:
      # Calculate y-coordinate of intersection point
        y_intersection = -(C + A*edge_x)/B
        # Check if intersection point is within rectangle bounds
        if y <= y_intersection <= y + height:
          intersection_points.append((edge_x, y_intersection))
  
  if len(intersection_points) == 2:
    x1_in, y1_in = intersection_points[-1]
    x2_in, y2_in = intersection_points[-2]
    line_length = math.sqrt((x2_in - x1_in) ** 2 + (y2_in - y1_in) ** 2)
    return line_length
  # Check for intersections with top and bottom edges
  if A != 0:  # Avoid division by zero
    for edge_y in [y, y + height]:
      # Calculate x-coordinate of intersection point
        x_intersection = -(C + B*edge_y)/A
        # Check if intersection point is within rectangle bounds
        if x <= x_intersection <= x + width:
          intersection_points.append((x_intersection, edge_y))

  # Find the intersection points that define the line segment inside the rectangle
  if len(intersection_points) >= 2:
    # Calculate the distance between the intersection points
    x1_in, y1_in = intersection_points[-1]
    x2_in, y2_in = intersection_points[-2]
    line_length = math.sqrt((x2_in - x1_in) ** 2 + (y2_in - y1_in) ** 2)
    return line_length
  else:
    # No intersection points found, so the line is not inside the rectangle
    return 0


def main():
    image_path = "data/goal_photo.jpg"
    bw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("data/converted_photo.jpg", bw_image)
    width, height = bw_image.shape[0], bw_image.shape[1]
    A = []
    b = []
    for angle_of_slope in range(2):
        new_col = []
        number = 0
        x1, y1 = width/2, height/2
        x2 = x1 + 1
        y2 = y1 + math.tan(math.radians(angle_of_slope))
        for i in range(width):
            for j in range(height):
                koef = compute_line_length_in_rectangle([(x1, y1), (x2, y2)], i, j, 1, 1)
                new_col.append(koef)
                number += koef*bw_image[i][j]
        A.append(new_col)
        b.append(number)
    X = np.linalg.pinv(A) @ b
    outputImage = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            outputImage[i][j] = X[i*height + j]
    cv2.imwrite("data/result_image.jpg", outputImage) 

if __name__ == "__main__":
    main()
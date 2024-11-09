import math
import cv2
import numpy as np
import time 
import argparse

def find_parallel_points(x1, y1, x2, y2, distance):

    # Находим угловой коэффициент k
    k = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    # Нормальный вектор
    if k != float('inf'):
        norm_vector = (-k, 1)
    else:
        norm_vector = (1, 0)

    # Длина нормализованного вектора
    length = math.sqrt(norm_vector[0]**2 + norm_vector[1]**2)

    # Нормализуем вектор
    unit_vector = (norm_vector[0] / length, norm_vector[1] / length)

    # Находим новые точки
    A_prime = (x1 - distance * unit_vector[0], y1 - distance * unit_vector[1])
    B_prime = (x2 - distance * unit_vector[0], y2 - distance * unit_vector[1])

    return A_prime, B_prime

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

def line_rectangle_intersection(x1, y1, x2, y2, rect_left_x, rect_bottom_y, rect_right_x, rect_top_y):
  """
  Finds the intersection points of a line with a rectangle.

  Args:
    x1: The x-coordinate of the starting point of the line.
    y1: The y-coordinate of the starting point of the line.
    x2: The x-coordinate of the ending point of the line.
    y2: The y-coordinate of the ending point of the line.
    rect_left_x: The x-coordinate of the rectangle's left x point.
    rect_bottom_y: The y-coordinate of the rectangle's less y point.
    rect_right_x: The x-coordinate of the rectangle's right x point.
    rect_top_y: The y-coordinate of the rectangle's top y point.

  Returns:
    A list of tuples representing the coordinates of the intersection points.
  """
  # Calculate the line's slope and y-intercept
  slope = (y2 - y1) / (x2 - x1) if x1 != x2 else float('inf')
  intercept = y1 - slope * x1 if x1 != x2 else x1

  # Find intersection points with each side of the rectangle
  intersection_points = []
  for x in [rect_left_x, rect_right_x]:
    y = slope * x + intercept if x1 != x2 else y1
    if rect_bottom_y <= y <= rect_top_y:
      intersection_points.append((x, y))
  if len(intersection_points) == 2:
    return intersection_points
  for y in [rect_bottom_y, rect_top_y]:
    x = (y - intercept) / slope if x1 != x2 else x1
    if rect_left_x <= x <= rect_right_x:
      intersection_points.append((x, y))

  return intersection_points[-2:]


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

def can_go(line_points, x, y, width, height):
  return compute_line_length_in_rectangle(line_points, x, y, 1, 1) > 0. and x >= 0 and x < width and y >= 0 and y < height

def bresenham_line_float_subsquares(x1, y1, x2, y2, width, height):
    """
    Draws a line using Bresenham's algorithm for float points and
    identifies unit squares intersected by the line.

    Args:
        x1: The x-coordinate of the starting point.
        y1: The y-coordinate of the starting point.
        x2: The x-coordinate of the ending point.
        y2: The y-coordinate of the ending point.

    Returns:
        A list of tuples representing the coordinates of the unit squares
        intersected by the line.
    """
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    # Create a list to store the intersected unit squares
    intersected_squares = []
    steps = 0
    ix = -1
    iy = -1
    for i in range(width):
      if compute_line_length_in_rectangle([(x1, y1), (x2, y2)], i, 0, 1, 1) > 0.:
        ix = i
        iy = 0
        if sy == -1:
          sx *= -1
          sy *= -1 
        break
      if compute_line_length_in_rectangle([(x1, y1), (x2, y2)], i, height - 1, 1, 1) > 0.:
        ix = i
        iy = height - 1
        if sy == 1:
          sx *= -1
          sy *= -1 
        break
    if ix == -1:
      for j in range(height):
        if compute_line_length_in_rectangle([(x1, y1), (x2, y2)], 0, j, 1, 1) > 0.:
          ix = 0
          iy = j
          if sx == -1:
            sy *= -1
            sx *= -1 
          break
        if compute_line_length_in_rectangle([(x1, y1), (x2, y2)], width - 1, j, 1, 1) > 0.:
          ix = width - 1
          iy = j
          if sx == 1:
            sy *= -1
            sx *= -1 
          break
    if ix == -1:
      return []
    # Iterate until we reach the end point
    while True:
        # Add the unit square to the intersected list
        intersected_squares.append((ix, iy))
        if can_go([(x1, y1), (x2, y2)], ix + sx, iy, width, height):
          ix = ix + sx
          iy = iy
          continue
        if can_go([(x1, y1), (x2, y2)], ix, iy + sy, width, height):
          ix = ix
          iy = iy + sy
          continue
        if can_go([(x1, y1), (x2, y2)], ix + sx, iy + sy, width, height):
          ix = ix + sx
          iy = iy + sy
          continue
        break
          
        

    return intersected_squares

def parse_args():
    parser = argparse.ArgumentParser(description='Image processing with scanners')
    parser.add_argument('--input_image', type=str, default='data/goal_photo.jpg',
                      help='Path to input image')
    parser.add_argument('--output_image', type=str, default='data/result_image.jpg',
                      help='Path to save result image')
    parser.add_argument('--bresenham', type=int, default=1,
                      help='Use Bresenham algorithm (1) or not (0)')
    parser.add_argument('--num_scanners', type=int, default=41,
                      help='Number of scanners')
    parser.add_argument('--distance_between_scanners', type=int, default=6,
                      help='Distance between scanners')
    return parser.parse_args()

def main():
    args = parse_args()
    bw_image = cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2GRAY)
    width, height = bw_image.shape[0], bw_image.shape[1]
    A = []
    b = []
    for angle_of_slope in range(360):
        angle_of_slope *= 1
        for scanner in range(args.num_scanners):
            if scanner % 2 == 0:
                distance = (scanner + 1)//2
            else:
                distance = -(scanner + 1)//2
            distance *= args.distance_between_scanners

            new_col = []
            number = 0.
            x1, y1 = width/2, height/2
            x2 = x1 + 1
            y2 = y1 + math.tan(math.radians(angle_of_slope))
            (x1, y1), (x2, y2) = find_parallel_points(x1, y1, x2, y2, distance)
            if args.bresenham == 0:
              for i in range(width):
                for j in range(height):
                  koef = compute_line_length_in_rectangle([(x1, y1), (x2, y2)], i, j, 1, 1)
                  new_col.append(koef)
                  number += koef*bw_image[i][j]
              A.append(new_col)
              b.append(number)
            else:
              sp_points = bresenham_line_float_subsquares(x1, y1, x2, y2, width, height)
              new_col = [0]*width*height
              for point in sp_points:
                x, y = point
                if 0 <= x and x < width and 0 <= y and y < height:
                  new_col[x*height + y] = 1
                  number += bw_image[x][y]
              A.append(new_col)
              b.append(number)
    X = np.linalg.pinv(A) @ b
    outputImage = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            outputImage[i][j] = X[i*height + j]
    cv2.imwrite(args.output_image, outputImage) 

if __name__ == "__main__":
    main()
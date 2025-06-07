import math
import cv2
import numpy as np
import time 
import argparse

cached_probabilities = None

def find_parallel_points(x1, y1, x2, y2, distance):

    k = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')

    if k != float('inf'):
        norm_vector = (-k, 1)
    else:
        norm_vector = (1, 0)

    length = math.sqrt(norm_vector[0]**2 + norm_vector[1]**2)

    unit_vector = (norm_vector[0] / length, norm_vector[1] / length)

    A_prime = (x1 - distance * unit_vector[0], y1 - distance * unit_vector[1])
    B_prime = (x2 - distance * unit_vector[0], y2 - distance * unit_vector[1])

    return A_prime, B_prime

def equation_of_line(x1, y1, angle_of_slope):
    
    angle_of_slope_in_radians = math.radians(angle_of_slope)
    k = math.tan(angle_of_slope_in_radians)
  
    b = y1 - k * x1
  
    return (k, b)

def line_rectangle_intersection(x1, y1, x2, y2, rect_left_x, rect_bottom_y, rect_right_x, rect_top_y):
  slope = (y2 - y1) / (x2 - x1) if x1 != x2 else float('inf')
  intercept = y1 - slope * x1 if x1 != x2 else x1

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
  x1, y1 = line_points[0]
  x2, y2 = line_points[1]
  A = y1 - y2
  B = x2 - x1
  C = -(A*x1 + B*y1)
  if np.abs(A*x + B*y + C)/np.sqrt(A*A + B*B) > width + height:
    return 0
  intersection_points = []
  if B != 0:
    for edge_x in [x, x + width]:
        y_intersection = -(C + A*edge_x)/B
        if y <= y_intersection <= y + height:
          intersection_points.append((edge_x, y_intersection))
  
  if len(intersection_points) == 2:
    x1_in, y1_in = intersection_points[-1]
    x2_in, y2_in = intersection_points[-2]
    line_length = math.sqrt((x2_in - x1_in) ** 2 + (y2_in - y1_in) ** 2)
    return line_length
  if A != 0:
    for edge_y in [y, y + height]:
        x_intersection = -(C + B*edge_y)/A
        if x <= x_intersection <= x + width:
          intersection_points.append((x_intersection, edge_y))

  if len(intersection_points) >= 2:
    x1_in, y1_in = intersection_points[-1]
    x2_in, y2_in = intersection_points[-2]
    line_length = math.sqrt((x2_in - x1_in) ** 2 + (y2_in - y1_in) ** 2)
    return line_length
  else:
    return 0

def can_go(line_points, x, y, width, height):
  return compute_line_length_in_rectangle(line_points, x, y, 1, 1) > 0. and x >= 0 and x < width and y >= 0 and y < height

def bresenham_line_float_subsquares(x1, y1, x2, y2, width, height):
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
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
    while True:
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

def cicle(it, size, A, v):
    return it % size

def symART(i, n, A, v):
    i = i % (2*n - 1)

    if i <= n - 1:
        return i
    else:
        return 2*n - 2 - i

def evenART(it, size, A, v):
    it %= len(A)
    h = (size + 1) // 2
    if it % 2 == 0:
        return (it // 2) + 1
    else:
        return h + (it // 2)

def proportional_solve(it, n, A, v):
    global cached_probabilities
    if cached_probabilities is None:  
      cached_probabilities = [np.linalg.norm(row) for row in A]
      cached_probabilities = cached_probabilities / np.sum(cached_probabilities)

    return np.random.choice(len(A), p=cached_probabilities)

def kaczmarz_solve(A, b, type_, max_iter=10000):
    A = np.array(A)
    n, m = A.shape
    x = np.zeros(m)
    if type_ == "cicle":
        j = cicle
    elif type_ == "symART":
        j = symART
    elif type_ == "evenART":
        j = evenART
    else:
       j = proportional_solve

    for _ in range(max_iter):
        i = j(_, n, A, x)
        ai = A[i, :]
        bi = b[i]
        x = x + (bi - np.dot(ai, x)) / np.dot(ai, ai) * ai

    return x

def parse_args():
    parser = argparse.ArgumentParser(description='Image processing with scanners')
    parser.add_argument('--input_image', type=str, default='data/goal_photo.jpg',
                      help='Path to input image')
    parser.add_argument('--output_image', type=str, default='data/result_image.jpg',
                      help='Path to save result image')
    parser.add_argument('--bresenham', type=int, default=1,
                      help='Use Bresenham algorithm (1) or not (0)')
    parser.add_argument('--kaczmarz', type=int, default=0,
                      help='Use Kaczmarz algorithm (1) or not (0)')
    parser.add_argument('--function_type', type=str, default="cicle",
                      help='Use different type of function for Kaczmarz algorithm: "cicle" or "symART" or "evenART" or "probabilities_algorithm"')
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
    for angle_of_slope in range(180):
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

    if args.kaczmarz == 1:
        X = kaczmarz_solve(A, b, args.function_type)
    else:
        X = np.linalg.pinv(A) @ b

    outputImage = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            outputImage[i][j] = X[i*height + j]
    cv2.imwrite(args.output_image, outputImage) 

if __name__ == "__main__":
    main()

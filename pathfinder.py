#!/usr/bin/python3

import numpy as np
import cv2
import math
import random
import time
import itertools
import sys
import threading

NUM_POINTS = 100
X_MAX = 500
Y_MAX = 500

def print_2Dtable(table):
    print('\n'.join([''.join([' | {} '.format(item) for item in row]) for row in table]))

def PathLength(path):
    total = 0
    lastPoint = None
    for p in path:
        if lastPoint is None:
            lastPoint = p
            continue
        total += lastPoint.distance(p)
        lastPoint = p
    return total

class WorkerThread(threading.Thread):

    def __init__(self, threadID, name, points, s, t, path_func, path_array):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.points = points
        self.s = s 
        self.t = t
        self.path_func = path_func
        self.path_array = path_array


    def run(self):
        print("Starting " + self.name)
        path = self.path_func(self.points, self.s, self.t)
        self.path_array[self.threadID] = path
        print("Exiting " + self.name)


class Point:

    def __init__(self, Xcoord, Ycoord, ID=0):
        self.Xcoord = Xcoord
        self.Ycoord = Ycoord
        self.ID = ID
        self.pathTo = []


    def __repr__(self):
        return "Point:" + str(self.ID)


    def __str__(self):
        return "(" + str(self.getX()) + ", " + str(self.getY()) + ")"


    def getPathTo(self):
        return self.pathTo


    def setPathTo(self, path):
        self.pathTo = path


    def getX(self):
        return self.Xcoord


    def getY(self):
        return self.Ycoord


    def getPathLength(self):
        return PathLength(self.getPathTo() + [self])


    def addToPath(self, point):
        if point in self.pathTo:
            print("Already in path!")
            return
        self.pathTo.append(point)


    def distance(self, point):
        """manhattan distance"""
        xDist = math.fabs(self.getX() - point.getX())
        yDist = math.fabs(self.getY() - point.getY())
        return xDist + yDist


def generate_points():
    """Generate points randomly in plane"""
    points = []

    for i in range(NUM_POINTS):
        points.append(Point(random.randrange(X_MAX), random.randrange(Y_MAX), i))

    return np.array(points)

def brute_force(points, s, t):
    permutations = list(itertools.permutations(points))
    min_path = None
    min_path_length = float("inf")
    for path in permutations:
        path_length = PathLength([s] + list(path) + [t])
        
        if min_path_length > path_length:
            min_path = list(path)
            min_path_length = path_length

    return [s] + min_path + [t]
    

def initialize_path_table(points, s):
    path_table = []
    col_one = []
    for p in points[1:-1]:
        col_one.append((p.distance(s),[s, p]))
    path_table.append(col_one)
    return path_table

def build_table(points, s):
    print("Progress: 0%", end=" \r")

    path_table = initialize_path_table(points, s)

    for col in range(1, NUM_POINTS - 2):
        last_col = path_table[col - 1]
        this_col = []

        for row in range(1, NUM_POINTS - 1):
            this_point = points[row]
            best_distance = float("inf")
            best_path = None

            for distPathPair in last_col:
                path = distPathPair[1]
                
                if path is None or this_point in path:
                    continue

                last_point = path[-1]
                distance = distPathPair[0] + last_point.distance(this_point)

                if distance <= best_distance:
                    best_distance = distance
                    best_path = path + [this_point]

            this_col.append((best_distance, best_path))

        path_table.append(this_col)

        print("Progress: {0:.2f}%".format(100 * col / NUM_POINTS), end=" \r")

    return path_table


def find_path(points, s, t):
    path_table = build_table(points, s)
    
    best_distance = float("inf")
    best_path = None

    for distPathPair in path_table[-1]:
        if distPathPair[0] == float("inf"):
            continue
        path = distPathPair[1]

        dist = distPathPair[0] + path[-1].distance(t)
        if dist < best_distance:
            best_path = path

    return best_path + [t]

def nearest_neighbor(points, s, t):
    unvisited = points[1:]
    path = [s]
    progress = 0

    while unvisited:
        print("Progress: {0:.2f}%".format(progress / NUM_POINTS), end=" \r")
        last_point = path[-1]

        best_point_idx = 0
        best_point = unvisited[0]
        best_distance = last_point.distance(best_point)

        for i, p in enumerate(unvisited):
            this_point = p
            this_distance = last_point.distance(this_point) 

            if this_distance < best_distance:
                best_point_idx = i
                best_point = this_point
                best_distance = this_distance

        unvisited.pop(best_point_idx)
        path.append(best_point)
        progress += 1

    return path

def draw_image(path):
    img = np.zeros((X_MAX,Y_MAX,3), np.uint8)

    lastPoint = None

    for p in path:
        if lastPoint is None:
            img = cv2.circle(img,(p.getY(),p.getX()), 1, (0,0,255), -1)
            lastPoint = p
            continue

        img = cv2.circle(img,(p.getY(),p.getX()), 1, (0,0,255), -1)
        img = cv2.line(img,(lastPoint.getY(),lastPoint.getX()),(p.getY(),p.getX()),(255,0,0),1)
        lastPoint = p

    return img


def display_path(path, image_name="image"):
    img = draw_image(path)
    cv2.namedWindow(image_name)
    cv2.moveWindow(image_name, 20, 20)
    cv2.imshow(image_name,img)

    while cv2.getWindowProperty(image_name, 0) >= 0:
        keycode = cv2.waitKey(50)

        if keycode == 27:
            print('ESC')
            break   
    cv2.destroyAllWindows()

def usage():
    print("pathfinder [NUM_POINTS] [X_MAX] [Y_MAX]")

def parse_arguments():
    global NUM_POINTS, X_MAX, Y_MAX

    try:
        if len(sys.argv) == 2:
            NUM_POINTS = int(sys.argv[1])
        elif len(sys.argv) == 4:
            NUM_POINTS = int(sys.argv[1])
            X_MAX = int(sys.argv[2])
            Y_MAX = int(sys.argv[3])
        return 0
    except:
        usage()
        return -1

def main():
    if parse_arguments() != 0:
        return -1

    print("NUM_POINTS: {} X_MAX: {} Y_MAX: {}".format(NUM_POINTS, X_MAX, Y_MAX))

    points = generate_points()

    s = points[0]
    t = points[NUM_POINTS - 1]

    path_array = [0,1]

    start = time.perf_counter()

    fasto_way_thread = WorkerThread(0, "fast-o way", points, s, t, find_path, path_array)
    fasto_way_thread.start()
    fasto_way_thread.join()

    end = time.perf_counter()

    display_path(path_array[0], "fast-o way")

    print("fast-o way: " + str(PathLength(path_array[0])))
    print("time: " + str(end-start))

    return 0

if __name__ == '__main__':
    main()

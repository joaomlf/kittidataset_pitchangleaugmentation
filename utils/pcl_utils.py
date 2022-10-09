import os, time
import numpy as np
import torch
import cv2
import pptk
import mayavi.mlab

import math
import random
from statistics import median

import utils.config as Cfg
import utils.kitti_utils as Ku


def process_pcl(number, pcls_list):

    # Format number
    number_format = format(number, '06d')

    # Auxiliary variable and lists
    obj_count_pcl = 0
    obj_height_list_aux = []
    theta_list_aux = []

    start_time = time.time()

    # Set Data paths
    lidar_path = 'data/KITTI/object/training/velodyne/' + number_format + '.bin'
    calib_path = 'data/KITTI/object/training/calib/' + number_format + '.txt'
    label_path = 'data/KITTI/object/training/label_2/' + number_format + '.txt'

    # Load Data
    pcl, Tr_Velo2Cam, labels = load_data(lidar_path, calib_path, label_path)

    # Filter PCL
    pcl_filtered, index_pcl = filter_pcl(pcl)

    # Get PCL's points in Camera coordinate system
    pcl_cam, pcl_filtered_r = get_pcl_in_cam_coord(pcl_filtered, Tr_Velo2Cam)

    labels_new = []
    pcl_new = pcl

    # Process PCL's objects
    for label in labels:
        # Get Object's data
        obj = Ku.Object3d(label)

        # Is the Object a...?
        if number in pcls_list and obj.cls_id == 0: # car+van
        # if number in pcls_list and obj.cls_id == 2: # cyclist
        # if number in pcls_list and ( obj.cls_id == 0 or obj.cls_id == 2 ): # all classes
            # Get Object's centroid (in Camera coordinate system)
            centroid_cam = np.array(obj.t)

            # Get centroid in Velodyne coordinate system
            centroid_velo = get_centroid_coord(centroid_cam, Tr_Velo2Cam, 'velo')

            # Define Object's bounding box in LiDAR coordinate system (WITHOUT yaw)
            bbox_velo = get_bbox_coord(centroid_velo, obj, 'velo')

            # Is the Object inside PCL's boundaries?
            if test_object(bbox_velo):
                # Define Object's bounding box in Camera coordinate system
                bbox_cam = get_bbox_coord(centroid_cam, obj, 'cam') # (WITHOUT yaw)

                # Apply Object's YAW angle to defined bbox in Camera coordinate system
                bbox_cam_yaw = bbox_yaw(obj.ry, centroid_cam, bbox_cam) # (WITH yaw)

                # Get PCL's points contained in Object's bounding box (in Camera coordinate system)
                object_cam, index_obj = get_object_points(pcl_cam, bbox_cam_yaw)

                # Is the Object having less than 10 points?
                if len(index_obj) <= 10:
                    label = label + ' 0.00'
                    labels_new.append(label)
                    continue

                else: object_cam = object_cam[0:3, :] # (4,N) --> (3,N)

                ## Get Object's points in Velodyne coordinate system
                ## object_velo = get_points_coord(object_cam, Tr_Velo2Cam, 'velo')

                # Generate pitch angle
                # theta_deg = random.randint(-30, 30) # experiment 1
                theta_deg = random.randint(-5, 5) # experiment 2

                # Fill auxiliary variable and lists
                obj_count_pcl += 1
                obj_height_list_aux.append(obj.h)
                theta_list_aux.append(theta_deg)

                if theta_deg == 0:
                    label = label + ' 0.00'
                    labels_new.append(label)
                    continue

                else:
                    theta = np.radians( theta_deg ) # Pitch angle (in degrees) given in radians

                    # Apply pitch rotation to Object's points and centroid (in Camera coordinate system)
                    object_cam_pitch, bbox_cam_pitch, centroid_cam_pitch = rotations_cam_coord(obj.ry, theta, centroid_cam, bbox_cam_yaw, object_cam)

                    # object_cam1 = object_cam.T # (3,N) -> (N,3)
                    object_cam_pitch1 = object_cam_pitch.T # (3,N) -> (N,3)

                    # Save NEW points in FILTERED PCL (in Camera coordinate system)
                    pcl_cam = change_pcl_cam(pcl_cam, object_cam_pitch1, index_obj)

                    # Get NEW PCL (in Velodyne coordinate system)
                    pcl_filtered_new = get_pcl_in_velo_coord(pcl_cam, pcl_filtered_r, Tr_Velo2Cam)

                    # Save NEW points in ORIGINAL PCL (in Velodyne coordinate system)
                    pcl_new = change_pcl(pcl_new, pcl_filtered_new, index_pcl)

                    ## Get NEW centroid coordinates (in Camera coordinate system)
                    ## centroid_new = new_centroid(object_cam_pitch)
                    ## labels_new = new_pcl_label(labels_new, obj, centroid_new, theta)

                    # Get Object's NEW label
                    labels_new = new_pcl_label(labels_new, obj, centroid_cam_pitch, theta)

                    ## --- VISUALIZATION ---
                    # if len(index_obj) >= 300:
                    if len(index_obj) >= 2500:

                        if obj.cls_id == 0:
                            print("Object class: Car or Van")
                        elif obj.cls_id == 2:
                            print("Object class: Cyclist")

                        print("len(index_obj) = ", len(index_obj))
                        print("Theta [deg] = ", theta_deg)

                        # Get object points (in Velodyne coordinate system)
                        bbox_velo_yaw = get_points_coord(bbox_cam_yaw, Tr_Velo2Cam, 'velo')
                        object_velo_new = get_points_coord(object_cam, Tr_Velo2Cam, 'velo')
                        object_velo_pitch = get_points_coord(object_cam_pitch, Tr_Velo2Cam, 'velo')
                        bbox_velo_pitch = get_points_coord(bbox_cam_pitch, Tr_Velo2Cam, 'velo')
                        object_velo1 = object_velo_new.T # (3,N) -> (N,3)
                        object_velo_pitch1 = object_velo_pitch.T # (3,N) -> (N,3)

                        # Visualize NEW rotated object and its bbox in PCL
                        visualize(pcl_filtered_new, object_velo1, object_velo_pitch1, bbox_velo_yaw, bbox_velo_pitch)
                        ## visualize(pcl_filtered, object_velo1, bbox_velo, bbox_velo_yaw)

            else:
                label = label + ' 0.00'
                labels_new.append(label)

        else:
            label = label + ' 0.00'
            labels_new.append(label)

    # Save NEW PCL data to file
    # save_new_pcl(pcl_new, number_format)

    # Save NEW LABEL data to file
    # save_new_labels(labels_new, number_format)

    end_time = round( time.time() - start_time, 2)
    # print("PCL processing time = ", end_time)

    return obj_count_pcl, obj_height_list_aux, theta_list_aux


def load_data(lidar_path, calib_path, label_path):
    # Load PCL
    pcl = np.fromfile( str(lidar_path), dtype=np.float32, count=-1).reshape([-1,4] )

    # Load Calibration data
    with open(calib_path) as f:
        lines = f.readlines()
    #R0_rect, Tr_Velo2Cam = PCLu.get_calib_data(lines)
    Tr_Velo2Cam = get_calib_data(lines)

    # Load Objects' labels
    labels = [line.rstrip() for line in open(label_path)]

    return pcl, Tr_Velo2Cam, labels


def get_calib_data(lines):
    # Obtain 'R0 rectified' matrix
    #line = lines[4].strip().split(' ')[1:]
    #R0_rect = np.array(line, dtype=np.float32)
    #R0_rect = np.reshape(R0_rect, (3,3))
    #R0_rect = np.hstack(( R0_rect, np.zeros((3,1)) ))
    #R0_rect = np.vstack(( R0_rect, np.array([0,0,0,1]) ))

    # Obtain 'Tr Velodyne to Camera' matrix
    line = lines[5].strip().split(' ')[1:]
    Tr_Velo2Cam = np.array(line, dtype=np.float32)
    Tr_Velo2Cam = np.reshape(Tr_Velo2Cam, (3,4)) # (1,12) --> (3,4)
    Tr_Velo2Cam = np.vstack((Tr_Velo2Cam, np.array([0,0,0,1]))) # (3,4) -->  (4,4)

    return Tr_Velo2Cam


def filter_pcl(pcl):
    X_min, X_max, Y_min, Y_max, Z_min, Z_max = get_pcl_boundaries()
    mask = np.where((pcl[:, 0] >= X_min) & (pcl[:, 0] <= X_max) &
                    (pcl[:, 1] >= Y_min) & (pcl[:, 1] <= Y_max) &
                    (pcl[:, 2] >= Z_min) & (pcl[:, 2] <= Z_max))
    pcl_filtered = pcl[mask]

    index_pcl = list( mask[0] )

    return pcl_filtered, index_pcl


def get_pcl_boundaries():
    X_min = Cfg.boundary['minX']
    X_max = Cfg.boundary['maxX']
    Y_min = Cfg.boundary['minY']
    Y_max = Cfg.boundary['maxY']
    Z_min = Cfg.boundary['minZ']
    Z_max = Cfg.boundary['maxZ']

    return X_min, X_max, Y_min, Y_max, Z_min, Z_max


def get_pcl_in_cam_coord(pcl, Tr_Velo2Cam):
    # Get PCL reflectance values
    pcl_r = np.array( pcl[:, 3] )
    pcl_r = np.reshape(pcl_r, (len(pcl_r),1))

    # Get PCL points' coordinates
    pcl_cam = pcl[:, 0:3]

    # Get homogeneous matrix for matrix multiplication
    pcl_cam = np.hstack(( pcl_cam, np.ones(( len(pcl_cam),1)) )).T # (N,3) -> (4,N)

    # Get PCL's points in Camera coordinate system
    pcl_cam = np.matmul(Tr_Velo2Cam, pcl_cam)
    #pcl_cam = np.matmul(R0_rect, pcl_cam)
    pcl_cam = pcl_cam[0:3, :].T # (4,N) -> (N,3)

    return pcl_cam, pcl_r


def get_centroid_coord(centroid, Tr_Velo2Cam, coord):
    centroid = np.reshape( centroid, (1,3) )
    centroid = np.hstack(( centroid, np.ones((1,1)) )).T

    # Obtain other matrices
    #R0_rect_inv = np.linalg.inv(R0_rect)
    Tr_Cam2Velo = np.linalg.inv(Tr_Velo2Cam)

    if coord == 'cam':
        #centroid_cam = np.matmul(R0_rect, centroid)
        centroid_cam = centroid[0:3]
        return centroid_cam

    elif coord == 'velo':
        #centroid_velo = np.matmul(R0_rect_inv, centroid)
        centroid_velo = np.matmul(Tr_Cam2Velo, centroid)
        centroid_velo = centroid_velo[0:3]
        return centroid_velo


def get_bbox_coord(centroid, obj, coord):
    '''
    Object:     |   Camera:     |   Velodyne:
        x ^     |     ^ z       |   y ^
          |     |     |         |     |
    y <---o z   |   y X---> x   |   z o ---> x

        0 -------- 1
       /|         /|
      2 -------- 3 .
      | |        | |
      . 4 -------- 5
      |/         |/
      6 -------- 7
    '''
    if coord == 'cam':
        xmin = float(centroid[0] - obj.l / 2)
        xmax = float(centroid[0] + obj.l / 2)
        ymin = float(centroid[1] - obj.h)
        ymax = float(centroid[1])
        zmin = float(centroid[2] - obj.w /2 )
        zmax = float(centroid[2] + obj.w /2 )

        point_0 = np.array((xmin, ymin, zmax))
        point_1 = np.array((xmax, ymin, zmax))
        point_2 = np.array((xmin, ymin, zmin))
        point_3 = np.array((xmax, ymin, zmin))
        point_4 = np.array((xmin, ymax, zmax))
        point_5 = np.array((xmax, ymax, zmax))
        point_6 = np.array((xmin, ymax, zmin))
        point_7 = np.array((xmax, ymax, zmin))

    elif coord == 'velo':
        xmin = float(centroid[0] - obj.l / 2)
        xmax = float(centroid[0] + obj.l / 2)
        ymin = float(centroid[1] - obj.w / 2)
        ymax = float(centroid[1] + obj.w / 2)
        zmin = float(centroid[2])
        zmax = float(centroid[2] + obj.h)

        point_0 = np.array((xmax, ymax, zmax))
        point_1 = np.array((xmax, ymin, zmax))
        point_2 = np.array((xmin, ymax, zmax))
        point_3 = np.array((xmin, ymin, zmax))
        point_4 = np.array((xmax, ymax, zmin))
        point_5 = np.array((xmax, ymin, zmin))
        point_6 = np.array((xmin, ymax, zmin))
        point_7 = np.array((xmin, ymin, zmin))

    return np.column_stack((point_0, point_1, point_2, point_3,
                            point_4, point_5, point_6, point_7))


def bbox_yaw(psi, centroid, bbox):
    centroid = np.reshape( centroid, (3,1) )

    # Get bbox coordinates in Object coordinate system
    bbox_obj = bbox - centroid

    # Get homogeneous matrix and translation matrix for matricial product
    bbox_obj = np.vstack(( bbox_obj, np.ones(( 1, len(bbox[0]) )) ))
    TrCam2Obj = np.hstack(( Ku.roty(psi), np.zeros((3,1)) ))
    TrCam2Obj = np.vstack(( TrCam2Obj, np.array([0,0,0,1]) ))

    # Add yaw rotation to bbox
    bbox_yaw = np.matmul( TrCam2Obj, bbox_obj )
    bbox_yaw = bbox_yaw[0:3, :]

    # Get bbox coordinates back in Camera coordinate system
    bbox_yaw = bbox_yaw + centroid

    return bbox_yaw


def test_object(bbox):
    X_min, X_max, Y_min, Y_max, Z_min, Z_max = get_pcl_boundaries()

    if ((bbox[0].min() >= X_min) & (bbox[0].max() <= X_max) &
        (bbox[1].min() >= Y_min) & (bbox[1].max() <= Y_max) &
        (bbox[2].min() >= Z_min) & (bbox[2].max() <= Z_max)):
        return True
    else:
        return False


def get_object_points(pcl, bbox):
    object = []
    index_obj = []
    idx = 0

    # Avoid selecting points belonging to the ground
    ymax = bbox[1].max() - 0.1

    for point in pcl:
        # Point between bbox Y-axis' limits?
        if (point[1] >= bbox[1].min()) & (point[1] <= ymax):
            # Point belongs to object?
            if test_point(point, bbox):
                object.append(point)
                index_obj.append(idx)
        # Increment PCL's point index
        idx += 1

    object = np.array(object).T # (N,4) --> (4,N)

    return object, index_obj


def test_point(point, bbox):
    point = np.reshape( point, (3,1) )

    # Point contained in Bbox's limits?
    if (point[0] >= bbox[0].min()) & (point[0] <= bbox[0].max()) & (point[2] >= bbox[2].min()) & (point[2] <= bbox[2].max()):
        return True
    else:
        return False


def get_points_coord(points, Tr_Velo2Cam, coord):
    # Get homogeneous matrix for matricial product
    points = np.vstack(( points, np.ones(len(points[0])) ))

    if coord == 'cam':
        points_cam = np.matmul(Tr_Velo2Cam, points)
        points_cam = points_cam[0:3]
        return points_cam

    elif coord == 'velo':
        Tr_Cam2Velo = np.linalg.inv(Tr_Velo2Cam)
        points_velo = np.matmul(Tr_Cam2Velo, points)
        points_velo = points_velo[0:3]
        return points_velo


def rotations_cam_coord(psi, theta, centroid, bbox, points):
    centroid_cam = np.reshape(centroid, (3,1))

    # Translation matrices
    Tr_Obj2Cam = np.hstack(( Ku.roty(psi), centroid_cam ))
    Tr_Obj2Cam = np.vstack(( Tr_Obj2Cam, np.array([0,0,0,1]) ))

    Tr_Cam2Obj = np.linalg.inv(Tr_Obj2Cam)

    # Get homogeneous matrices: (3,N) -> (4,N)
    points_h = np.vstack(( points, np.ones(len(points[0])) ))
    bbox_h = np.vstack(( bbox, np.ones(len(bbox[0])) ))

    # Get points' coordinates in Object coordinate system
    points_obj = np.matmul(Tr_Cam2Obj, points_h)
    bbox_obj = np.matmul(Tr_Cam2Obj, bbox_h)

    # Apply pitch rotation to object and its bbox
    Tr_Obj2Rot = np.hstack(( Ku.rotz(theta).T, np.zeros((3,1)) ))
    Tr_Obj2Rot = np.vstack(( Tr_Obj2Rot, np.array([0,0,0,1]) ))
    points_obj_rot = np.matmul(Tr_Obj2Rot, points_obj)
    bbox_obj_rot = np.matmul(Tr_Obj2Rot, bbox_obj)

    # Get NEW points' coordinates back in Camera's coordinate system
    points_cam_rot = np.matmul(Tr_Obj2Cam, points_obj_rot)
    points_cam_rot = points_cam_rot[0:3] # (4,N) -> (3,N)

    bbox_cam_rot = np.matmul(Tr_Obj2Cam, bbox_obj_rot)
    bbox_cam_rot = bbox_cam_rot[0:3] # (4,N) -> (3,N)

    # Apply translation to avoid points below real ground
    y_thres = bbox[1].max() - bbox_cam_rot[1].max()
    # y_thres_points = points[1].max() - points_cam_rot[1].max()
    # points_cam_rot[1,:] += y_thres_points
    points_cam_rot[1,:] += y_thres

    # y_thres_bbox = bbox[1].max() - bbox_cam_rot[1].max()
    # bbox_cam_rot[1,:] += y_thres_bbox
    bbox_cam_rot[1,:] += y_thres

    centroid_rot = centroid.copy()
    # centroid_rot[1] += y_thres_bbox
    centroid_rot[1] += y_thres

    return points_cam_rot, bbox_cam_rot, centroid_rot


def change_pcl_cam(pcl, object, index_obj):
    pcl[index_obj] = object

    return pcl


def get_pcl_in_velo_coord(pcl, pcl_r, Tr_Velo2Cam):
    # Obtain other matrices
    #R0_rect_inv = np.linalg.inv(R0_rect)
    Tr_Cam2Velo = np.linalg.inv(Tr_Velo2Cam)

    # Get homogeneous matrix for matricial product
    pcl = np.hstack(( pcl, np.ones(( len(pcl),1)) )).T # (N,3) -> (4,N)

    # Get PCL's points in Velodyne coordinate system
    #pcl_velo = np.matmul(R0_rect_inv, pcl)
    pcl_velo = np.matmul(Tr_Cam2Velo, pcl)
    pcl_velo = pcl_velo[0:3, :].T # (4,N) -> (N,3)

    # Add reflectance values
    pcl_velo = np.hstack(( pcl_velo, pcl_r )) # (N,3) -> (N,4)

    return pcl_velo


def change_pcl(pcl, pcl_filtered, index_pcl):
    pcl[index_pcl] = pcl_filtered

    return pcl


# def new_centroid(object_rot):
# #def new_centroid(bbox_velo_rotyz):
#     ''' As seen in https://stackoverflow.com/questions/77936/whats-the-best-way-to-calculate-a-3d-or-n-d-centroid '''
#     x = object_rot[0,:]
#     y = object_rot[1,:]
#     z = object_rot[2,:]
#
#     # We should (not?) ignore outliers
#     #x_new = round( np.mean(x, axis=0), 2 )
#     #y_new = round( np.mean(y, axis=0), 2 )
#     #z_new = round( np.mean(z, axis=0), 2 )
#
#     # 30/01/2022 - centroid should be the MIDDLE and not the MEAN of all points (more info in thesis' report)
#     x_new = round( (x.max() + x.min())/2 , 2 )
#     y_new = round( (y.max() + y.min())/2 , 2 )
#     z_new = round( (z.max() + z.min())/2 , 2 )
#
#     centroid = [x_new, y_new, z_new]
#
#     return centroid


def new_pcl_label(labels_new, object, centroid, theta):
    kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                % (object.type, object.truncation, int(object.occlusion), object.alpha, object.box2d[0], object.box2d[1],
                   object.box2d[2], object.box2d[3], object.h, object.w, object.l, centroid[0], centroid[1], centroid[2],
                   object.ry, theta)

    labels_new.append(kitti_str)

    return labels_new


def visualize(pcl, object, object_pitch, bbox_yaw, bbox_pitch):
    # PCL after pitch
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    plot_pcl(fig, pcl)
    # draw_bbox(fig, bbox_yaw, 'yaw')
    draw_bbox(fig, bbox_pitch, 'pitch')

    # Object before pitch
    fig1 = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    plot_object(fig1, object)
    draw_bbox(fig1, bbox_yaw, 'yaw')

    # Object after pitch
    fig2 = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    plot_object(fig2, object_pitch)
    draw_bbox(fig2, bbox_yaw, 'yaw')
    draw_bbox(fig2, bbox_pitch, 'pitch')

    mayavi.mlab.show()


def plot_pcl(fig, pcl):
    x = pcl[:, 0]
    y = pcl[:, 1]
    z = pcl[:, 2]
    #r = pcl[:, 3]  # reflectance value of point
    # Map Distance from sensor
    d = np.sqrt(x ** 2 + y ** 2)

    mayavi.mlab.points3d(x, y, z,
                         d,          # Values used for Color
                         mode='point',
                         colormap='gnuplot',
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig)
    # Draw origin
    mayavi.mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

    return fig


def plot_object(fig, object):
    x = object[:, 0]
    y = object[:, 1]
    z = object[:, 2]
    #r = pcl[:, 3]  # reflectance value of point
    # Map Distance from sensor
    d = np.sqrt(x ** 2 + y ** 2)

    mayavi.mlab.points3d(x, y, z,
                         d,          # Values used for Color
                         mode='point',
                         colormap='Blues',
                         figure=fig)

    return fig


def draw_bbox(fig, bbox, rot):
    # Which color to draw?
    if rot == 'yaw': aux = (0,1,0) # green
    elif rot == 'pitch': aux = (1,0,0) # red
    else: aux = (1,1,1) # white

    bbox_top_x = [ bbox[0,0], bbox[0,1], bbox[0,3], bbox[0,2], bbox[0,0] ]
    bbox_top_y = [ bbox[1,0], bbox[1,1], bbox[1,3], bbox[1,2], bbox[1,0] ]
    bbox_top_z = [ bbox[2,0], bbox[2,1], bbox[2,3], bbox[2,2], bbox[2,0] ]
    mayavi.mlab.plot3d(bbox_top_x, bbox_top_y, bbox_top_z,
                       color = aux,
                       line_width = 0.25,
                       figure = fig)

    bbox_bottom_x = [ bbox[0,4], bbox[0,5], bbox[0,7], bbox[0,6], bbox[0,4] ]
    bbox_bottom_y = [ bbox[1,4], bbox[1,5], bbox[1,7], bbox[1,6], bbox[1,4] ]
    bbox_bottom_z = [ bbox[2,4], bbox[2,5], bbox[2,7], bbox[2,6], bbox[2,4] ]
    mayavi.mlab.plot3d(bbox_bottom_x, bbox_bottom_y, bbox_bottom_z,
                       color = aux,
                       line_width = 0.25,
                       figure = fig)

    bbox_front_x = [ bbox[0,0], bbox[0,1], bbox[0,5], bbox[0,4], bbox[0,0] ]
    bbox_front_y = [ bbox[1,0], bbox[1,1], bbox[1,5], bbox[1,4], bbox[1,0] ]
    bbox_front_z = [ bbox[2,0], bbox[2,1], bbox[2,5], bbox[2,4], bbox[2,0] ]
    mayavi.mlab.plot3d(bbox_front_x, bbox_front_y, bbox_front_z,
                       color = aux,
                       line_width = 0.25,
                       figure = fig)

    bbox_back_x = [ bbox[0,2], bbox[0,3], bbox[0,7], bbox[0,6], bbox[0,2] ]
    bbox_back_y = [ bbox[1,2], bbox[1,3], bbox[1,7], bbox[1,6], bbox[1,2] ]
    bbox_back_z = [ bbox[2,2], bbox[2,3], bbox[2,7], bbox[2,6], bbox[2,2] ]
    mayavi.mlab.plot3d(bbox_back_x, bbox_back_y, bbox_back_z,
                       color = aux,
                       line_width = 0.25,
                       figure = fig)

    return fig


def save_new_labels(labels_new, number):
    label_path_new = 'data_new/experiment-2/all_classes/label_2/' + number + '.txt'
    label_new_file = open( label_path_new, "wt")
    for label_new in labels_new: label_new_file.write(label_new + '\n')
    label_new_file.close()


def save_new_pcl(pcl_new, number):
    pcl_path_new = 'data_new/experiment-2/all_classes/velodyne/' + number + '.bin'
    pcl_new_file = open( pcl_path_new, "wb" )
    pcl_new_file.write( bytes(pcl_new) )
    pcl_new_file.close()

# Set Data paths
lidar_path = 'data/KITTI/object/training/velodyne/000007.bin'
calib_path = 'data/KITTI/object/training/calib/000007.txt'
label_path = 'data/KITTI/object/training/label_2/000007.txt'

# Load Data
pcl, Tr_Velo2Cam, labels = PCLu.load_data(lidar_path, calib_path, label_path)

# Filter PCL
pcl_filtered, index_pcl = PCLu.filter_pcl(pcl)

# Get PCL's points in Camera coordinate system
pcl_cam, pcl_filtered_r = PCLu.get_pcl_in_cam_coord(pcl_filtered, Tr_Velo2Cam)

labels_new = []
pcl_new = pcl

# Process PCL's objects
for label in labels:
    # Get Object's data
    obj = Ku.Object3d(label)

    # Is Object a CAR?
    if obj.cls_id == 0:
        # Get object's centroid (in Camera coordinate system)
        centroid_cam = np.array(obj.t)

        # Get centroid in Velodyne coordinate system
        centroid_velo = PCLu.get_centroid_coord(centroid_cam, Tr_Velo2Cam, 'velo')

        # Limits of object's bounding box (WITHOUT yaw)
        bbox_cam = PCLu.get_bbox_coord(centroid_cam, obj, 'cam')
        bbox_velo = PCLu.get_bbox_coord(centroid_velo, obj, 'velo')

        # Limits of object's bounding box (WITH yaw)
        bbox_cam_yaw = PCLu.bbox_yaw(obj.ry, centroid_cam, bbox_cam)
        bbox_velo_yaw = PCLu.get_points_coord(bbox_cam_yaw, Tr_Velo2Cam, 'velo')

        # Is the CAR inside PCL's boundaries?
        if PCLu.test_object(bbox_velo):
            # Get PCL's points contained in object's bounding box (in Camera coordinate system)
            object_cam, index_obj = PCLu.get_object_points(pcl_cam, bbox_cam_yaw)

            # Get object's points in Velodyne coordinate system
            object_velo = PCLu.get_points_coord(object_cam, Tr_Velo2Cam, 'velo')

            # Apply pitch rotation
            theta = np.radians(15) # Pitch angle (in degrees) given in radians
            object_cam_pitch, bbox_cam_pitch = PCLu.rotations_cam_coord(obj.ry, theta, centroid_cam, bbox_cam_yaw, object_cam)

            #object_cam1 = object_cam.T # (3,N) -> (N,3)
            object_cam_pitch1 = object_cam_pitch.T # (3,N) -> (N,3)

            # Save NEW points in filtered PCL (in Camera coordinate system)
            pcl_cam = PCLu.change_pcl_cam(pcl_cam, object_cam_pitch1, index_obj)

            # Get NEW PCL in Velodyne coordinate system
            pcl_filtered_new = PCLu.get_pcl_in_velo_coord(pcl_cam, pcl_filtered_r, Tr_Velo2Cam)

            # Save NEW points in ORIGINAL PCL (in Velodyne coordinate system)
            pcl_new = PCLu.change_pcl(pcl_new, pcl_filtered_new, index_pcl)

            # Get NEW centroid coordinates (in Camera coordinate system)
            centroid_new = PCLu.new_centroid(object_cam_pitch)

            # Get NEW label data
            labels_new = PCLu.new_pcl_label(labels_new, obj, centroid_new, theta)

            # Get object points in Velodyne coordinate system (for visualization)
            #object_velo_new = PCLu.get_points_coord(object_cam, Tr_Velo2Cam, 'velo')
            #object_velo_pitch = PCLu.get_points_coord(object_cam_pitch, Tr_Velo2Cam, 'velo')
            #bbox_velo_pitch = PCLu.get_points_coord(bbox_cam_pitch, Tr_Velo2Cam, 'velo')

            #object_velo1 = object_velo_new.T # (3,N) -> (N,3)
            #object_velo_pitch1 = object_velo_pitch.T # (3,N) -> (N,3)

            # Visualize NEW rotated object and its bbox in PCL
            #PCLu.visualize(pcl_filtered_new, object_velo1, object_velo_pitch1, bbox_velo, bbox_velo_yaw, bbox_velo_pitch)
            #PCLu.visualize(pcl_filtered, object_velo1, bbox_velo, bbox_velo_yaw)

        else:
            label = label + " 0.00"
            labels_new.append(label)

    else:
        label = label + " 0.00"
        labels_new.append(label)

# Save NEW PCL file
pcl_new_file = open( "data_new/velodyne/000007.bin", "wb" )
pcl_new_file.write( bytes(pcl_new) )
pcl_new_file.close()

# Save NEW LABEL file
label_new_file = open( "data_new/label_2/000007.txt", "wt")
for label_new in labels_new: label_new_file.write(label_new + '\n')
label_new_file.close()

end_time = round( time.time() - start_time, 2)
print("Processing time = ", end_time)

total_time = total_time + end_time


''' APONTAMENTOS '''
# Get centroid in rectified Camera coordinate system
centroid_cam = PCLu.get_centroid_coord(centroid, Tr_Velo2Cam, 'cam')


point_test = np.matmul(Ry_t, point) - np.matmul(Ry_t, centroid)
R0_inv = np.linalg.inv( kitti_utils.rotz(-psi+(math.pi/2)) )


def rotations_bbox(psi, theta, centroid, points):
    # Get points' coordinates in object's coordinate system
    points_obj = points - centroid
    # Apply yaw rotation to correct coordinate system
    '''
    Velodyne:              Object:
    y ^                         ^ x
      |         ---Rz-->        |
    z o--> x               y <--o z
    '''
    if theta == 0:
        points_obj_rot = np.matmul(kitti_utils.rotz(-psi+(math.pi/2)), points_obj)
    # Apply pitch rotation
    else:
        R0_pitch = np.matmul(kitti_utils.rotz(-psi+(math.pi/2)), kitti_utils.roty(theta))
        points_obj_rot = np.matmul(R0_pitch, points_obj)
    # Get points' coordinates back in Velodyne's coordinate system
    points_velo_rot = points_obj_rot + centroid
    return points_velo_rot


def rotations_obj(psi, theta, centroid, points):
    # Get points' coordinates in object's coordinate system
    points_obj = points - centroid
    # Apply yaw rotation to correct coordinate system
    '''
    Velodyne:              Object:
    y ^                         ^ x
      |         ---Rz-->        |
    z o--> x               y <--o z
    '''
    if theta == 0:
        points_obj_rot = np.matmul(kitti_utils.rotz(-psi+(math.pi/2)), points_obj)
    # Apply pitch rotation
    else:
        R0_pitch = np.matmul(kitti_utils.roty(theta), kitti_utils.rotz(-(math.pi/2)+psi))
        points_obj_rot = np.matmul(R0_pitch, points_obj)
    # Get points' coordinates back in Velodyne's coordinate system
    points_velo_rot = points_obj_rot + centroid
    return points_velo_rot


def get_object_points(obj, pcl, centroid, bbox):
    object = []
    #print("Psi = ", obj.ry)

    # Avoid selecting points belonging to the ground
    zmin = bbox[2].min() + 0.05

    for point in pcl:
        # Point between bbox Z-axis' limits? (VELODYNE)
        if ( (point[2] >= zmin) & (point[2] <= bbox[2].max()) ):
            # Point belongs to object?
            if test_point(obj.ry, centroid, point, bbox):
                object.append(point)

    object = np.array(object).T # (N,4) --> (4,N)
    object = object[0:3, :] # (4,N) --> (3,N)

    return object


def test_point(psi, centroid, point, bbox):
    #centroid = np.reshape( centroid, (3,1) )
    point = np.reshape( point, (3,1) )

    # Get point coordinates in Object coordinate system
    #point_obj = point - centroid

    # Get homogeneous vector and matrix for matricial product
    #point_obj = np.vstack(( point_obj, np.ones((1,1)) ))
    #TrObj2Cam = np.hstack(( Ku.roty(psi).T, np.zeros((3,1)) ))
    #TrObj2Cam = np.vstack(( TrObj2Cam, np.array([0,0,0,1]) ))

    # Take yaw rotation of the object and check if it belongs to unrotated bbox
    #point_test = np.matmul( TrObj2Cam, point_obj )
    #point_test = point_test[0:3, :]

    # Get point coordinates back in Camera coordinate system
    #point_test = point_test + centroid

    # Point contained in Bbox's limits?
    if ( (point[0] > bbox[0].min()) & (point[0] < bbox[0].max()) &
         (point[2] > bbox[2].min()) & (point[2] < bbox[2].max()) ):
        return True
    else: return False


def test_point1(x, y, bbox):
    ''' As seen in https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library '''
    # 3 consequent vertices of rotated rectangle (bbox)
    xA = bbox[0,0]
    yA = bbox[1,0]
    xB = bbox[0,1]
    yB = bbox[1,1]
    xC = bbox[0,3]
    yC = bbox[1,3]

    # Lengths of sides
    lAB = np.sqrt( np.square(xB-xA) + np.square(yB-yA) )
    lBC = np.sqrt( np.square(xC-xB) + np.square(yC-yB) )

    # Normalized direction vectors
    uAB = [(xB-xA)/lAB, (yB-yA)/lAB]
    uBC = [(xC-xB)/lBC, (yC-yB)/lBC]

    # Vector BP
    BP = [(x-xB), (y-yB)]

    # Signed distances from sides to point using cross product
    SignedDistABP = BP[0]*uAB[1] - BP[1]*uAB[0]
    SignedDistBCP = - BP[0]*uBC[1] + BP[1]*uBC[0]

    if np.sign(SignedDistABP) == np.sign(SignedDistBCP):
        if ((abs(SignedDistABP) <= lBC) & (abs(SignedDistBCP) <= lAB)): return True
        else: return False
    else: return False


def edit_pcl(pcl, object, object_rot):
    idx_pcl = 0
    idx_obj = 0
    counter = 0
    for point_pcl in pcl:
        for point_obj in object:
            if np.array_equal(point_pcl, point_obj):
                #print(idx_pcl, " vs ", idx_obj)
                #print("Before: ", pcl[idx_pcl,:])
                pcl[idx_pcl,:] = object_rot[idx_obj,:]
                #print("After: ", pcl[idx_pcl,:])
                counter += 1
            idx_obj += 1
        if counter == len(object): break
        else: idx_obj = 0
        idx_pcl += 1
    #print("Counter: ", counter)

    return pcl


def draw_bbox_yaw(fig, bbox):
    bbox_top_x = [ bbox[0,0], bbox[0,1], bbox[0,3], bbox[0,2], bbox[0,0] ]
    bbox_top_y = [ bbox[1,0], bbox[1,1], bbox[1,3], bbox[1,2], bbox[1,0] ]
    bbox_top_z = [ bbox[2,0], bbox[2,1], bbox[2,3], bbox[2,2], bbox[2,0] ]
    mayavi.mlab.plot3d(bbox_top_x, bbox_top_y, bbox_top_z,
                       color=(0,1,0), # green
                       line_width=0.25,
                       figure=fig)

    bbox_bottom_x = [ bbox[0,4], bbox[0,5], bbox[0,7], bbox[0,6], bbox[0,4] ]
    bbox_bottom_y = [ bbox[1,4], bbox[1,5], bbox[1,7], bbox[1,6], bbox[1,4] ]
    bbox_bottom_z = [ bbox[2,4], bbox[2,5], bbox[2,7], bbox[2,6], bbox[2,4] ]
    mayavi.mlab.plot3d(bbox_bottom_x, bbox_bottom_y, bbox_bottom_z,
                       color=(0,1,0),
                       line_width=0.25,
                       figure=fig)

    bbox_front_x = [ bbox[0,0], bbox[0,1], bbox[0,5], bbox[0,4], bbox[0,0] ]
    bbox_front_y = [ bbox[1,0], bbox[1,1], bbox[1,5], bbox[1,4], bbox[1,0] ]
    bbox_front_z = [ bbox[2,0], bbox[2,1], bbox[2,5], bbox[2,4], bbox[2,0] ]
    mayavi.mlab.plot3d(bbox_front_x, bbox_front_y, bbox_front_z,
                       color=(0,1,0),
                       line_width=0.25,
                       figure=fig)

    bbox_back_x = [ bbox[0,2], bbox[0,3], bbox[0,7], bbox[0,6], bbox[0,2] ]
    bbox_back_y = [ bbox[1,2], bbox[1,3], bbox[1,7], bbox[1,6], bbox[1,2] ]
    bbox_back_z = [ bbox[2,2], bbox[2,3], bbox[2,7], bbox[2,6], bbox[2,2] ]
    mayavi.mlab.plot3d(bbox_back_x, bbox_back_y, bbox_back_z,
                       color=(0,1,0),
                       line_width=0.25,
                       figure=fig)

    return fig


def draw_bbox_pitch(fig, bbox):
    bbox_top_x = [ bbox[0,0], bbox[0,1], bbox[0,3], bbox[0,2], bbox[0,0] ]
    bbox_top_y = [ bbox[1,0], bbox[1,1], bbox[1,3], bbox[1,2], bbox[1,0] ]
    bbox_top_z = [ bbox[2,0], bbox[2,1], bbox[2,3], bbox[2,2], bbox[2,0] ]
    mayavi.mlab.plot3d(bbox_top_x, bbox_top_y, bbox_top_z,
                       color=(1,0,0), # red
                       line_width=0.25,
                       figure=fig)

    bbox_bottom_x = [ bbox[0,4], bbox[0,5], bbox[0,7], bbox[0,6], bbox[0,4] ]
    bbox_bottom_y = [ bbox[1,4], bbox[1,5], bbox[1,7], bbox[1,6], bbox[1,4] ]
    bbox_bottom_z = [ bbox[2,4], bbox[2,5], bbox[2,7], bbox[2,6], bbox[2,4] ]
    mayavi.mlab.plot3d(bbox_bottom_x, bbox_bottom_y, bbox_bottom_z,
                       color=(1,0,0),
                       line_width=0.25,
                       figure=fig)

    bbox_front_x = [ bbox[0,0], bbox[0,1], bbox[0,5], bbox[0,4], bbox[0,0] ]
    bbox_front_y = [ bbox[1,0], bbox[1,1], bbox[1,5], bbox[1,4], bbox[1,0] ]
    bbox_front_z = [ bbox[2,0], bbox[2,1], bbox[2,5], bbox[2,4], bbox[2,0] ]
    mayavi.mlab.plot3d(bbox_front_x, bbox_front_y, bbox_front_z,
                       color=(1,0,0),
                       line_width=0.25,
                       figure=fig)

    bbox_back_x = [ bbox[0,2], bbox[0,3], bbox[0,7], bbox[0,6], bbox[0,2] ]
    bbox_back_y = [ bbox[1,2], bbox[1,3], bbox[1,7], bbox[1,6], bbox[1,2] ]
    bbox_back_z = [ bbox[2,2], bbox[2,3], bbox[2,7], bbox[2,6], bbox[2,2] ]
    mayavi.mlab.plot3d(bbox_back_x, bbox_back_y, bbox_back_z,
                       color=(1,0,0),
                       line_width=0.25,
                       figure=fig)

    return fig

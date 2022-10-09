import random
import time
import matplotlib.pyplot as plt

import utils.pcl_utils as PCLu


start_time = time.time()

# Get list of PCLs to process
# with open('data/KITTI/ImageSets/manipulation-exp1.txt', 'r') as f: # experiment 1
with open('data/KITTI/ImageSets/manipulation-exp2.txt', 'r') as f: # experiment 2
    data = f.read()

pcls_list = [int(i) for i in data.split()]

# Auxiliary variable and lists
obj_count_total = 0
obj_height_list = []
theta_list = []

numbers_list = random.sample( range(0, 7481), 7481 )

for number in numbers_list:
    print("PCL no.: ", number)

    # Process current PCL
    obj_count_pcl, obj_height_list_aux, theta_list_aux = PCLu.process_pcl(number, pcls_list)

    obj_count_total += obj_count_pcl

    for height in obj_height_list_aux: obj_height_list.append(height)

    for theta in theta_list_aux: theta_list.append(theta)

    elapse_time = round( time.time() - start_time, 2)
    print( "Elapsed time = ", time.strftime( "%H:%M:%S", time.gmtime(elapse_time) ) )

print( "Total time = ", time.strftime( "%H:%M:%S", time.gmtime(elapse_time) ) )

# print( "Total number of Car and Van classes' objects = ", obj_count_total )
# print( "Total number of Cyclist classes' objects = ", obj_count_total )
print( "Total number of All classes' objects = ", obj_count_total )

# print( "Car and Van height - minimum = ", min(obj_height_list) )
# print( "Cyclist height - minimum = ", min(obj_height_list) )
print( "All classes height - minimum = ", min(obj_height_list) )

# print( "Car and Van height - maximum = ", max(obj_height_list) )
# print( "Cyclist height - maximum = ", max(obj_height_list) )
print( "All classes height - maximum = ", max(obj_height_list) )

# print( "Car and Van height - average = ", sum(obj_height_list) / len(obj_height_list) )
# print( "Cyclist height - average = ", sum(obj_height_list) / len(obj_height_list) )
print( "All classes height - average = ", sum(obj_height_list) / len(obj_height_list) )

list_file = open( 'data_new/experiment-2/all_classes/list_full_dataset.txt', "wt" )
for number in numbers_list: list_file.write(str(number) + '\n')
list_file.close()

height_file = open( 'data_new/experiment-2/all_classes/obj_height_list.txt', "wt" )
for height in obj_height_list: height_file.write(str(height) + '\n')
height_file.close()

theta_file = open( 'data_new/experiment-2/all_classes/theta_list.txt', "wt" )
for theta in theta_list: theta_file.write(str(theta) + '\n')
theta_file.close()

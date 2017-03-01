import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os
#%matplotlib inline
import pickle
import cv2
import csv

#steering angle small positive and negative 0-0.3
steering_angle_small_negative = []
steering_angle_small_positive = []
#steering angle medium positive and negative 0.3-0.5
steering_angle_medium_negative = []
steering_angle_medium_positive = []
#steering angle large positive and negative 0.5-1.0
steering_angle_large_negative = []
steering_angle_large_positive = []
#steering angle zero
steering_angle_zero = []
#steering angle negative one
steering_angle_negative_one = []
#steering angle positive one
steering_angle_positive_one = []

image_path = "sunny_data"
center_image_file = "sunny_data/driving_log_feb22_25.csv"
with open(center_image_file, 'rt') as f:
     reader = csv.reader(f, delimiter=',')
     #next(reader, None)
     for row in reader:
         #Get image path and steering value
         #center_image_path = row[0]
         steering_value = row[3]
         steering_value = float(steering_value)

         if abs(steering_value) == 0:
            steering_angle_zero.append(row)
         elif steering_value == -1:
            steering_angle_negative_one.append(row)
         elif steering_value == 1:
            steering_angle_positive_one.append(row)
         elif steering_value < 0 and steering_value >= -0.3:
            steering_angle_small_negative.append(row)
         elif steering_value < -0.3 and steering_value >= -0.5:
            steering_angle_medium_negative.append(row)
         elif steering_value < -0.5 and steering_value > -1:
            steering_angle_large_negative.append(row)
         elif steering_value > 0 and steering_value <= 0.3:
            steering_angle_small_positive.append(row)
         elif steering_value > 0.3 and steering_value <= 0.5:
            steering_angle_medium_positive.append(row)
         else:
            steering_angle_large_positive.append(row)

print (np.array(steering_angle_small_positive).shape)
print (np.array(steering_angle_medium_positive).shape)
print (np.array(steering_angle_large_positive).shape)
print (np.array(steering_angle_small_negative).shape)
print (np.array(steering_angle_medium_negative).shape)
print (np.array(steering_angle_large_negative).shape)
print (np.array(steering_angle_zero).shape)
print (np.array(steering_angle_positive_one).shape)
print (np.array(steering_angle_negative_one).shape)

x_master = []
y_master = []

for i in range(0, 400):
    steering_value = steering_angle_zero[i][3]
    center_image_path = steering_angle_zero[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)
    #print (center_image_path)
    #print (steering_value)

for i in range(0, 1355):
    steering_value = steering_angle_small_negative[i][3]
    center_image_path = steering_angle_small_negative[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)
    #print (center_image_path)
    #print (steering_value)

for i in range(0, 741):
    steering_value = steering_angle_small_positive[i][3]
    center_image_path = steering_angle_small_positive[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)

for i in range(0, 569):
    steering_value = steering_angle_medium_negative[i][3]
    center_image_path = steering_angle_medium_negative[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)
    #print (center_image_path)
    #print (steering_value)

for i in range(0, 293):
    steering_value = steering_angle_medium_positive[i][3]
    center_image_path = steering_angle_medium_positive[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)

for i in range(0, 300):
    steering_value = steering_angle_large_negative[i][3]
    center_image_path = steering_angle_large_negative[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)
    #print (center_image_path)
    #print (steering_value)

for i in range(0, 300):
    steering_value = steering_angle_large_positive[i][3]
    center_image_path = steering_angle_large_positive[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)

for i in range(0, 200):
    steering_value = steering_angle_positive_one[i][3]
    center_image_path = steering_angle_positive_one[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)

for i in range(0, 200):
    steering_value = steering_angle_negative_one[i][3]
    center_image_path = steering_angle_negative_one[i][0]
    steering_value = float(steering_value)
    y_master.append(steering_value)
    image = mpimg.imread(center_image_path)
    x_master.append(image)

# #add flipped large steering angle data
# for i in range(0, 50):
#     steering_value = steering_angle_large[i][3]
#     center_image_path = steering_angle_large[i][0]
#     steering_value = float(steering_value)
#     steering_value = steering_value * -1
#     y_master.append(steering_value)
#     image = mpimg.imread(center_image_path)
#     image = np.array(image)
#     image_flipped = np.fliplr(image)
#     x_master.append(image_flipped)
#     #print (center_image_path)
#     #print (steering_value)

print (np.array(x_master).shape)
print (np.array(y_master).shape)


new_images_dict = []
#this loop resize the image by half its size
for i in range(0, np.array(x_master).shape[0]):
      image = x_master[i]
      image = image[80:,:]
      new_images_dict.append(cv2.resize(image, (0,0), fx=0.5, fy=0.5))
#
#
print (np.array(new_images_dict).shape)
#
# ##### JC  Add code here to pickle both the images and the y values
#
pickle.dump(new_images_dict, open( "sample_data_x.p", "wb"))
pickle.dump(y_master, open( "sample_data_y.p", "wb"))
#
print ("sample data x and y saved")




































# #add sample data
#
# ####JC Add arrow for steering data
# steering_dict = []
#
#  ##### JC  Make this section read the images and the steering values at the same time and then save them to the proper arrays
# image_data_array = []
# image_path = "sample_data"
# center_image_file = "sample_data/driving_log.csv"
# with open(center_image_file, 'rt') as f:
#      reader = csv.reader(f, delimiter=',')
#      next(reader, None)
#      for row in reader:
#          #Get image path and steering value
#          center_image_path = row[0]
#          steering_value = row[3]
#
#          #add image and steering value to proper arrays
#          steering_value = float(steering_value)
#          steering_dict.append(steering_value)
#          image = mpimg.imread(image_path + "/" + center_image_path)
#          image_data_array.append(image)
#
# # Add Sunny data secion
# center_image_file = "sunny_data/driving_log.csv"
# with open(center_image_file, 'rt') as f:
#      reader = csv.reader(f, delimiter=',')
#      for row in reader:
#          #Get image path and steering value
#          center_image_path = row[0]
#          steering_value = row[3]
#
#          #add image and steering value to proper arrays
#          steering_value = float(steering_value)
#          steering_dict.append(steering_value)
#          image = mpimg.imread(center_image_path)
#          image_data_array.append(image)
#
# # #Add additional data 1
# # center_image_file = "sunny_data/driving_log_feb22.csv"
# # print ("adding additional data")
# # with open(center_image_file, 'rt') as f:
# #      reader = csv.reader(f, delimiter=',')
# #      for row in reader:
# #          #Get image path and steering value
# #          steering_value = float(row[3])
# #          center_image_path = row[0]
# #          if steering_value != 0:
# #               #center_image_path = row[0]
# #               #steering_value = row[3]
# #               #steering_value = float(steering_value)
# #               steering_dict.append(steering_value)
# #               image = mpimg.imread(center_image_path)
# #               image_data_array.append(image)
# #               print (row[0])
#
# #Add additional data 2
# print("Data with problem")
# center_image_file = "sunny_data/driving_log_feb22_23_3.csv"
# print ("adding additional data")
# with open(center_image_file, 'rt') as f:
#      reader = csv.reader(f, delimiter=',')
#      for row in reader:
#          #Get image path and steering value
#          steering_value = row[3]
#          center_image_path = row[0]
#          print(steering_value)
#          print(center_image_path)
#          steering_value = float(steering_value)
#          if abs(steering_value) >= 0.1:
#               #center_image_path = row[0]
#               #steering_value = row[3]
#               #steering_value = float(steering_value)
#               steering_dict.append(steering_value)
#               image = mpimg.imread(center_image_path)
#               image_data_array.append(image)
#               print (row[0])
#
#
# # # Add Sample data secion with flip
# # print("Adding flipped images")
# # image_path = "sample_data"
# # center_image_file = "sample_data/driving_log.csv"
# # with open(center_image_file, 'rt') as f:
# #      reader = csv.reader(f, delimiter=',')
# #      next(reader, None)
# #      for row in reader:
# #          #Get image path and steering value
# #          center_image_path = row[0]
# #          steering_value = row[3]
# #          steering_value = float(steering_value)
# #
# #          # How to skip 0's
# #          if steering_value != 0:
# #              #if steering!=0 then
# #              #add image and steering value to proper arrays
# #              ### flip logic goes here before adding values to arrays
# #              steering_value = -steering_value
# #              print(steering_value)
# #              steering_dict.append(steering_value)
# #              image = mpimg.imread(image_path + "/" + center_image_path)
# #              image_flipped = np.fliplr(image)
# #              image_data_array.append(image_flipped)
#
#
#
#
# new_images_dict = []
# #this loop resize the image by half its size
# for i in range(0, np.array(image_data_array).shape[0]):
#     image = image_data_array[i]
#     image = image[80:,:]
#     new_images_dict.append(cv2.resize(image, (0,0), fx=0.5, fy=0.5))
#
# #resized_image_dict = tf.image.resize_images(new_images_dict, [80, 160])
#
# print (np.array(new_images_dict).shape)
#
# ##### JC  Add code here to pickle both the images and the y values
#
# pickle.dump( new_images_dict, open( "sample_data_x.p", "wb"))
# pickle.dump(steering_dict, open( "sample_data_y.p", "wb"))
#
# print ("sample data x and y saved")

This report explains my approach to Udacity Behavioral Cloning project.
Please visit https://github.com/sunnymarketingpdx/udacity_behavioral_cloning/issues/1
for the full version with images. The following version doesn't include images.

Files include:

data_process_x_y.py

model_sunny.py

drive.py

model.h5

behavioral_cloning.mp4

README.md

Step 1: Data collection

I collected data by recording the car moving from the edge of the road to the center of the road on the full track. For sharp turns (such as near the lake), I recorded 3 times for each sharp turn to collect enough training data.

Center lane driving sample picture

![center_2017_02_25_11_56_07_537](https://cloud.githubusercontent.com/assets/11469505/23437375/bc0e3b9c-fdc2-11e6-8b00-2ede248a5971.jpg)

Recovering from the left side and right sides of the road back to center

center_2017_02_25_11_57_11_761

center_2017_02_25_11_57_15_731

center_2017_02_25_11_56_29_697

center_2017_02_25_11_56_36_029

Sample of cropped image: 80X320
screen shot 2017-02-28 at 3 34 45 pm

Step 2: Data processing

File: data_process_x_y.py

How to process mages:

Add my sample data - match the image name with the name in CSV file
Crop the images to 80x320
Resize the image by 50%
Save all data in sample_data_x.p pickle file

How to process CSV files:

Add my sample data - steering angles
Save all data in sample_data_y.p pickle file

Histogram of y data (steering angle)
screen shot 2017-02-28 at 3 31 40 pm

After many trails and errors, I found when the car drives inside the lanes, most times the steering angle is below 0.3. In sharp turns, the angle is around 0.3 to 0.5. Also the number of zero angles, negative 1 angle and positive 1 angle will add more bias which causes the car to drive off the road very quickly. To get better result, the number of zero data need to match the number of positive 1 and negative 1 number combined.

Therefore I put data into following categories:

Steering Angle Small Positive, including steering data larger than 0 and smaller than 0.3 (total 741)
Steering Angle Small Negative, including steering data smaller than 0 and larger than -0.3 (total 1355)
Steering Angle Medium Positive, including steering data larger than 0.3 and smaller than 0.5 (total 293)
Steering Angle Medium Negative, including steering data larger than -0.5 and smaller than -0.3 (total 569)
Steering Angle Large Positive, including steering data larger than 0.5 and smaller than 1 (total 624)
Steering Angle Large Negative, including steering data larger than -1 and smaller than -0.5 (total 1278)
Steering Angle Zero, including all zero steering data (total 14201)
Steering Angle Negative One, including all -1 steering data (total 1115)
Steering Angle Positive One, including all 1 steering data (total 326)
And I distributed data as the following:

Steering Angle Small Positive - 741 (that's the total number of small positive angle in my dataset)
Steering Angle Small Negative - 1355 (that's the total number of small positive angle in my dataset)
Steering Angle Medium Positive - 293 (that's the total number of medium positive angle in my dataset)
Steering Angle Medium Negative - 569 (that's the total number of medium negative angle in my dataset)
Steering Angle Large Positive - 300
Steering Angle Large Negative - 300
Steering Angle Zero - 400
Steering Angle Negative One - 200
Steering Angle Positive One - 200
Step 3: Build model (Got inspiration from NVIDIA model and couple blog posts posted by Udacity)

File: model_sunny.py

Why I chose the model:

I got inspiration mostly from NVIDIA model and blog posts of fellow students posted by David Silver.
End to End Learning for Self-Driving Cars
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

How Udacity’s Self-Driving Car Students Approach Behavioral Cloning
https://medium.com/udacity/how-udacitys-self-driving-car-students-approach-behavioral-cloning-5ffbfd2979e5#.mc5ivytov

The training time is below 30 minutes which is ideal
How the model works:

Reload images and steering angles from pickle files
Shuffle data
Set batch size = 64, class = 1 (steering angle), epoch = 10
Build 4 convolutional filters
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2
Set size of max pooling to pool_size = (2, 2)
Set convolution kernel size kernel_size = (3, 3)
How the model works:
1st layer convert 1 channel to 16 channels
Apply ReLU
2nd layer convert 16 channels to 8 channels
Apply ReLu
3rd layer convert 8 channels to 4 channels
Apply ReLu
4th layer convert 4 channels to 2 channels
Apply ReLu
Apply Max Pooling for 2X2
Apply dropout 25%
Flatten the matrix. Input size 360, Output size 16
Apply ReLu
Input size 16 and Output size 16
Apply ReLu
Input size 16 and Output size 16 (2nd time)
Apply ReLu
Dropout 50%
Use Adam Optimizer
Split data into training data (80%) and testing data (20%), then split training data into training data (80%) and validation data (20%).
Got accuracy of about 0.09%
Step 4: Test on simulator (Run drive.py and model.h5)

Center driving

screen shot 2017-02-28 at 4 31 58 pm
screen shot 2017-02-28 at 4 32 14 pm

Bridge
screen shot 2017-02-28 at 4 32 23 pm

Sharp left turn
screen shot 2017-02-28 at 4 32 56 pm

Sharp right turn
screen shot 2017-02-28 at 4 33 11 pm

Conclusion:

1. The distribution of small, medium and large angles is the key of success. I was stuck with sharp turns for couple weeks. After reading more blog posts of other students, I got inspired and added the right number of zero and extreme steering angles (-1 and 1) and successfully decreased the bias.

2. Small angles (from -0.3 to 0.3) should be the highest amount in the whole dataset. The key of passing sharp turns is to break it down to small steering angles.

3. I tried flipping images, add more epochs and more data at difficult turns. However they didn't work unless the distribution of data is right. This is a very important lesson I learned from this project.

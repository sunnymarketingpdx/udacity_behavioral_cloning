This report explains my approach to Udacity Behavioral Cloning project.

Files include:

1. data_process_x_y.py

2. model_sunny.py

3. drive.py

4. model.h5

5. behavioral_cloning.mp4

6. README.md

Step 1: Data collection

I collected data by recording the car moving from the edge of the road to the center of the road on the full track. For sharp turns (such as near the lake), I recorded 3 times for each sharp turn to collect enough training data.

Center lane driving sample picture

![center_2017_02_25_11_56_07_537](https://cloud.githubusercontent.com/assets/11469505/23437375/bc0e3b9c-fdc2-11e6-8b00-2ede248a5971.jpg)

Recovering from the left side and right sides of the road back to center

![center_2017_02_25_11_57_11_761](https://cloud.githubusercontent.com/assets/11469505/23437440/04cd6a92-fdc3-11e6-8524-8fcbefdebad5.jpg)

![center_2017_02_25_11_57_15_731](https://cloud.githubusercontent.com/assets/11469505/23437444/09431608-fdc3-11e6-84de-b6faf2e03db7.jpg)

![center_2017_02_25_11_56_29_697](https://cloud.githubusercontent.com/assets/11469505/23437452/0ea59580-fdc3-11e6-9350-19f3d06a8671.jpg)

![center_2017_02_25_11_56_36_029](https://cloud.githubusercontent.com/assets/11469505/23437454/113c61b6-fdc3-11e6-815f-c05b49b91ecd.jpg)

Sample of cropped image: 80X320

![screen shot 2017-02-28 at 3 34 45 pm](https://cloud.githubusercontent.com/assets/11469505/23439454/7f9fb088-fdcb-11e6-81fc-dbd897f972b0.png)

Step 2: Data processing

File: data_process_x_y.py

How to process mages:

1. Add my sample data - match the image name with the name in CSV file
2. Crop the images to 80x320
3. Resize the image by 50%
4. Save all data in sample_data_x.p pickle file

How to process CSV files:

1. Add my sample data - steering angles
2. Save all data in sample_data_y.p pickle file

Histogram of y data (steering angle)

![screen shot 2017-02-28 at 3 31 40 pm](https://cloud.githubusercontent.com/assets/11469505/23439576/24da3f5a-fdcc-11e6-91dc-c3b8664997b5.png)

After many trails and errors, I found when the car drives inside the lanes, most times the steering angle is below 0.3. In sharp turns, the angle is around 0.3 to 0.5. Also the number of zero angles, negative 1 angle and positive 1 angle will add more bias which causes the car to drive off the road very quickly. To get better result, the number of zero data need to match the number of positive 1 and negative 1 number combined.

Therefore I put data into following categories:

1. Steering Angle Small Positive, including steering data larger than 0 and smaller than 0.3 (total 741)
2. Steering Angle Small Negative, including steering data smaller than 0 and larger than -0.3 (total 1355)
3. Steering Angle Medium Positive, including steering data larger than 0.3 and smaller than 0.5 (total 293)
4. Steering Angle Medium Negative, including steering data larger than -0.5 and smaller than -0.3 (total 569)
5. Steering Angle Large Positive, including steering data larger than 0.5 and smaller than 1 (total 624)
6. Steering Angle Large Negative, including steering data larger than -1 and smaller than -0.5 (total 1278)
7. Steering Angle Zero, including all zero steering data (total 14201)
8. Steering Angle Negative One, including all -1 steering data (total 1115)
9. Steering Angle Positive One, including all 1 steering data (total 326)

And I distributed data as the following:

1. Steering Angle Small Positive - 741 (that's the total number of small positive angle in my dataset)
2. Steering Angle Small Negative - 1355 (that's the total number of small positive angle in my dataset)
3. Steering Angle Medium Positive - 293 (that's the total number of medium positive angle in my dataset)
4. Steering Angle Medium Negative - 569 (that's the total number of medium negative angle in my dataset)
5. Steering Angle Large Positive - 300
6. Steering Angle Large Negative - 300
7. Steering Angle Zero - 400
8. Steering Angle Negative One - 200
9. Steering Angle Positive One - 200

Step 3: Build model (Got inspiration from NVIDIA model and couple blog posts posted by Udacity)

File: model_sunny.py

Why I chose the model:

1. I got inspiration mostly from NVIDIA model and blog posts of fellow students posted by David Silver.
End to End Learning for Self-Driving Cars
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

How Udacity’s Self-Driving Car Students Approach Behavioral Cloning
https://medium.com/udacity/how-udacitys-self-driving-car-students-approach-behavioral-cloning-5ffbfd2979e5#.mc5ivytov

2. The training time is below 30 minutes which is ideal

How the model works:

1. Reload images and steering angles from pickle files
2. Shuffle data
3. Set batch size = 64, class = 1 (steering angle), epoch = 10
4. Build 4 convolutional filters
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2
5. Set size of max pooling to pool_size = (2, 2)
6. Set convolution kernel size kernel_size = (3, 3)

How the model works:

- 1st layer convert 1 channel to 16 channels
- Apply ReLU
- 2nd layer convert 16 channels to 8 channels
- Apply ReLu
- 3rd layer convert 8 channels to 4 channels
- Apply ReLu
- 4th layer convert 4 channels to 2 channels
- Apply ReLu
- Apply Max Pooling for 2X2
- Apply dropout 25%
- Flatten the matrix. Input size 360, Output size 16
- Apply ReLu
- Input size 16 and Output size 16
- Apply ReLu
- Input size 16 and Output size 16 (2nd time)
- Apply ReLu
- Dropout 50%
- Use Adam Optimizer
- Split data into training data (80%) and testing data (20%), then split training data into training data (80%) and validation data (20%).
- Got accuracy of about 0.09%

Step 4: Test on simulator (Run drive.py and model.h5)

Center driving

![screen shot 2017-02-28 at 4 31 58 pm](https://cloud.githubusercontent.com/assets/11469505/23440915/f80abd08-fdd3-11e6-884f-f63bc62ad5fb.png)

![screen shot 2017-02-28 at 4 32 14 pm](https://cloud.githubusercontent.com/assets/11469505/23440921/fa0e5862-fdd3-11e6-839d-7e3aed4cb358.png)

Bridge

![screen shot 2017-02-28 at 4 32 23 pm](https://cloud.githubusercontent.com/assets/11469505/23440924/010724aa-fdd4-11e6-9ed8-a48db91be50f.png)

Sharp left turn

![screen shot 2017-02-28 at 4 32 56 pm](https://cloud.githubusercontent.com/assets/11469505/23440931/085bada2-fdd4-11e6-8383-f03be1074a5f.png)

Sharp right turn

![screen shot 2017-02-28 at 4 33 11 pm](https://cloud.githubusercontent.com/assets/11469505/23440938/0e6d3df0-fdd4-11e6-849c-f3f61da94c5d.png)

Conclusion:

1. The distribution of small, medium and large angles is the key of success. I was stuck with sharp turns for couple weeks. After reading more blog posts of other students, I got inspired and added the right number of zero and extreme steering angles (-1 and 1) and successfully decreased the bias.

2. Small angles (from -0.3 to 0.3) should be the highest amount in the whole dataset. The key of passing sharp turns is to break it down to small steering angles.

3. I tried flipping images, add more epochs and more data at difficult turns. However they didn't work unless the distribution of data is right. This is a very important lesson I learned from this project.

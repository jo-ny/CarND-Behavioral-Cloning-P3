# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_camera.jpg "center_camera"
[image2]: ./images/left_camera.jpg "left_camera"
[image3]: ./images/right_camera.jpg "right_camera"
[image4]: ./images/original_image.jpg "original_image"
[image5]: ./images/fliped_image.jpg "fliped_image"
[image6]: ./images/crop_image.jpg "crop_image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* model.py also containing the script to create and train the model, load data with generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 drive autonomously video of sample data set
* run2.mp4 drive autonomously video of my data set

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network：
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| normalized         	| lambda x: x/255.0 - 0.5   				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 156x316x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 78x158x6   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 74x154x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 37x77x6   				|
| Fully connected		| 128 node, outputs 128.      					|
| Fully connected		| 84 node, outputs 84.       					|
| Fully connected		| 1 node, outputs 1.        					|

The model includes RELU layers to introduce nonlinearity.  
The data is normalized in the model using a Keras lambda layer.   

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The data sets contain：  
- forward_data * 2
- reverse_data * 2
- curve_data 

To further reduce overfitting, add flipped images.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, on the curve driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

非常抱歉，英语水平不是很高，为了清洗表达我的意思，这里使用中文进行描述:

总体的思路是：  

首先使用了udacity提供的数据，并最简单的数据进行建模，测试时发现效果很差。 

紧接着使用了LeNet模型，这次效果有所改善，但是在弯道处还是会出现超出跑道的问题。  

之后我使用了NVIDIA的模型，这次车子可以自动行驶较远的距离，但是还是会在没有路牙的地方超出跑道。  

之后我使用模拟器自己收集数据，一共收集了完整的正向两圈，反向两圈的数据，还有一些专门在弯道处录制的数据；同时使用这些数据中左，中，右三个摄像头的数据，并对图像进行了翻转；最终数据集扩充到90K；还对数据做了归一化处理，并且裁减掉了图片上方50像素，下方25像素的无关区域。  

最终，训练出的模型能够使模拟器中的小车流畅地完成整个赛道，不出现超出跑道或者车子在跑道上左右摇摆等问题。  

为了加速测试过程，我将模拟器的默认速度从9提高到了20，整个测试过程大大加快。  

最后，我将使用udacity提供的数据建立的模型进行了测试，并输出了测试结果，视频文件为 run1.MP4。  

使用我自己收集的数据的建立的模型，输出了测试结果，视频文件为run2.MP4。  

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| normalized         	| lambda x: x/255.0 - 0.5   				    |
| Cropping2D         	| 50,25, 0,0   			            			| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 40x158x24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 18x77x36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 7x37x48 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 5x35x64    |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x33x64    |
| RELU					|												|
| Fully connected		| 100 node, outputs 100.      					|
| Fully connected		| 50 node, outputs 50.       					|
| Fully connected		| 10 node, outputs 10.        					|
| Fully connected		| 1 node, outputs 1.        					|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][image1]

To simulate going back to the middle from left/right of the road, I used images from cameras on both sides of the car, add/subtract a correction factor for the current measurements, respectively.


![left][image2]
![right][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images. For example, here is an image that has then been flipped:

![original][image4]
![flipped][image5]

In order to reduce the impact of sky, trees and so on in the image, I cut the area of 50 above and 25 below the image.

![crop][image6]

After the collection process, I had 97284 number of data points.

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by loss and val_loss are small. I used an adam optimizer so that manually training the learning rate wasn't necessary.

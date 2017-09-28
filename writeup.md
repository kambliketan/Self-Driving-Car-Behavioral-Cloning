# **Behavioral Cloning** 

## Writeup Template

### Udacity Self Driving Car Nanodegree: Project 3: Behavioral Cloning.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./nvidia-cnn-architecture.png "NVIDIA CNN Architecture"
[image2]: ./recovery_from_right.jpg "Recovery From Right"
[image3]: ./recovery_from_left.jpg "Recovery From Left"
[image4]: ./dirt_training.jpg "Training on Dirt Edge"
[image5]: ./center_driving.jpg "Center Driving"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to generate data, create model and train the model. The final model code is also copied down below.
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```

#### 3. Submission code is usable and readable

The Ipython notebook model.ipynb contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.ipynb). Final model employs 5 convolutional layers followed by 4 fully connected layers. 

The data is preprocessed by normalizing and cropping it. Normalization employs Keras lambda layer, while cropping is perfomed using `Cropping2D` Keras function.

#### 2. Attempts to reduce overfitting in the model

In the trial models I used dropout layers in order to reduce overfitting. The final model was trained and validated on different data sets to ensure that the model was not overfitting for example a reverse lap helps generalize the model better. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road along with the sample data that was provided for the project.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with minimun number of layers and then build iteratively.

My first step was to just try fully connected layers which did not keep the vehicle on the road for long. I then used a convolution neural network model similar to the LENET. I thought this model might be appropriate because this model gave >95% of accuracy with the Traffic signs data. And since this project also involves learing from pictorial data, this was the first default choice. After first attempts with fully connected layers, it was clear that the kinda of understanding of images we desire requires a convolutional neural network, which are really good at understanding spatial data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include few Dropout layers. This gave limited success but the vehicle still had trouble staying on course and it could not even reach the first curve.

Then I tried various data ammendment strategies - driving the lap in reverse, recover from right side edge, recover from left side ede etc which helped generalize the model and prevent overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track - for example the dirt edge around the middle of the track. To improve the driving behavior in these cases, I collected special recovery data focused on those areas.

I them employed andvanced architecture described in following section. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a combination of convolution neural network with fully connected layers. Here're the detailed layers and layer sizes -

This architecture was coined by NVIDIA and was discussed briefly in lessons: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

![alt text][image1]

| Layer | Input | Output |
|:------:|:----:|:-------:|
| Normalization Layer | Input: 160 x 320 x 3 | Output: 160 x 320 x 3 |
| Cropping Layer | Input: 160 x 320 x 3 | Output: 65 x 320 x 3 |
| Convolutional Layer 1 5x5 filter with output depth 24 | Input: 65 x 320 x 3 | Output: 31 x 158 x 24 |
| Convolutional Layer 2 5x5 filter with output depth 36 | Input: 31 x 158 x 24 | Output: 14 x 77 x 36 |
| Convolutional Layer 3 5x5 filter with output depth 48 | Input: 14 x 77 x 36 | Output: 5 x 37 x 48 |
| Convolutional Layer 4 3x3 filter with output depth 64 | Input: 5 x 37 x 48 | Output: 3 x 35 x 64 |
| Convolutional Layer 5 3x3 filter with output depth 64 | Input: 3 x 35 x 64 | Output: 1 x 33 x 64 |
| Fully Connected Layer 1 | Input: 2112 | Output: 100 |
| Fully Connected Layer 2 | Input: 100 | Output: 50 |
| Fully Connected Layer 3 | Input: 50 | Output: 10 |
| Fully Connected Layer 4 | Input: 10 | Output: 1 |

```sh
#######################################################################
# Final Model: Employing advanced NVIDIA architecture
#######################################################################

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Lambda, Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# normalize
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
# cropping
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# convolution layer # 1
# input size: 65 x 320 x 3
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# output size: 31 x 158 x 24
## new_height = (input_height - filter_height + 2 * P)/S + 1

# convolution layer # 2
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# output size: 14 x 77 x 36

# convolution layer # 3
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
# output size: 5 x 37 x 48

# convolution layer # 4
model.add(Convolution2D(64, 3, 3, activation='relu'))
# output size: 3 x 35 x 64

# convolution layer # 5
model.add(Convolution2D(64, 3, 3, activation='relu'))
# output size: 1 x 33 x 64

# flatten
model.add(Flatten())
# output size: 2112

# fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=4)

callback_model_checkpoint = ModelCheckpoint('./model_final_checkpoints/model_{epoch:02d}.h5',  verbose = 1)
history = model.fit(X, y, validation_split = 0.2, shuffle = True, nb_epoch = 20, callbacks = [callback_model_checkpoint])

model.save('model_final.h5')
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center of the lane. These images show what a recovery looks like:

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped all images I captured along with corresponding steering angle measurement thinking that this would give me more data and generalize the model better.
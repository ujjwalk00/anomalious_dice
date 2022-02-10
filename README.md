# anomalious_dice

## Preprocessing Data

We started this project by looking at the data provided. The image dataset contains 6571 images
of dice facing towards the camera. They are 128x128 and divided into 11 classes. Each of the
6 faces has several directions in which they can be oriented. This explains why there are
so many classes.

A first step was to edit these images so they can be divided into only 6 classes. One for each face.
That would in the long run save us inference time of any running model, because less comparisons 
or generations would need to be made for training and predicting.

### Rotations

In order to get these dice facing the correct direction a method was used that draws a line halfway 
along the image and counts the amount of dark pixels. We then rotate the image by 1 degree and repeat.
When doing this for all the diagonals we, in the end, can select the one with the most dark pixels,
and rotate in this way. 

![](visuals/dice-perprocessing.gif)

For doing this we had to crop circles out of our images in order to avoid misclassifications with the 
edge of the image.

From these templates were made by squashing the numpy arrays together.

1                           |2                           |3                           |4                           |5                           |6                           |
:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
![](visuals/templates/1.png)|![](visuals/templates/2.png)|![](visuals/templates/3.png)|![](visuals/templates/4.png)|![](visuals/templates/5.png)|![](visuals/templates/6.png)

You can clearly see this approach was not perfect in the last template. This could be solved by adjusting
thresholds but in the interest of saving time it was left to a later stage.

This approach did have an added benefit that anomalies would actually affect the symmetry of the result.
In the following examples you can see that the anomalies are sometimes rotated in a way that differs from 
how normal dice are rotated. When comparing these values to the templates, we can immediately recognize a
number of anomalies without any modeling.

![](visuals/ano3.gif)
![](visuals/ano4.gif)

### Thresholds

After this stage images of all classes were compared to the templates and using loss functions the differences
would be calculated for each of these templates. Doing this gave an idea of whithin what range a correct classification 
would be. and these were then used as thresholds in later stages of the project.

This is an example using MSEloss

category 1    |category 2    |category 3    |category 4    |category 5    |category 6    |
:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
62.02780473883|60.43002574095|66.30725609998|69.93280870737|74.30941474250|82.75002536741|

## Numpy approach

For this approach we relied heavily on the thresholds calculated in the previous chapter.
When a prediction is made on a sample, that sample is compared to each of the thresholds. 
If it falls within the boundaries of one of these it is calculated as a normal sample.

But when a sample like this one is given: 

![](processed_data/test_set/ano/17_11_21_anomalies_005.png)

![](visuals/templates/1.png)|![](visuals/templates/2.png)|![](visuals/templates/3.png)|![](visuals/templates/4.png)|![](visuals/templates/5.png)|![](visuals/templates/6.png)
:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
65.5125 > 62.0278047        |75.4151 > 60.43002574       |67.5423 >66.3072560999      |72.4521 > 69.9328087        |77.5124 > 74.3094147        |91.542 > 82.750025  
MSEloss > thresh 1        |MSEloss > thresh 2        |MSEloss > thresh 3        |MSEloss > thresh 4        |MSEloss > thresh 5        |MSEloss > thresh 6  

And MSEloss for this sample falls outside of the boundaries for each category it is
classified as an anomaly.

Over the training data, an f1-score was reached of 0.90. Over the test data the f1 was 0.87.



## Classification Model

We created classification model to classify dices with different numbers. This is created with the Convolutional Neural Network(CNN).
We started with 1 convolution layer. Below is the architecture of first model.

[first model](visuals/classification/classification model architecture.png)

This is how it is performing over each epoc.

[accuracy loss graph](visuals/classification/img2.png)

We see from the begining only accuracy was quite high and loss is almost zero for validation data. Which doesn't seem correct in normal case.
So we added 1 more convolution layer with max pooling layer after it.

[model architecture](visuals/classification/classification model architecture.png)

We have used 2 convolutional layers with max pooling layer. we also added dropout layer to tackle with overfitting.
This is how it is performing over each epoc.

[accuracy loss graph](visuals/classification/classification_graph.png)




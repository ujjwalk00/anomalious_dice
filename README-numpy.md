# Numpy approach

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

metric|score
:--------------------------:|:--------------------------:
f1|0.9051089462333606
Accuracy|0.9051724137931034
ROC|0.9053571428571427
rand_score|0.8268365817091454

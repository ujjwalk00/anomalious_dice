# Numpy approach

For this approach we relied heavily on the thresholds calculated in the previous chapter.
When a prediction is made on a sample, that sample is compared to each of the thresholds. 
If it falls within the boundaries of one of these it is calculated as a normal sample.

But when a sample like this one is given: 

![](processed_data/test_set/ano/17_11_21_anomalies_005.png)

![](visuals/templates/1.png)|![](visuals/templates/2.png)|![](visuals/templates/3.png)|![](visuals/templates/4.png)|![](visuals/templates/5.png)|![](visuals/templates/6.png)
:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
65.5125 > 62.0278047        |75.4151 > 60.43002574       |67.5423 >66.3072560999      |72.4521 > 69.9328087        |77.5124 > 74.3094147        |91.542 > 82.750025  
MSEloss > theshold 1        |MSEloss > theshold 2        |MSEloss > theshold 3        |MSEloss > theshold 4        |MSEloss > theshold 5        |MSEloss > theshold 6  

And MSEloss for this sample falls outside of the boundaries for each category it is
classified as an anomaly.

Over the training data, an f1-score was reached of 0.90. Over the test data the f1 was 0.87.


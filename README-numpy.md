# Numpy approach

For this approach we relied heavily on the thresholds calculated in the previous chapter.
When a prediction is made on a sample, that sample is compared to each of the thresholds. 
If it falls within the boundaries of one of these it is calculated as a normal sample.

But when a sample like this one is given: 

![](processed_data/test_set/ano/17_11_21_anomalies_003.png)

![](visuals/templates/1.png)|![](visuals/templates/2.png)|![](visuals/templates/3.png)|![](visuals/templates/4.png)|![](visuals/templates/5.png)|![](visuals/templates/6.png)
:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:
MSEloss > theshold 1        |MSEloss > theshold 2        |MSEloss > theshold 3        |MSEloss > theshold 4        |MSEloss > theshold 5        |MSEloss > theshold 6  

And MSEloss for this sample falls outside of the boundaries for each category it is
classified as an anomaly.

Over the training data, an f1-score was reached of 0.90. Over the test data the f1 was 0.87.


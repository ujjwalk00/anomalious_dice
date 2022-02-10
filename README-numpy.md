# Numpy approach

For this approach we relied heavily on the thresholds calculated in the previous chapter.
When a prediction is made on a sample, that sample is compared to each of the thresholds. 
If it falls within the boundaries of one of these it is calculated as a normal sample.

But when a sample is given that falls outside of the boundaries for each category it is
classified as an anomaly.

Over the training data, an f1-score was reached of 0.90. Over the test data the f1 was 0.87.


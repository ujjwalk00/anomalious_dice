- numPy approach with original data

per category (reduced from 10 to 6) we now calculate the average difference in pixels where the difference is larger than a 
threshold. These differences are in turn used to set a new threshold between anomaly and normal state.

thresholds for determining a faulty pixel per category when matching the template to a sample
{
    1: 50,
    2: 50,
    3: 50,
    4: 50,
    5: 50,
    6: 25
}

threshold for differnces, lower is normal, higher is abnormal. The average of these is used.:
{1: 8342.193731307994,
 2: 9245.507973570555,
 3: 9124.096731898671,
 4: 9812.492146573579,
 5: 9487.44521608403,
 6: 9691.586149802533,
 7: 10578.044685076677,
 8: 11433.220490513479,
 9: 11418.693099662218,
 10: 11457.256992088007}

- numPy approach with cropped and rotated data.

per category (reduced from 10 to 6) we now calculate the average difference in pixels where the difference is larger than a 
threshold. These differences are in turn used to set a new threshold between anomaly and normal state.

thresholds for determining a faulty pixel per category when matching the template to a sample
{
    1: 35,
    2: 25,
    3: 50,
    4: 75,
    5: 75,
    6: 100
}

threshold for differnces, lower is normal, higher is abnormal. The average of these is used.:
{1: 5303.65006181887,
 2: 6062.10768472119,
 3: 6069.470310742332,
 4: 6194.754365767376,
 5: 6228.959007811023,
 6: 6506.915903462124}

f1 score = 0.44
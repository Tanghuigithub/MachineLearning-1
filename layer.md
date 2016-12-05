# layer

### Batch Normalization

This layer computes Batch Normalization described in [1]. For
each channel in the data (i.e. axis 1), it subtracts the mean and divides
by the variance, where both statistics are computed across both spatial
dimensions and across the different examples in the batch.
*
By default, during training time, the network is computing global mean/
variance statistics via a running average, which is then used at test
time to allow deterministic outputs for each input. You can manually
toggle whether the network is accumulating or using the statistics via the
use_global_stats option. IMPORTANT: for this feature to work, you MUST
set the learning rate to zero for all three parameter blobs, i.e.,
param {lr_mult: 0} three times in the layer definition.

### Crop

[介绍](https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8)
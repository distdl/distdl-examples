This is a DistDL implementation of the ResNet series of networks, described here: https://arxiv.org/pdf/1512.03385.pdf

The DistDL implementation is based on the TorchVision implementation:  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

Loading pre-computed models is not currently supported.

The current example is designed to train on the Intel Image Classification challege.  This is largely because the data set is small and this is only an example.  Many real-world applications ahve terabytes of image data for training.  Just as a demonstration, we artificially upscale the input images.

To run, download the data from here (https://www.kaggle.com/puneet6060/intel-image-classificationhttps://neurohive.io/en/popular-networks/vgg16/) and populate the `intel_challenge` directory as shown by the experiment scripts.

You can supply a different partitioning of the input tensors in `network.py` by supplying a different `parts` parameter.  The number of MPI processes should be at least as much as `parts[0] * parts[1]`.
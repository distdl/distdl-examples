This is a DistDL implementation of the VGG series of networks, described here: https://neurohive.io/en/popular-networks/vgg16/

The current implementation runs through variant "D".  The DistDL implementation is based on the TorchVision implementation:  https://github.com/pytorch/vision/blob/v0.5.1/torchvision/models/vgg.py

Loading pre-computed models is not currently supported.

The current example is designed to train on the Intel Image Classification challege.  This is largely because the data set is small and this is only an example.  Many real-world applications ahve terabytes of image data for training.

To run, download the data from here (https://www.kaggle.com/puneet6060/intel-image-classificationhttps://neurohive.io/en/popular-networks/vgg16/) and populate the `intel_challenge` directory as shown by the experiment scripts.

To run in parallel, this implementation requires a square number of processors.

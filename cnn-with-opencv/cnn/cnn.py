"""
Convolutional neural networks are used to solve classification, object detection, and tracking problems on images.

What is a Convolution Operation?

It detects features such as edges or convex shapes. For instance, if our image is a dog, a feature detector can identify specific features like the dog's ear or tail.

Feature map = convolution. It involves the element-wise multiplication of matrices.

It involves sliding over the image.

As a result of these operations, the original image size is reduced, which is crucial for the model's efficiency.

We create multiple feature maps because we use multiple feature detectors (filters).

Activation Function

After the convolutional layer, we use the ReLU activation function. This function breaks linearity and enables the model to learn nonlinear structures (sets negative values to 0).
Pixel Padding

As convolutional layers continue, the image size will decrease faster than desired. In the initial layers of our network, we aim to preserve as much information about the original input size as possible in order to extract low-level features effectively. Therefore, we apply pixel padding.
Pooling

Performs down-sampling or sub-sampling (reduces the number of parameters).

Enables the detection of features that are invariant to scale or orientation changes.

Reduces the number of parameters and computations in the network, thus controlling overfitting.

Flattening

Converts two-dimensional data into a vector.
Fully Connected

Neurons in a layer are connected to all activations in the previous layer. For example, artificial neural networks.

Performs the classification process.

Dropout

A technique where randomly selected neurons are ignored during training. It prevents overfitting.
Data Augmentation

To avoid overfitting, we need to artificially expand our dataset.

We can alter the training data with minor transformations to recreate variations.

"""
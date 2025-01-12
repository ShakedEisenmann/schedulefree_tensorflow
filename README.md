# Schedule-Free Optimizers in TensorFlow
## TensorFlow implementations of the [Schedule-Free optimizers](https://github.com/facebookresearch/schedule_free/tree/main) originally introduced by Meta Platforms, Inc. and affiliates.

Schedule-Free learning eliminates the need for a learning rate schedule, improving training efficiency. 

For more details, please visit the [original README by Meta Platforms, Inc.](https://github.com/facebookresearch/schedule_free/blob/main/README.md).

Note: This implementation is based on the PyTorch version, adapted for TensorFlow by Shaked Eisenmann in 2025.

The LICENSE file is located in the root directory.

## Requirements

This implementation was developed using Python 3.10 and TensorFlow 2.12.1. These specific versions are required for the code to work as intended.



# Basic MNIST Example

```bash
pip install -r requirements.txt
python -m mnist_example.main
```


### AdamWScheduleFree - TensorFlow

```
Test set: Average loss: 0.0328, Accuracy: 9884/10000 (98.84%)
Test set: Average loss: 0.0271, Accuracy: 9911/10000 (99.11%)
Test set: Average loss: 0.0218, Accuracy: 9924/10000 (99.24%)
Test set: Average loss: 0.0193, Accuracy: 9937/10000 (99.37%)
Test set: Average loss: 0.0199, Accuracy: 9935/10000 (99.35%)
Test set: Average loss: 0.0172, Accuracy: 9943/10000 (99.43%)
Test set: Average loss: 0.0171, Accuracy: 9944/10000 (99.44%)
Test set: Average loss: 0.0163, Accuracy: 9949/10000 (99.49%)
Test set: Average loss: 0.0174, Accuracy: 9945/10000 (99.45%)
Test set: Average loss: 0.0199, Accuracy: 9943/10000 (99.43%)
Test set: Average loss: 0.0221, Accuracy: 9944/10000 (99.44%)
Test set: Average loss: 0.0210, Accuracy: 9946/10000 (99.46%)
Test set: Average loss: 0.0224, Accuracy: 9944/10000 (99.44%)
Test set: Average loss: 0.0204, Accuracy: 9948/10000 (99.48%)
```

### AdamW - Default TensorFlow Implementation

```
Test set: Average loss: 0.0465, Accuracy: 9855/10000 (98.55%)
Test set: Average loss: 0.0336, Accuracy: 9887/10000 (98.87%)
Test set: Average loss: 0.0291, Accuracy: 9894/10000 (98.94%)
Test set: Average loss: 0.0256, Accuracy: 9910/10000 (99.10%)
Test set: Average loss: 0.0250, Accuracy: 9920/10000 (99.20%)
Test set: Average loss: 0.0242, Accuracy: 9918/10000 (99.18%)
Test set: Average loss: 0.0247, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0232, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0222, Accuracy: 9934/10000 (99.34%)
Test set: Average loss: 0.0235, Accuracy: 9932/10000 (99.32%)
Test set: Average loss: 0.0299, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0240, Accuracy: 9929/10000 (99.29%)
Test set: Average loss: 0.0228, Accuracy: 9926/10000 (99.26%)
Test set: Average loss: 0.0208, Accuracy: 9930/10000 (99.30%)
```

### AdamWScheduleFree - PyTorch (Results based on those found [here](https://github.com/facebookresearch/schedule_free/tree/main/examples/mnist))

```
Test set: Average loss: 0.0367, Accuracy: 9873/10000 (98.73%)
Test set: Average loss: 0.0288, Accuracy: 9896/10000 (98.96%)
Test set: Average loss: 0.0273, Accuracy: 9907/10000 (99.07%)
Test set: Average loss: 0.0248, Accuracy: 9926/10000 (99.26%)
Test set: Average loss: 0.0257, Accuracy: 9930/10000 (99.30%)
Test set: Average loss: 0.0268, Accuracy: 9929/10000 (99.29%)
Test set: Average loss: 0.0268, Accuracy: 9921/10000 (99.21%)
Test set: Average loss: 0.0275, Accuracy: 9929/10000 (99.29%)
Test set: Average loss: 0.0279, Accuracy: 9931/10000 (99.31%)
Test set: Average loss: 0.0278, Accuracy: 9933/10000 (99.33%)
Test set: Average loss: 0.0274, Accuracy: 9935/10000 (99.35%)
Test set: Average loss: 0.0278, Accuracy: 9936/10000 (99.36%)
Test set: Average loss: 0.0289, Accuracy: 9938/10000 (99.38%)
Test set: Average loss: 0.0304, Accuracy: 9935/10000 (99.35%)
```
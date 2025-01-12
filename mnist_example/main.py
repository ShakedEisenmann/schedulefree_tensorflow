# This file is adapted from the original PyTorch implementation by Meta Platforms, Inc. and affiliates.  
# The corresponding file in the original project can be found at:  
# https://github.com/facebookresearch/schedule_free/blob/main/examples/mnist/main.py  
#  
# This TensorFlow version was created by Shaked Eisenmann in 2025.  
#  
# This source code is licensed under the license found in the LICENSE file  
# located in the root directory of this source tree.  

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow 2.12.1 info and warning messages

import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from schedulefree_tensorflow.adamw_schedulefree import AdamWScheduleFree

print('Starting TensorFlow MNIST Example')

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    return parser.parse_args()

def train(model, optimizer, train_dataset, epoch, log_interval):
    optimizer.train(model.trainable_variables)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    for batch_idx, (data, target) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            output = model(data, training=True)
            loss = loss_fn(target, output)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if batch_idx != 0 and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataset)*len(data)}] Loss: {loss.numpy():.6f}')

def test(model, optimizer, test_dataset):
    optimizer.eval(model.trainable_variables)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    test_loss = 0
    correct = 0
    for data, target in test_dataset:
        output = model(data, training=False)
        test_loss += loss_fn(target, output).numpy() # sum up batch loss
        pred = np.argmax(output.numpy(), axis=1)  # get the index of the max log-probability
        correct += np.sum(pred == target.numpy())
    test_loss /= len(test_dataset)
    accuracy = 100. * correct / (len(test_dataset) * len(data))
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataset)*len(data)} ({accuracy:.2f}%)')

def main():

    args = parse_args()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Scale to [0, 1] and normalize with mean 0.1307 and std 0.3081
    mean = 0.1307
    std = 0.3081

    train_images = (train_images[..., np.newaxis] / 255.0 - mean) / std
    test_images = (test_images[..., np.newaxis] / 255.0 - mean) / std

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)) \
        .shuffle(10000).batch(args.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)) \
        .batch(args.test_batch_size)

    # Initialize model and optimizer
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    optimizer = AdamWScheduleFree(learning_rate=args.lr)

    model.compile(optimizer=optimizer)

    optimizer.build(model.trainable_variables)

    # Training and Testing loop
    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        train(model, optimizer, train_dataset, epoch, log_interval=args.log_interval)
        test(model, optimizer, test_dataset)

    if args.save_model:
        model.save("/tmp/mnist_cnn.h5")

if __name__ == '__main__':
    main()
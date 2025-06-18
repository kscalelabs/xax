#!/usr/bin/env python3
"""
Standalone MNIST example with JAX profiling.
This example demonstrates JAX profiling without requiring the XAX framework.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from jax.profiler import start_trace, stop_trace
import os
import time
from pathlib import Path
import contextlib

# Simple CNN model for MNIST
class CNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    
    def __init__(self, *, key):
        conv1_key, conv2_key, linear1_key, linear2_key = jax.random.split(key, 4)
        # Equinox Conv2d expects input shape [H, W, C_in]
        self.conv1 = eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, key=conv1_key)
        self.conv2 = eqx.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, key=conv2_key)
        self.linear1 = eqx.nn.Linear(9216, 128, key=linear1_key)
        self.linear2 = eqx.nn.Linear(128, 10, key=linear2_key)
    
    def __call__(self, x):
        # Input shape: [H, W, C_in]
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.max_pool(x, (2, 2), strides=(2, 2))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.max_pool(x, (2, 2), strides=(2, 2))
        # Flatten for the linear layer
        x = x.reshape(-1)
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)

# Configuration
batch_size = 64
learning_rate = 0.001
num_epochs = 1  # Reduced for faster execution
random_seed = 42

# A decorator for profiling
def profile(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

@profile("compute_loss")
def compute_loss(model, x, y):
    # In JAX's equinox, Conv2d expects input shape [H, W, C_in], but our data is [N, C_in, H, W]
    # We need to move the batch dimension outside for vmap
    def apply_model(single_x):
        # Reshape from [C_in, H, W] to [H, W, C_in]
        x_hwc = single_x.transpose(1, 2, 0)
        return model(x_hwc)
    
    # Use vmap to apply the model to each item in the batch
    logits = jax.vmap(apply_model)(x)
    one_hot = jax.nn.one_hot(y, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == y)
    return loss, accuracy

@profile("train_step")
def train_step(model, opt_state, optimizer, x, y):
    def loss_fn(model):
        loss, accuracy = compute_loss(model, x, y)
        return loss, (loss, accuracy)
    
    (loss, (loss_val, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss_val, accuracy

@profile("validation_step")
def validation_step(model, x, y):
    loss, accuracy = compute_loss(model, x, y)
    return loss, accuracy

def load_data():
    # Create synthetic MNIST-like data
    print("Creating synthetic MNIST data...")
    key = jax.random.PRNGKey(random_seed)
    key1, key2 = jax.random.split(key)
    
    # Create training data: 6000 images of size 28x28 (reduced size for quicker testing)
    # The data shape is [N, C, H, W] where C=1 for grayscale, H=W=28 for MNIST
    x_train = jax.random.normal(key1, (600, 1, 28, 28))  # Reduced size for faster execution
    y_train = jax.random.randint(key1, (600,), 0, 10)
    
    # Create test data: 1000 images of size 28x28
    x_test = jax.random.normal(key2, (100, 1, 28, 28))  # Reduced size for faster execution
    y_test = jax.random.randint(key2, (100,), 0, 10)
    
    # Convert to numpy for easier handling
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def main():
    # Create profile directory
    profile_dir = Path("./profiles/mnist_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"Profile directory: {os.path.abspath(profile_dir)}")
    
    # Initialize model and optimizer
    key = jax.random.PRNGKey(random_seed)
    model = CNN(key=key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # JIT-compile the steps
    jit_train_step = jax.jit(lambda m, o, x, y: train_step(m, o, optimizer, x, y))
    jit_validation_step = jax.jit(validation_step)
    
    # Profile key training steps (not the entire training)
    print("\nRunning training with profiling for selected steps...")
    
    # Training loop
    steps_per_epoch = len(x_train) // batch_size
    step = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]
        
        # Training
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Profile specific steps
            if step % 5 == 0:  # Profile every 5 steps
                # Perform separate profiling for each step we want to profile
                print(f"\nStarting profiling for step {step}...")
                sub_profile_dir = profile_dir / f"train_step_{step}"
                sub_profile_dir.mkdir(parents=True, exist_ok=True)
                
                # Start the trace for this step
                start_trace(str(sub_profile_dir))
                
                # Run the step
                model, opt_state, loss, accuracy = jit_train_step(
                    model, opt_state, x_batch, y_batch
                )
                
                # Stop the trace
                stop_trace()
                print(f"Profiling completed for step {step}")
            else:
                model, opt_state, loss, accuracy = jit_train_step(
                    model, opt_state, x_batch, y_batch
                )
            
            # Log training progress
            if step % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}, "
                      f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            step += 1
        
        # Validation
        print("\nStarting validation profiling...")
        val_profile_dir = profile_dir / f"validation_epoch_{epoch}"
        val_profile_dir.mkdir(parents=True, exist_ok=True)
        start_trace(str(val_profile_dir))
        
        val_losses, val_accuracies = [], []
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size]
            
            loss, accuracy = jit_validation_step(model, x_batch, y_batch)
            
            val_losses.append(loss)
            val_accuracies.append(accuracy)
        
        # Stop validation profiling
        stop_trace()
        print("Validation profiling completed")
        
        # Log validation results
        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        print(f"Validation - Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")
    
    print("\nAll profiling completed!")
    
    print(f"\nProfiles saved to: {profile_dir}")
    print("\nTo view the profiles, you can:")
    print("1. Use Perfetto UI:")
    print("   Go to https://ui.perfetto.dev")
    print("   Click 'Open trace file' and select any of the .trace.json.gz files in the profile directories")
    print("\n2. Use Chrome Tracing:")
    print("   Open Chrome and go to chrome://tracing")
    print("   Click 'Load' and select any of the .trace.json.gz files")

if __name__ == "__main__":
    main() 
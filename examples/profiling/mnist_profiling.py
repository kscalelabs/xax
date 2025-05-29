"""MNIST example with profiling enabled."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import contextlib
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Iterator, Optional, Callable, Generator

from xax.core.state import State
from xax.task.script import Script, ScriptConfig
from xax.utils.profiling import annotate

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

@dataclass
class MNISTConfig(ScriptConfig):
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 5
    random_seed: int = 42
    
    # Enable profiling by default
    enable_profiling: bool = True
    profile_every_n_steps: int = 10
    profile_duration_ms: int = 5000

class MNISTTask(Script[MNISTConfig]):
    def __init__(self, config: MNISTConfig):
        super().__init__(config)
        self.model = None
        self.optimizer = None
    
    def setup(self):
        # Initialize the model and optimizer
        key = jax.random.PRNGKey(self.config.random_seed)
        self.model = CNN(key=key)
        self.optimizer = optax.adam(self.config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
    
    def load_data(self):
        # Create synthetic MNIST-like data
        print("Creating synthetic MNIST data...")
        key = jax.random.PRNGKey(self.config.random_seed)
        key1, key2 = jax.random.split(key)
        
        # Create training data: 6000 images of size 28x28 (reduced size for quicker testing)
        # The data shape is [N, C, H, W] where C=1 for grayscale, H=W=28 for MNIST
        x_train = jax.random.normal(key1, (6000, 1, 28, 28))
        y_train = jax.random.randint(key1, (6000,), 0, 10)
        
        # Create test data: 1000 images of size 28x28
        x_test = jax.random.normal(key2, (1000, 1, 28, 28))
        y_test = jax.random.randint(key2, (1000,), 0, 10)
        
        # Convert to numpy for easier handling
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    @annotate("compute_loss")
    def compute_loss(self, model, x, y):
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
    
    @annotate("train_step")
    def train_step(self, model, opt_state, x, y):
        def loss_fn(model):
            loss, accuracy = self.compute_loss(model, x, y)
            return loss, (loss, accuracy)
        
        (loss, (loss_val, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss_val, accuracy
    
    @annotate("validation_step")
    def validation_step(self, model, x, y):
        loss, accuracy = self.compute_loss(model, x, y)
        return loss, accuracy
    
    def run(self):
        self.setup()
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        # Create JIT-compiled versions of steps
        jit_train_step = jax.jit(self.train_step)
        jit_validation_step = jax.jit(self.validation_step)
        
        # Training loop
        steps_per_epoch = len(x_train) // self.config.batch_size
        step = 0
        
        for epoch in range(self.config.num_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Training
            for i in range(0, len(x_train), self.config.batch_size):
                x_batch = x_train_shuffled[i:i+self.config.batch_size]
                y_batch = y_train_shuffled[i:i+self.config.batch_size]
                
                # Create a synthetic State for profiling
                state = State.from_dict(num_steps=step, phase="train")
                
                # Profile the training step if needed
                if self.should_run_profiler(state):
                    with self.profile_context(state, name=f"epoch_{epoch}_batch_{i//self.config.batch_size}"):
                        self.model, self.opt_state, loss, accuracy = jit_train_step(
                            self.model, self.opt_state, x_batch, y_batch
                        )
                else:
                    self.model, self.opt_state, loss, accuracy = jit_train_step(
                        self.model, self.opt_state, x_batch, y_batch
                    )
                
                # Log training progress
                if i % (10 * self.config.batch_size) == 0:
                    try:
                        self.logger.log_scalar("train/loss", loss)
                        self.logger.log_scalar("train/accuracy", accuracy)
                    except:
                        pass  # Ignore logging errors
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {step}, "
                          f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
                step += 1
            
            # Validation
            val_losses, val_accuracies = [], []
            for i in range(0, len(x_test), self.config.batch_size):
                x_batch = x_test[i:i+self.config.batch_size]
                y_batch = y_test[i:i+self.config.batch_size]
                
                # Profile validation occasionally
                if i == 0:
                    state = State.from_dict(num_steps=step, phase="valid")
                    with self.profile_context(state, name=f"validation_epoch_{epoch}"):
                        loss, accuracy = jit_validation_step(self.model, x_batch, y_batch)
                else:
                    loss, accuracy = jit_validation_step(self.model, x_batch, y_batch)
                
                val_losses.append(loss)
                val_accuracies.append(accuracy)
            
            # Log validation results
            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracies)
            try:
                self.logger.log_scalar("val/loss", avg_val_loss)
                self.logger.log_scalar("val/accuracy", avg_val_accuracy)
            except:
                pass  # Ignore logging errors
            print(f"Validation - Epoch {epoch+1}/{self.config.num_epochs}, "
                  f"Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")

if __name__ == "__main__":
    config = MNISTConfig()
    task = MNISTTask(config)
    task.run() 
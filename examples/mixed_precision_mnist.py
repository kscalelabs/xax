"""MNIST example with mixed precision training."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from dataclasses import dataclass
from typing import Tuple

from xax.core.state import State
from xax.task.script import Script, ScriptConfig
from xax.utils.mixed_precision import Policy, get_default_half_dtype

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
    
    # Enable mixed precision training
    enable_mixed_precision: bool = True
    
    # Mixed precision policy (options: "mixed", "float16", "default", 
    # or custom like "params=float32,compute=float16,output=float32")
    precision_policy: str = "mixed"
    
    # Loss scaling approach ("none", "static", or "dynamic")
    loss_scaling: str = "dynamic"
    
    # Initial scale for loss scaling
    loss_scale_value: float = 2**15
    
    # Skip updates with non-finite gradients
    skip_nonfinite_updates: bool = True

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
        
        # Print mixed precision info
        if self.config.enable_mixed_precision:
            print(f"Mixed precision training enabled with policy: {self.policy}")
            print(f"Using {self.loss_scale.__class__.__name__} with initial scale: {self.loss_scale.loss_scale}")
            print(f"Compute dtype: {self.policy.compute_dtype}")
            print(f"Parameter dtype: {self.policy.param_dtype}")
            print(f"Output dtype: {self.policy.output_dtype}")
    
    def load_data(self):
        # Create synthetic MNIST-like data for demonstration
        print("Creating synthetic MNIST data...")
        key = jax.random.PRNGKey(self.config.random_seed)
        key1, key2 = jax.random.split(key)
        
        # Create training data: 6000 images of size 28x28
        # The data shape is [N, H, W, C] where C=1 for grayscale
        x_train = jax.random.normal(key1, (6000, 28, 28, 1))
        y_train = jax.random.randint(key1, (6000,), 0, 10)
        
        # Create test data: 1000 images of size 28x28
        x_test = jax.random.normal(key2, (1000, 28, 28, 1))
        y_test = jax.random.randint(key2, (1000,), 0, 10)
        
        # Convert to numpy for easier handling
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        print(f"Data shapes: x_train={x_train.shape}, y_train={y_train.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def compute_loss(self, model, x, y):
        # Forward pass
        logits = jax.vmap(model)(x)
        one_hot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == y)
        return loss, accuracy
    
    def train_step(self, model, opt_state, x, y):
        # Define loss function with mixed precision support
        def loss_fn(model):
            # Cast model and inputs to compute precision
            if self.config.enable_mixed_precision:
                model = self.cast_params_to_compute(model)
            
            # Compute loss
            loss, accuracy = self.compute_loss(model, x, y)
            
            # Scale loss for mixed precision training
            if self.config.enable_mixed_precision:
                loss = self.scale_loss(loss)
                
            return loss, (loss, accuracy)
        
        # Compute gradients
        (_, (loss, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
        
        # Handle mixed precision updates
        if self.config.enable_mixed_precision:
            # Unscale gradients
            grads = self.unscale_grads(grads)
            
            # Check if gradients are finite
            grads_finite = self.check_grads_finite(grads)
            
            # Apply optimizer update function
            def optimizer_update(params, opt_state):
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = eqx.apply_updates(params, updates)
                return new_params, new_opt_state
            
            # Apply mixed precision update
            model, opt_state, new_loss_scale = self.mixed_precision_update(
                model, grads, optimizer_update, opt_state
            )
            
            # Update loss scale
            self.set_loss_scale(new_loss_scale)
        else:
            # Standard update
            updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            opt_state = new_opt_state
        
        # Cast model back to parameter precision if using mixed precision
        if self.config.enable_mixed_precision:
            model = self.cast_params_to_storage(model)
            
        return model, opt_state, loss, accuracy
    
    def validation_step(self, model, x, y):
        # Cast model to compute precision for validation
        if self.config.enable_mixed_precision:
            model = self.cast_params_to_compute(model)
            
        loss, accuracy = self.compute_loss(model, x, y)
        return loss, accuracy
    
    def run(self):
        self.setup()
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        # Create JIT-compiled versions of steps
        jit_train_step = jax.jit(self.train_step)
        jit_validation_step = jax.jit(self.validation_step)
        
        # Training loop
        step = 0
        
        for epoch in range(self.config.num_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Training
            epoch_losses = []
            epoch_accuracies = []
            
            for i in range(0, len(x_train), self.config.batch_size):
                x_batch = x_train_shuffled[i:i+self.config.batch_size]
                y_batch = y_train_shuffled[i:i+self.config.batch_size]
                
                # Create a synthetic State for consistent interface
                state = State.from_dict(num_steps=step, phase="train")
                
                self.model, self.opt_state, loss, accuracy = jit_train_step(
                    self.model, self.opt_state, x_batch, y_batch
                )
                
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
                
                # Log training progress
                if i % (10 * self.config.batch_size) == 0:
                    try:
                        self.logger.log_scalar("train/loss", loss)
                        self.logger.log_scalar("train/accuracy", accuracy)
                        if self.config.enable_mixed_precision:
                            self.logger.log_scalar("train/loss_scale", float(self.loss_scale.loss_scale))
                    except:
                        pass  # Ignore logging errors
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {step}, "
                          f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
                step += 1
            
            # Calculate epoch averages
            avg_train_loss = np.mean(epoch_losses)
            avg_train_accuracy = np.mean(epoch_accuracies)
            print(f"Training - Epoch {epoch+1}/{self.config.num_epochs}, "
                  f"Avg Loss: {avg_train_loss:.4f}, Avg Accuracy: {avg_train_accuracy:.4f}")
            
            # Validation
            val_losses, val_accuracies = [], []
            for i in range(0, len(x_test), self.config.batch_size):
                x_batch = x_test[i:i+self.config.batch_size]
                y_batch = y_test[i:i+self.config.batch_size]
                
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
            
            # Report current loss scale
            if self.config.enable_mixed_precision:
                print(f"Current loss scale: {float(self.loss_scale.loss_scale)}")

        # Report final model parameter data types
        if self.config.enable_mixed_precision:
            def print_dtypes(x):
                if isinstance(x, (jnp.ndarray, jax.Array)):
                    return x.dtype
                return None
            
            print("\nFinal model parameter dtypes:")
            dtypes = jax.tree_util.tree_map(print_dtypes, self.model)
            
            # Print some sample dtypes
            flat_dtypes = jax.tree_util.tree_leaves(dtypes)
            dtype_counts = {}
            for dt in flat_dtypes:
                if dt is not None:
                    dtype_counts[str(dt)] = dtype_counts.get(str(dt), 0) + 1
            
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} parameters")

if __name__ == "__main__":
    config = MNISTConfig()
    task = MNISTTask(config)
    task.run() 
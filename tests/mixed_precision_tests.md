# Mixed Precision Tests

This directory contains tests for mixed precision functionality in XAX. The tests validate the core utilities and training capabilities with mixed precision.

## Test Files

- **test_mixed_precision.py**: Tests for core mixed precision utilities
- **test_mixed_precision_train.py**: Tests for mixed precision training capabilities

## Running the Tests

You can run the tests using pytest:

```bash
# Run all mixed precision tests
python3 -m pytest xax/tests/test_mixed_precision*.py -v

# Run specific test files
python3 -m pytest xax/tests/test_mixed_precision.py -v
python3 -m pytest xax/tests/test_mixed_precision_train.py -v

# Run a specific test class
python3 -m pytest xax/tests/test_mixed_precision.py::TestPolicies -v
```

## Test Coverage

### Core Utilities (test_mixed_precision.py)

Tests for the core mixed precision utilities in `xax.utils.mixed_precision`:

1. **Precision Policies**:
   - Parsing policy strings ("default", "mixed", "float16", custom)
   - Tree mapping for dtype conversion 
   - Parameter casting between precision modes

2. **Loss Scaling**:
   - No-op loss scaling
   - Static loss scaling
   - Dynamic loss scaling with adjustment

3. **Mixed Precision Mixin**:
   - Policy creation
   - Loss scale creation
   - Mixed precision methods
   - Training with mixed precision

### Training (test_mixed_precision_train.py)

Tests for mixed precision training capabilities:

1. **Training Comparison**:
   - Comparing standard vs. mixed precision training results
   - Verifying similar loss and prediction accuracy

2. **Loss Scaling**:
   - Testing loss scaling and unscaling
   - Testing dynamic loss scale adjustment

3. **Precision Policies**:
   - Testing parameter casting for models
   - Verifying correct dtype conversions

## Expected Results

All tests should pass, validating that:

1. Mixed precision utilities function correctly
2. Loss scaling works as expected for gradient stability
3. Mixed precision training produces results comparable to full precision
4. Policy and dtype conversions maintain numerical stability

Any test failures indicate potential issues with mixed precision functionality. 
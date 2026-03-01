# Transformer Project Testing Guide

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ -v --cov=. --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/test_dataset.py -v
```

### Run specific test function
```bash
pytest tests/test_dataset.py::test_contract_dataset_loading -v
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── test_dataset.py       # Data loading tests
├── test_model.py         # Model tests
├── test_tokenizer.py     # Tokenizer tests
├── test_metrics.py       # Metrics tests
└── test_training.py      # Training tests
```

## Writing New Tests

### Example test:
```python
def test_my_feature():
    # Arrange
    data = load_test_data()
    
    # Act
    result = my_function(data)
    
    # Assert
    assert result is not None
    assert len(result) > 0
```

### Using fixtures:
```python
def test_with_model(sample_model):
    # sample_model is provided by conftest.py
    output = sample_model(input_data)
    assert output.shape == expected_shape
```

## Coverage Reports

After running tests with coverage, open `htmlcov/index.html` to view detailed coverage report.

## Continuous Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Manual workflow dispatch

See `.github/workflows/ci.yml` for CI configuration.

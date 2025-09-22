# Tests

## Overview

This directory contains all test files for the Multi-Language Sentiment Engine project. The tests are organized to cover all core functionality of the sentiment analysis system.

## Test Structure

- `test_core.py` - Core functionality tests including sentiment analysis and model operations
- `__init__.py` - Test package initialization

## Running Tests

### Prerequisites

Ensure you have pytest installed:
```bash
pip install pytest
```

### Running All Tests

From the project root directory:
```bash
pytest src/tests/
```

### Running Specific Tests

To run specific test files:
```bash
pytest src/tests/test_core.py
```

To run specific test methods:
```bash
pytest src/tests/test_core.py::test_sentiment_analysis
```

## Test Coverage

The test suite covers:
- Sentiment analysis functionality
- Model loading and prediction
- API endpoint responses
- Data preprocessing utilities
- Multi-language support
- Error handling

## Writing New Tests

When adding new features, please include corresponding tests:
1. Follow pytest naming conventions (`test_*.py`)
2. Use descriptive test method names
3. Include both positive and negative test cases
4. Add docstrings to test methods
5. Mock external dependencies when necessary

## Test Configuration

Test configuration and fixtures are defined in `conftest.py` (when created).

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Merges to master branch
- Scheduled builds

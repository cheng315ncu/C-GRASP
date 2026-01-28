# Contributing to CGRASP

Thank you for your interest in contributing to CGRASP! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use the issue templates when available
- Provide detailed reproduction steps for bugs
- Include system information (OS, Python version, GPU, etc.)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install development dependencies**: `pip install -r requirements.txt`
3. **Make your changes** with clear, descriptive commits
4. **Test your changes** thoroughly
5. **Update documentation** if needed
6. **Submit a pull request** with a clear description

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and reasonably sized

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues when relevant (#123)

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CGRASP.git
cd CGRASP

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create test data directory
mkdir -p data clinical_pdfs outputs
```

## Project Structure

Key files for contributors:

- `config.py` - Configuration and parameters
- `main.py` - Main entry point
- `inference_steps.py` - Step 1-7 inference logic
- `pikerag_medical_integration.py` - RAG implementation
- `utils.py` - Utility functions

## Areas for Contribution

- **Documentation**: Improve README, add tutorials
- **Testing**: Add unit tests, integration tests
- **Features**: New analysis steps, output formats
- **Performance**: Optimization, memory efficiency
- **Models**: Support for additional LLMs

## Questions?

Feel free to open an issue for questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

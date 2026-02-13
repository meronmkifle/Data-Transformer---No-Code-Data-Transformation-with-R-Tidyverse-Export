# Contributing to Data Transformer

Thanks for your interest in contributing! Here's how you can help.

## Reporting Issues

- Use GitHub Issues to report bugs
- Include:
  - OS and Python version
  - Steps to reproduce
  - Expected vs actual behavior
  - Sample data if possible (anonymized)

## Feature Requests

- Open an issue with the label `enhancement`
- Describe the feature and use case
- Include examples if applicable

## Development Setup

```bash
git clone https://github.com/yourusername/data-transformer.git
cd data-transformer
bash setup.sh
source venv/bin/activate
streamlit run data_transformer_app.py
```

## Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

## Testing

Run these checks before submitting:

```bash
# Syntax check
python -m py_compile data_transformer_app.py

# Lint
pip install flake8
flake8 data_transformer_app.py

# Test imports
python -c "import streamlit; import pandas"
```

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and test thoroughly
3. Commit with clear messages: `git commit -m "Add: New transformation type"`
4. Push to GitHub: `git push origin feature/my-feature`
5. Open a Pull Request with:
   - Description of changes
   - Motivation and context
   - Testing done
   - Screenshots if UI changes

## Roadmap

Planned features:
- [ ] Join operations (inner, left, right, full)
- [ ] String manipulation (regex, substring, case)
- [ ] Date/time operations
- [ ] Window functions (lag, lead, rank)
- [ ] Data quality checks
- [ ] SQL backend support
- [ ] Python code generation
- [ ] Undo/redo stack with history view
- [ ] Sharing via URL
- [ ] R server integration

## Questions?

Open a GitHub Discussion or issue. We're here to help!

---

**Thank you for contributing! 🎉**

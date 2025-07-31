# Faculty AI ML Engineer Interview Prep

## Advanced Pandas & Data Science Exercises

This repository contains practice exercises designed to prepare for your 90-minute pair programming interview at Faculty AI.

## Quick Start

### 1. Set up the environment

```bash
# The project uses uv for dependency management
uv sync  # Install dependencies from existing lock file

# Or if starting fresh:
# uv add pandas numpy matplotlib seaborn
```

### 2. Generate practice datasets

```bash
# Test that everything works (includes ML functionality test)
uv run python test_setup.py

# Generate individual datasets (already done if test passes)
cd datasets
uv run python generate_sample_sales.py
uv run python generate_ecommerce_data.py
uv run python generate_sensor_data.py
uv run python generate_ab_test_data.py  # For ML exercises
```

### 3. Start practicing

```bash
# Follow the practice session guide
cat practice_session_guide.md

# Begin with the warm-up exercise
cat exercises/00_warmup_data_exploration.md
```

## Interview Strategy

- **Ask questions first**: Understand the business context before coding
- **Think aloud**: Explain your approach and reasoning
- **Consider edge cases**: Discuss potential data quality issues
- **Optimize iteratively**: Start with a working solution, then improve

## Exercise Structure

- **Warm-up** (15 min): Basic data exploration and communication
- **Core Challenge** (60 min): Complex data manipulation and feature engineering
- **ML Pipeline** (60 min): End-to-end machine learning model development
- **Advanced Topics** (30-45 min): Time series, performance optimization, A/B testing

## Files

- `exercises/` - Practice problems organized by difficulty
- `datasets/` - Sample data for exercises (1M+ rows total)
- `solutions/` - Reference solutions with explanations
- `interview_tips.md` - Communication and collaboration strategies
- `practice_session_guide.md` - Structured practice approach
- `test_setup.py` - Verify environment and datasets

## Key Skills Covered

### Data Engineering & Pandas

- Complex groupby operations with multiple aggregations
- Advanced merging and joining strategies
- Time series analysis and resampling
- Memory optimization techniques
- Data validation and quality checks

### Machine Learning Engineering

- End-to-end ML pipeline development
- Feature engineering and selection
- Model training, validation, and comparison
- Cross-validation and hyperparameter tuning
- Model evaluation and interpretation
- A/B testing for ML model validation
- Production deployment considerations

## Dataset Overview

- **Sample Sales** (1K rows): Warm-up exploration with data quality issues
- **E-commerce** (55K rows): Customer/product/transaction pipeline for feature engineering
- **IoT Sensors** (13M rows): Time series anomaly detection and performance optimization
- **A/B Test Results** (10K rows): ML model evaluation and statistical analysis

Ready to practice? Run `uv run python test_setup.py` to get started! 🚀

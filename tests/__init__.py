"""
Investment Tool Test Suite.

This test suite mirrors the source code structure for easy navigation:

tests/
├── conftest.py                          # Shared pytest fixtures
├── data/
│   ├── repositories/
│   │   └── test_repositories.py         # Repository pattern tests
│   └── validation/
│       └── test_validation.py           # Validation pipeline & validators
├── domain/
│   ├── backtesting/
│   │   └── test_backtester.py           # Backtester metrics & calculations
│   ├── investment_strategy/
│   │   └── test_investment_strategy_sentiment_ewm.py
│   └── model_data/
│       └── entities/
│           ├── test_portfolio.py        # Portfolio entity tests
│           └── test_stock.py            # Stock entity & technical indicators
├── output/
│   ├── test_export.py                   # Output export tests
│   ├── test_metrics.py                  # Metrics tests
│   ├── test_output_manager.py           # Output manager tests
│   ├── test_report.py                   # Report generation tests
│   └── test_strategy_comparator.py      # Strategy comparison tests
├── strategies/
│   ├── optimization_strategies/
│   │   ├── solver_comparison/           # Solver comparison framework tests
│   │   └── test_platypus_optimization.py
│   ├── prediction_strategies/
│   │   ├── prediction_comparison/       # Prediction comparison framework tests
│   │   ├── test_prediction_strategy.py
│   │   └── test_ridge_varx_prediction_strategy.py
│   └── selection_strategies/
│       └── test_selection_strategies.py  # Selection strategy tests
├── test_constants.py                    # Constants & enums tests
├── test_exceptions.py                   # Custom exceptions tests
└── test_utils.py                        # Utility functions tests

Run all tests:
    poetry run pytest tests/ -v

Run specific module tests:
    poetry run pytest tests/domain/backtesting/ -v
    poetry run pytest tests/data/validation/ -v
    poetry run pytest tests/data/repositories/ -v

Run with coverage:
    poetry run pytest tests/ --cov=investment_tool --cov-report=html
"""

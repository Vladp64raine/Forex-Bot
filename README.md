# Enhanced Trading Bot

## Overview
The Enhanced Trading Bot is an algorithmic trading system designed to operate in the Forex market. It utilizes machine learning models and technical analysis to generate trading signals and execute trades through the MetaTrader 5 platform. The bot is built with modular components for easy maintenance and extensibility.

## Project Structure
```
enhanced-trading-bot
├── enhanced_trading_botv2.py  # Main logic for the trading bot
├── trade_log.csv               # Log of trade details
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
```

## Key Components
- **enhanced_trading_botv2.py**: Contains the core functionality of the trading bot, including:
  - **MT5Connection**: Manages the connection to the MetaTrader 5 platform.
  - **FeatureEngineer**: Implements feature engineering techniques to prepare data for model training.
  - **TradingModel**: Defines individual trading models (XGBoost, LightGBM, CatBoost) for making predictions.
  - **EnsembleTradingModel**: Combines predictions from multiple models to improve accuracy and confidence.

- **trade_log.csv**: Automatically generated file that logs trade activities, including:
  - Symbol
  - Order Type (Buy/Sell)
  - Lot Size
  - Entry Price
  - Result (Success/Failure)
  - Profit/Loss
  - Timestamp

- **requirements.txt**: Lists all necessary Python packages for the project, including:
  - `numpy`
  - `pandas`
  - `ta`
  - `MetaTrader5`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `catboost`

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd enhanced-trading-bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that MetaTrader 5 is installed and configured on your machine.

## Usage
To run the trading bot, execute the following command:
```
python enhanced_trading_botv2.py
```

The bot will connect to MetaTrader 5, fetch market data, and begin trading based on the implemented strategies.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
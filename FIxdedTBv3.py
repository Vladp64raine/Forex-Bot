import time
import os
import csv
import numpy as np
import pandas as pd
import ta
import MetaTrader5 as mt5
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import json
from collections import deque
import traceback
import joblib
import sys

# ========================
# ENHANCED CONFIGURATION
# ========================
class Config:
    SYMBOL = 'EURUSD'
    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
    }
    RISK_PARAMS = {
        'max_daily_drawdown': -0.05,
        'max_risk_per_trade': 0.01,
        'atr_sl_multiplier': 1.5,
        'atr_tp_multiplier': 3.0,
        'max_margin_ratio': 0.5,
        'max_spread': 20  # in points
    }
    MODEL_PARAMS = {
        'retrain_interval_hours': 6,
        'feature_count': 25,
        'data_points': 5000,
        'min_calibration_samples': 100,
        'confidence_threshold': 0.65,
        'prob_spread_threshold': 0.25
    }
    TRADE_LOG = "trade_log.csv"
    MODEL_DIR = "models/"

# ========================
# ROBUST UTILITY FUNCTIONS
# ========================
def get_mt5_connection(max_retries=3, retry_delay=5):
    """Establish reliable MT5 connection"""
    if mt5.initialize():
        return mt5
        
    for i in range(max_retries):
        print(f"Connection attempt {i+1}/{max_retries}")
        time.sleep(retry_delay)
        if mt5.initialize():
            return mt5
            
    raise ConnectionError("MT5 initialization failed")

def safe_feature_serialization(features):
    """Handle serialization of complex feature objects"""
    print("Serializing features:", features)
    if features is None:
        return ""
    try:
        # Convert numpy types to native Python types
        if isinstance(features, dict):
            return json.dumps({k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in features.items()})
        return json.dumps(features)
    except (TypeError, ValueError):
        return str(features)

def log_trade(trade_data):
    """Enhanced trade logging with error handling"""
    try:
        trade_data['features'] = safe_feature_serialization(trade_data.get('features'))
        trade_data['timestamp'] = pd.Timestamp.now().isoformat()
        
        file_exists = os.path.exists(Config.TRADE_LOG)
        mode = 'a' if file_exists else 'w'
        
        with open(Config.TRADE_LOG, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trade_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
    except Exception as e:
        print(f"Trade logging failed: {str(e)}")

def fetch_market_data(mt5, symbol, timeframe, bars=10000):
    """Robust data fetching with validation"""
    tf = Config.TIMEFRAMES.get(timeframe)
    if not tf:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"No data received for {timeframe}, retrying...")
        time.sleep(2)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None:
            raise ValueError(f"No data received for {timeframe} after retry")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.set_index('time')

def prune_correlated_features(df, threshold=0.85):
    """Remove highly correlated features with validation"""
    if df.empty:
        return df
        
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)

# ========================
# FEATURE ENGINEERING
# ========================
class FeatureEngineer:
    @staticmethod
    def engineer_features(df, prefix=''):
        """Create technical features without lookahead bias"""
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Handle missing data
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # Price transformations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features (using shifted data to prevent lookahead)
        df['ATR'] = ta.volatility.AverageTrueRange(
            high=df['high'].shift(1), 
            low=df['low'].shift(1), 
            close=df['close'].shift(1), 
            window=14
        ).average_true_range()
        
        df['volatility_20'] = df['returns'].rolling(20).std().shift(1)
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['close'].shift(1), window=14)
        
        # MACD with proper shifting
        macd_indicator = ta.trend.MACD(df['close'].shift(1))
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        
        # Trend indicators
        for window in [20, 50, 100]:
            sma = ta.trend.sma_indicator(df['close'].shift(1), window)
            df[f'SMA_{window}'] = sma
            df[f'price_sma_{window}_ratio'] = df['close'].shift(1) / (sma + 1e-9)
            
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Safe NaN handling (no lookahead)
        df = df.ffill().fillna(0)
        
        return df.add_prefix(prefix)

    @staticmethod
    def validate_features(df):
        """Ensure feature quality before modeling"""
        if df.empty:
            return df
            
        # Remove constant features
        nunique = df.nunique()
        constant_cols = nunique[nunique == 1].index.tolist()
        df = df.drop(columns=constant_cols)
        
        # Remove high-correlation features
        return prune_correlated_features(df, threshold=0.8)

# ========================
# MODEL ARCHITECTURE
# ========================
class LeakProofTradingModel:
    """Prevents data leakage with strict time-series validation"""
    def __init__(self, model_type='xgb'):
        self.model_type = model_type
        self.model = None
        self.calibrator = None
        self.selected_features = None
        self.scaler = RobustScaler()
        
    def train(self, X, y, k_features=20):
        if X.empty or len(y) == 0:
            raise ValueError("Empty dataset provided for training")
            
        # Strict time-series split
        tscv = TimeSeriesSplit(n_splits=3)
        feature_scores = pd.Series(0, index=X.columns)
        
        # Cross-validated feature selection
        for train_idx, _ in tscv.split(X):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            # Skip if not enough data
            if len(y_train) < 10:
                continue
                
            # Feature selection per fold
            selector = SelectKBest(f_classif, k='all')
            selector.fit(X_train, y_train)
            
            # Accumulate scores
            valid_scores = pd.Series(selector.scores_, index=X_train.columns)
            feature_scores = feature_scores.add(valid_scores, fill_value=0)
        
        # Select top features
        k_features = min(k_features, len(feature_scores))
        self.selected_features = feature_scores.nlargest(k_features).index.tolist()
        X = X[self.selected_features]
        
        # Time-based split
        split_idx = int(0.8 * len(X))
        X_train, X_cal = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_cal = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Handle class imbalance with weights
        class_ratio = max(1, len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])))
        model = self._init_model(class_ratio)
        
        # Train with robust scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # Calibrate with hold-out set
        if len(X_cal) >= Config.MODEL_PARAMS['min_calibration_samples']:
            X_cal_scaled = self.scaler.transform(X_cal)
            self.calibrator = CalibratedClassifierCV(
                model, method='sigmoid', cv='prefit')
            self.calibrator.fit(X_cal_scaled, y_cal)
        else:
            self.model = model

    def _init_model(self, class_ratio):
        """Initialize model with class weights"""
        if self.model_type == 'xgb':
            return xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, 
                scale_pos_weight=class_ratio, eval_metric='logloss')
        elif self.model_type == 'lgbm':
            return lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_child_samples=20, class_weight='balanced')
        elif self.model_type == 'catboost':
            return cb.CatBoostClassifier(
                iterations=200, depth=5, learning_rate=0.05, 
                auto_class_weights='Balanced', verbose=0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, X):
        """Generate calibrated predictions"""
        if not self.model:
            raise RuntimeError("Model not trained")
            
        if self.selected_features is None:
            raise RuntimeError("Features not selected")
            
        # Select only trained features
        available_features = [f for f in self.selected_features if f in X.columns]
        if not available_features:
            raise ValueError("No valid features for prediction")
            
        X = X[available_features]
        X_scaled = self.scaler.transform(X)
        
        if self.calibrator:
            proba = self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            proba = self.model.predict_proba(X_scaled)[:, 1]
            
        return (proba > 0.5).astype(int), proba

class AdvancedEnsemble:
    """Meta-model with dynamic weighting and feature consensus"""
    def __init__(self):
        self.base_models = {
            'xgb': LeakProofTradingModel('xgb'),
            'lgbm': LeakProofTradingModel('lgbm'),
            'catboost': LeakProofTradingModel('catboost')
        }
        self.meta_model = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)
        self.feature_consensus = None
        self.model_weights = None
        
    def train(self, X, y, k_features=25):
        # Phase 1: Base model training
        self.feature_consensus = self._get_feature_consensus(X, y, k_features)
        
        # Phase 2: Out-of-fold predictions
        oof_predictions = self._generate_oof_predictions(
            X[self.feature_consensus], y)
        
        # Train meta-model
        self.meta_model.fit(oof_predictions, y)
        
        # Dynamic model weighting
        self.model_weights = self._calculate_model_weights(
            oof_predictions, y)

    def _get_feature_consensus(self, X, y, k_features):
        """Select features present in multiple models"""
        feature_scores = {}
        for name, model in self.base_models.items():
            try:
                model.train(X, y, k_features)
                for row in model.feature_importance_.itertuples():
                    feature_scores.setdefault(row.feature, []).append(row.importance)
            except Exception as e:
                print(f"Model {name} training failed: {str(e)}")
                
        # Select features present in ≥2 models
        if not feature_scores:
            return X.columns.tolist()[:k_features]
            
        consensus_features = [f for f, scores in feature_scores.items() 
                             if len(scores) >= 2]
        k_features = min(k_features, len(consensus_features))
        return consensus_features[:k_features]
        
    def _generate_oof_predictions(self, X, y):
        """Generate out-of-fold predictions for stacking"""
        tscv = TimeSeriesSplit(n_splits=3)
        oof_predictions = np.full((len(X), len(self.base_models)), np.nan)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    fold_model = LeakProofTradingModel(model.model_type)
                    fold_model.train(X_train, y_train, 
                                    k_features=len(X_train.columns))
                    
                    _, probas = fold_model.predict(X_val)
                    oof_predictions[val_idx, i] = probas
                except Exception as e:
                    print(f"Fold {fold} model {name} failed: {str(e)}")
                    oof_predictions[val_idx, i] = 0.5  # Neutral probability
                
        # Handle remaining NaNs
        oof_predictions = np.where(np.isnan(oof_predictions), 0.5, oof_predictions)
        return oof_predictions
        
    def _calculate_model_weights(self, oof_predictions, y):
        """Calculate model weights based on performance"""
        weights = np.ones(len(self.base_models))
        try:
            losses = []
            for i in range(oof_predictions.shape[1]):
                loss = log_loss(y, oof_predictions[:, i])
                losses.append(loss)
                
            # Invert losses (better models get higher weight)
            inv_losses = 1 / np.array(losses)
            weights = inv_losses / inv_losses.sum()
        except Exception as e:
            print(f"Error calculating weights: {str(e)}")
            
        return weights
        
    def predict(self, X):
        """Ensemble prediction with confidence metrics"""
        # Get base model predictions
        base_probs = []
        for name, model in self.base_models.items():
            try:
                _, proba = model.predict(X[self.feature_consensus])
                base_probs.append(proba)
            except Exception as e:
                print(f"Model {name} prediction failed: {str(e)}")
                base_probs.append(np.full(len(X), 0.5))
                
        base_probs = np.array(base_probs)
        weighted_probs = np.average(base_probs, weights=self.model_weights, axis=0)
        meta_probs = self.meta_model.predict_proba(base_probs.T)[:, 1]
        prob_spread = np.max(base_probs, axis=0) - np.min(base_probs, axis=0)
        
        return (weighted_probs > 0.5).astype(int), meta_probs, prob_spread

# ========================
# TRADING EXECUTION
# ========================
class TradeManager:
    def __init__(self, mt5_conn):
        self.mt5 = mt5_conn
        self.open_positions = {}
        self.equity_curve = deque(maxlen=1000)
        self.update_account_info()
        
    def update_account_info(self):
        self.account_info = mt5.account_info()
        if self.account_info:
            self.equity_curve.append(self.account_info.equity)
        
    def calculate_position_size(self, confidence, volatility):
        """Risk-based position sizing"""
        risk_amount = self.account_info.equity * Config.RISK_PARAMS['max_risk_per_trade']
        risk_amount *= max(0.1, min(1.0, confidence))  # Clamp confidence
        return round(risk_amount / (volatility + 1e-9), 2)
        
    def execute_trade(self, signal, features, confidence):
        """Execute trade with proper risk management"""
        symbol = Config.SYMBOL
        
        # Confirm trade conditions
        if not self._confirm_trade_conditions(features, signal):
            log_trade({
                'symbol': symbol,
                'type': 'buy' if signal == 1 else 'sell',
                'result': 'CANCELED',
                'reason': 'Trade conditions not met',
                'confidence': confidence
            })
            return False
        
        volatility = features.get('volatility_20', 0.01)
        lot_size = self.calculate_position_size(confidence, volatility)
        order_type = 'buy' if signal == 1 else 'sell'
        
        # Get current price
        price = mt5.symbol_info_tick(symbol).ask if order_type == 'buy' else mt5.symbol_info_tick(symbol).bid
        
        # Calculate stops
        atr = features.get('ATR', 0.001)
        sl_points = atr * Config.RISK_PARAMS['atr_sl_multiplier']
        tp_points = atr * Config.RISK_PARAMS['atr_tp_multiplier']
        
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - sl_points if order_type == 'buy' else price + sl_points,
            "tp": price + tp_points if order_type == 'buy' else price - tp_points,
            "deviation": 10,
            "comment": "ML Trading Bot",
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        
        # Execute trade
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log_trade({
                'symbol': symbol,
                'type': order_type,
                'lot': lot_size,
                'price': price,
                'sl': request['sl'],
                'tp': request['tp'],
                'confidence': confidence,
                'features': features,
                'result': 'SUCCESS'
            })
            return True
        else:
            log_trade({
                'symbol': symbol,
                'type': order_type,
                'error': result.comment,
                'result': 'FAILED',
                'confidence': confidence
            })
            return False

    def _confirm_trade_conditions(self, features, signal):
        """Check additional market conditions"""
        # Check spread
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if symbol_info.spread > Config.RISK_PARAMS['max_spread']:
            return False
            
        # Check volume
        if features.get('volume', 0) < 100:
            return False
            
        # Check trading session (London/NY overlap)
        hour = pd.Timestamp.now().hour
        if not (7 <= hour < 16):
            return False
            
        return True

    def monitor_positions(self):
        """Check and manage open positions"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if positions is None:
            return
        for pos in positions:
            current_price = mt5.symbol_info_tick(Config.SYMBOL).ask
            
            # Check SL/TP
            if pos.type == mt5.ORDER_TYPE_BUY:
                if current_price <= pos.sl or current_price >= pos.tp:
                    mt5.Close(pos.ticket)
            else:  # SELL
                if current_price >= pos.sl or current_price <= pos.tp:
                    mt5.Close(pos.ticket)
                    
            # Check expiration (for pending orders)
            if pos.time_expiration != 0 and pd.Timestamp.now() > pd.Timestamp(pos.time_expiration):
                mt5.Close(pos.ticket)

    def check_risk_limits(self):
        """Validate risk parameters"""
        if len(self.equity_curve) < 10:
            return True
            
        # Calculate drawdown
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_dd = np.max(drawdown)
        
        if max_dd > abs(Config.RISK_PARAMS['max_daily_drawdown']):
            print(f"Drawdown limit breached: {max_dd:.2%}")
            return False
            
        # Check margin
        margin_ratio = self.account_info.margin / self.account_info.equity
        if margin_ratio > Config.RISK_PARAMS['max_margin_ratio']:
            print(f"Margin limit breached: {margin_ratio:.2%}")
            return False
            
        return True

# ========================
# MAIN TRADING BOT
# ========================
class TradingBot:
    def __init__(self):
        self.model = None
        self.last_retrain = None
        self.data_store = {}  # Store data by timeframe
        self.confidence_optimizer = ConfidenceOptimizer()
        self.trade_manager = None
        self.train_step = 0
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

    def run(self):
        mt5 = get_mt5_connection()
        try:
            self.trade_manager = TradeManager(mt5)
            # Initial data load
            self._update_data(mt5, full_history=True)
            
            while True:
                try:
                    # Update account info
                    self.trade_manager.update_account_info()
                    
                    # Update market data
                    self._update_data(mt5, full_history=False)
                    
                    # Check for retraining
                    if self._should_retrain():
                        print("Starting model retraining...")
                        self._train_model()
                        print("Model retraining complete")
                    
                    # Generate trading signal
                    signal, confidence, prob_spread = self._generate_signal()
                    
                    # Execute trading logic
                    self._execute_trading_logic(signal, confidence, prob_spread)
                    
                    # Monitor open positions
                    # Check if any positions are open before monitoring
                    positions = self.trade_manager.mt5.positions_get(symbol=Config.SYMBOL)
                    if positions and len(positions) > 0:
                        self.trade_manager.monitor_positions()
                    
                    # Check risk limits
                    if not self.trade_manager.check_risk_limits():
                        print("Risk limits breached - stopping bot")
                        break
                        
                    # Sleep until next candle
                    self._sleep_until_next_candle()
                    
                except Exception as e:
                    self._handle_error(e)
                    time.sleep(60)
        finally:
            mt5.shutdown()

    def _update_data(self, mt5, full_history=False):
        """Fetch and align data across timeframes"""
        main_df = None
        min_bars_needed = 50  # or higher if your longest rolling window is larger
        bars = Config.MODEL_PARAMS['data_points'] if full_history else min_bars_needed
        
        for tf_name in Config.TIMEFRAMES:
            try:
                # Fetch data
                df = fetch_market_data(mt5, Config.SYMBOL, tf_name, bars)
                if df is None or df.empty:
                    continue
                    
                # Process features
                df_feat = FeatureEngineer.engineer_features(df, prefix=f"{tf_name}_")
                df_feat = FeatureEngineer.validate_features(df_feat)
                
                # Store in data store
                self.data_store[tf_name] = df_feat
                
                # Align to highest timeframe (H1)
                if main_df is None:
                    main_df = df_feat
                else:
                    # Resample to align time indices
                    resampled = df_feat.resample('5T').last().ffill()
                    main_df = main_df.join(resampled, how='left')
                
                print(f"{tf_name}: fetched {len(df)} bars")
                
            except Exception as e:
                print(f"Error processing {tf_name}: {str(e)}")
                
        if main_df is None:
            raise ValueError("No data fetched from MT5")
            
        # Forward fill and clean
        main_df = main_df.ffill().dropna()
        return main_df

    def _should_retrain(self):
        if not self.last_retrain:
            return True
            
        hours_since_retrain = (pd.Timestamp.now() - self.last_retrain).total_seconds() / 3600
        return hours_since_retrain >= Config.MODEL_PARAMS['retrain_interval_hours']

    def _train_model(self):
        X, y = self._prepare_dataset()
        self.model = AdvancedEnsemble()
        self.model.train(X, y, Config.MODEL_PARAMS['feature_count'])
        
        # Save model checkpoint
        model_path = os.path.join(Config.MODEL_DIR, f"model_{self.train_step}.pkl")
        joblib.dump(self.model, model_path)
        
        self.last_retrain = pd.Timestamp.now()
        self.train_step += 1

    def _prepare_dataset(self):
        """Prepare training dataset from stored data"""
        # Use H1 as base timeframe
        if 'H1' not in self.data_store or self.data_store['H1'].empty:
            raise ValueError("H1 data not available for training")
            
        df = self.data_store['H1'].copy()
        
        # Add features from other timeframes
        for tf_name, data in self.data_store.items():
            if tf_name == 'H1':
                continue
                
            # Merge using nearest time alignment
            merged = pd.merge_asof(
                df, 
                data, 
                left_index=True, 
                right_index=True,
                direction='nearest',
                suffixes=('', f"_{tf_name}")
            )
            df = merged
            
        # Clean data
        df = df.ffill().dropna()
        
        # Create target - next hour return
        df['target'] = (df['H1_returns'].shift(-1) > 0).astype(int)
        df = df.dropna(subset=['target'])
        
        # Prepare X and y
        X = df.drop(columns=['target'])
        y = df['target']
        
        return X, y

    def _generate_signal(self):
        """Generate trading signal from latest data"""
        latest_features = self._get_latest_features()
        if latest_features.empty:
            raise ValueError("No features available for prediction")
            
        prediction, confidence, prob_spread = self.model.predict(latest_features)
        return prediction[0], confidence[0], prob_spread[0]

    def _execute_trading_logic(self, signal, confidence, prob_spread):
        """Execute trading strategy with market context"""
        market_context = self._get_market_context()
        confidence_threshold = self.confidence_optimizer.adjust_threshold(confidence)
        
        trade_conditions = {
            'confidence': confidence > confidence_threshold,
            'volatility': market_context['volatility'] < 0.02,
            'session': self._is_optimal_trading_session(),
            'trend_alignment': self._is_signal_aligned(signal, market_context),
            'model_agreement': prob_spread < Config.MODEL_PARAMS['prob_spread_threshold']
        }
        
        print("\nTrade Conditions:")
        for k, v in trade_conditions.items():
            print(f"- {k}: {v}")
            
        if all(trade_conditions.values()):
            features = self._get_latest_features().iloc[0].to_dict()
            result = self.trade_manager.execute_trade(signal, features, confidence)
            # Remove: self.tb_writer.add_scalar('trade/executed', int(result), self.train_step)

    def _get_latest_features(self):
        """Get latest features from H1 timeframe"""
        if 'H1' not in self.data_store or self.data_store['H1'].empty:
            raise ValueError("H1 data not available")
            
        return self.data_store['H1'].iloc[[-1]]

    def _get_market_context(self):
        """Get current market context"""
        if 'H1' not in self.data_store or self.data_store['H1'].empty:
            return {'volatility': 0.01}
            
        latest = self.data_store['H1'].iloc[-1]
        return {
            'volatility': latest.get('volatility_20', 0.01),
            'trend': latest.get('H1_MACD', 0) > latest.get('H1_MACD_signal', 0),
            'sentiment': latest.get('H1_RSI', 50)
        }

    def _is_optimal_trading_session(self):
        """Check if current time is optimal for trading"""
        hour = pd.Timestamp.now().hour
        # London/NY overlap (8 AM - 12 PM EST)
        return 13 <= hour < 17  # GMT times

    def _is_signal_aligned(self, signal, market_context):
        """Check if signal aligns with market trend"""
        if signal == 1:  # Buy signal
            return market_context['trend'] and market_context['sentiment'] > 40
        else:  # Sell signal
            return not market_context['trend'] and market_context['sentiment'] < 60

    def _sleep_until_next_candle(self):
        """Sleep until next 5-minute candle"""
        now = pd.Timestamp.now()
        next_candle = (now + pd.Timedelta(minutes=5)).replace(second=0, microsecond=0)
        sleep_seconds = (next_candle - now).total_seconds()
        print(f"Sleeping {sleep_seconds:.1f} seconds until next candle")
        time.sleep(max(1, sleep_seconds))

    def _handle_error(self, e):
        """Handle errors gracefully"""
        print(f"Error: {str(e)}")
        traceback.print_exc()
        log_trade({
            'event': 'ERROR',
            'message': str(e),
            'result': 'ERROR'
        })

# ========================
# CONFIDENCE OPTIMIZATION
# ========================
class ConfidenceOptimizer:
    def __init__(self):
        self.confidence_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.threshold = Config.MODEL_PARAMS['confidence_threshold']
        
    def update(self, confidence, actual_pnl):
        self.confidence_history.append(confidence)
        self.accuracy_history.append(1 if actual_pnl > 0 else 0)
        
    def adjust_threshold(self, current_confidence):
        if len(self.accuracy_history) < 30:
            return self.threshold
            
        # Calculate recent accuracy
        recent_accuracy = np.mean(list(self.accuracy_history)[-30:])
        
        # Dynamic threshold adjustment
        if recent_accuracy < 0.55 and current_confidence > 0.6:
            return max(0.5, self.threshold * 0.95)
        elif recent_accuracy > 0.65:
            return min(0.75, self.threshold * 1.05)
            
        return self.threshold

# ========================
# RUN THE BOT
# ========================
if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
    print(sys.executable)
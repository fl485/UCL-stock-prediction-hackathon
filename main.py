import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import clone
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
import os
import logging
import random

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#paths
test_data_path = 'test.csv'
train_data_path = 'train.csv'
submission_path = 'submission.csv'

#column names
date_col = 'Date'
target_col = "Close"
open_col = 'Open'
high_col = 'High'
low_col = 'Low'
id_col = 'ID'
volume_col = 'Volume'
random_state = random.randint(0, 100)
N_SPLITS = 5
DO_HYPERPARAM_TUNING = False

def load_data(train_path: 'str', test_path: 'str'):
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Please check the path.")
        return None, None
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    if date_col in train_df.columns:
        train_df[date_col] = pd.to_datetime(train_df[date_col], utc=True)
    if date_col in test_df.columns:
        test_df[date_col] = pd.to_datetime(test_df[date_col], utc=True)


    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    print("\nTrain Head:")
    print(train_df.head())

    print("\nTrain Info:")
    print(train_df.info())

    
    return train_df, test_df

def feature_engineering(df: pd.DataFrame, is_train: bool = True):
    df = df.copy()
    
    #sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)

    #date-based features

    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    #cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
    #flags
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)


    #technical indicators

    #lags (past prices)
    lag_target = target_col 
    
    #print(f"DEBUG: Generating Features for {lag_target}...")
    if lag_target not in df.columns:
        print(f"WARNING: {lag_target} missing. Skipping price features.")
        return df
    prev_close = df[lag_target].shift(1) 
    lags = [1, 2, 3, 5, 10, 21] 
    for lag in lags:
        df[f'lag_{lag}'] = df[lag_target].shift(lag)

    #log returns of previous day
    df['log_ret_1'] = np.log(prev_close / prev_close.shift(1))

    # Simple returns
    df['ret_1'] = prev_close / prev_close.shift(1) - 1
    df['ret_5'] = prev_close / prev_close.shift(5) - 1
    df['ret_10'] = prev_close / prev_close.shift(10) - 1

    # Rolling mean of returns (trend)
    for window in [5, 10, 20]:
        df[f'ret_mean_{window}'] = df['ret_1'].rolling(window=window).mean()
        df[f'ret_std_{window}'] = df['ret_1'].rolling(window=window).std()
    
    #volatility (ATR - Average True Range)
    high_prev = df[high_col].shift(1)
    low_prev = df[low_col].shift(1)
    close_prev_2 = df[lag_target].shift(2)
    
    tr1 = high_prev - low_prev
    tr2 = (high_prev - close_prev_2).abs()
    tr3 = (low_prev - close_prev_2).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    #MACD on previous close
    ema12 = prev_close.ewm(span=12, adjust=False).mean()
    ema26 = prev_close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # MACD relative to signal
    df['macd_over_signal'] = df['macd'] - df['macd_signal']

    #momentum (RSI) on previous close
    delta = prev_close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    #bollinger bands on previous close
    bb_window = 20
    df['bb_mid'] = prev_close.rolling(window=bb_window).mean()
    df['bb_std'] = prev_close.rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
    
    #features describing position within bands (using prev_close)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-9)
    df['price_vs_bb'] = (prev_close - df['bb_mid']) / (2 * df['bb_std'] + 1e-9)

    # Interaction of trend strength & volatility
    df['trend_x_vol'] = df['ret_mean_10'] * df['ret_std_10']

    # RSI relative to BB position
    df['rsi_x_price_vs_bb'] = df['rsi_14'] * df['price_vs_bb']

    # Price above/below medium MA
    df['above_ma_50'] = (prev_close > prev_close.rolling(window=50).mean()).astype(int)

    # Volatility regime: high vs low (e.g., compared to rolling std)
    vol = df['ret_1'].rolling(window=20).std()
    df['vol_regime'] = (vol > vol.median()).astype(int)

    #rolling statistics on previous close
    windows = [5, 10, 20, 50]
    for window in windows:
        df[f'roll_mean_{window}'] = prev_close.rolling(window=window).mean()
        df[f'roll_std_{window}'] = prev_close.rolling(window=window).std()
        df[f'roll_min_{window}'] = prev_close.rolling(window=window).min()
        df[f'roll_max_{window}'] = prev_close.rolling(window=window).max()
        df[f'roll_skew_{window}'] = prev_close.rolling(window=window).skew()
        df[f'roll_kurt_{window}'] = prev_close.rolling(window=window).kurt()
        df[f'roll_zscore_{window}'] = (prev_close - df[f'roll_mean_{window}']) / (df[f'roll_std_{window}'] + 1e-9)
        df[f'dist_ma_{window}'] = (prev_close - df[f'roll_mean_{window}']) / (df[f'roll_mean_{window}'] + 1e-9)


    #stochastic oscillator & williams %R (momentum)
    k_window = 14
    low_min = df['Low'].shift(1).rolling(window=k_window).min()
    high_max = df['High'].shift(1).rolling(window=k_window).max()
    
    #fast stochastic  %K
    df['stoch_k'] = 100 * ((prev_close - low_min) / (high_max - low_min + 1e-9))
    #slow stochastic %D (3-day SMA of %K)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    #williams %R
    df['williams_r'] = -100 * ((high_max - prev_close) / (high_max - low_min + 1e-9))
    
    #rate of change (ROC)
    roc_window = 10
    df['roc_10'] = ((prev_close - prev_close.shift(roc_window)) / (prev_close.shift(roc_window) + 1e-9)) * 100
    
    #CCI (Commodity Channel Index)
    tp = (df['High'].shift(1) + df['Low'].shift(1) + prev_close) / 3
    tp_sma = tp.rolling(window=20).mean()
    #mean deviation
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (tp - tp_sma) / (0.015 * mad + 1e-9)

    #add lag features for close price
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = prev_close.shift(lag)

    #add lag features for open price
    for lag in [1, 2, 3, 5, 10]:
        df[f'open_lag_{lag}'] = df['Open'].shift(lag)

    #add lag features for high price
    for lag in [1, 2, 3, 5, 10]:
        df[f'high_lag_{lag}'] = df['High'].shift(lag)

    #add lag features for low price
    for lag in [1, 2, 3, 5, 10]:
        df[f'low_lag_{lag}'] = df['Low'].shift(lag)

    # Volume lags
    for lag in [1, 2, 5, 10]:
        df[f'vol_lag_{lag}'] = df['Volume'].shift(lag)

    # Rolling volume statistics
    for window in [5, 10, 20]:
        roll = df['Volume'].shift(1).rolling(window=window)
        df[f'vol_roll_mean_{window}'] = roll.mean()
        df[f'vol_roll_std_{window}'] = roll.std()
        df[f'vol_zscore_{window}'] = (
            (df['Volume'].shift(1) - df[f'vol_roll_mean_{window}']) /
            (df[f'vol_roll_std_{window}'] + 1e-9)
        )
    return df

def data_prpeprocessing(train_df, test_df):

    ignore_cols = [target_col, date_col, id_col, 'target_return']
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    #drop starting rows with NaN due to rolling in train df only
    train_df_clean = train_df.dropna(subset=feature_cols + ['target_return']).reset_index(drop=True)
    
    X = train_df_clean[feature_cols]
    y = train_df_clean['target_return']
    
    X_test = test_df[feature_cols]
    X_test = X_test.ffill().fillna(0)
    
    print(f"Final Feature/Target shapes:")
    print(f"X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    
    return X, y, X_test, feature_cols, train_df_clean[date_col]
    '''
        ignore_cols = [target_col, date_col, id_col]
        feature_cols = [c for c in train_df.columns if c not in ignore_cols]
        train_df_clean = train_df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        
        x = train_df_clean[feature_cols]
        y = train_df_clean[target_col]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle=False, random_state=0)
        return x_train, x_test, y_train, y_test
    '''
    
def evaluate_predictions(y_true, y_pred, X_val=None):
    metrics = {}
    
    mae = mean_absolute_error(y_true, y_pred)
    metrics['mae'] = mae
    
    return metrics

def train_and_evaluate_model(model, X, y, X_test, model_name: str, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    #store metrics for each fold
    fold_metrics = {"mae": []}
    
    print(f"\ntraining {model_name}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        step_model = clone(model)
        step_model.fit(X_train_fold, y_train_fold)
        
        y_pred_val_log_ret = step_model.predict(X_val_fold)
        
        #reconstruct prices
        if 'lag_1' in X_val_fold.columns:
            prev_close_val = X_val_fold['lag_1']
        else:
            prev_close_val = X.iloc[val_idx]['lag_1']

        
        #predicted price = prev price * exp(pred log return)
        y_pred_val_price = prev_close_val * np.exp(y_pred_val_log_ret)
        
        #reconstruct true prices
        y_val_true_price = prev_close_val * np.exp(y_val_fold)
        
        metrics = evaluate_predictions(y_val_true_price, y_pred_val_price, X_val_fold)
        
        for k in fold_metrics:
            if k in metrics:
                fold_metrics[k].append(metrics[k])
                
        print(f"Fold {fold}: {metrics['mae']}")
        
    #compute averages
    avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
    print(f"{model_name} - Avg: {avg_metrics['mae']}")
    
    #retrain on full data for final submission
    final_model = clone(model)
    final_model.fit(X, y)
    
    #return average MAE and the full avg metrics
    return avg_metrics, final_model

def tune_random_forest(X, y):
    print("\ntuning random forest")
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5]
    }
    
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=15, 
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X, y)
    print(f"best params: {search.best_params_}")
    print(f"best score: {search.best_score_:.5f}")
    return search.best_estimator_

def tune_hist_gradient_boosting(X, y):
    print("\ntuning histgradientboosting")
    hgb = HistGradientBoostingRegressor(loss='absolute_error', random_state=random_state)
    
    param_dist = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0.0, 0.1, 1.0]
    }
    
    search = RandomizedSearchCV(
        estimator=hgb,
        param_distributions=param_dist,
        n_iter=15, 
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X, y)
    print(f"best params: {search.best_params_}")
    print(f"best score: {search.best_score_:.5f}")
    return search.best_estimator_

if __name__ == '__main__':

    train_df, test_df = load_data(train_data_path, test_data_path)
    
    if train_df is None:
        print("execution stopped due to missing data")
    else:
        print('yes')
        train_df['is_train'] = 1
        test_df['is_train'] = 0
        
        real_data_path = 'real.csv'
        has_real_data = False
        
        if os.path.exists(real_data_path):
            print(f"Found {real_data_path}. Loading for validation...")
            real_df = pd.read_csv(real_data_path)
            #ensure id alignment
            if id_col in real_df.columns and id_col in test_df.columns:
                 #map real close to test_df
                 real_map = dict(zip(real_df[id_col], real_df['close' if 'close' in real_df.columns else target_col]))
                 test_df[target_col] = test_df[id_col].map(real_map)
                 has_real_data = True
                 print(f"Loaded {len(real_df)} rows from real.csv into test set for Validation.")
            else:
                 print("WARNING: real.csv found but ID column missing or mismatch. Skipping validation usage.")
        
        if target_col not in test_df.columns:
            test_df[target_col] = np.nan
            
        full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        print("\nEngineering Features (Full History)...")
        full_df = feature_engineering(full_df, is_train=True)
        
        #calculate log returns for the target variable
        full_df['target_return'] = np.log(full_df[target_col] / full_df[target_col].shift(1))
               
        #split back
        train_df = full_df[full_df['is_train'] == 1].copy()
        test_df = full_df[full_df['is_train'] == 0].copy()
        train_df.drop(columns=['is_train'], inplace=True)
        test_df.drop(columns=['is_train'], inplace=True)

        X, y, X_test, feature_cols, train_dates = data_prpeprocessing(train_df, test_df)

        
        models_config = [
            ("LinearReg", LinearRegression()),
            ("RandomForest", RandomForestRegressor(n_estimators=300, min_samples_split=2, min_samples_leaf=2, max_features=0.5, max_depth=None, random_state=random_state, n_jobs=-1)),
            ("HistHGBR_Abs", HistGradientBoostingRegressor(loss='absolute_error', min_samples_leaf=5, max_leaf_nodes=15, max_iter=300, max_depth=None, learning_rate=0.05, l2_regularization=0.1, random_state=random_state)),
            ("GBR_Abs", GradientBoostingRegressor(loss='absolute_error', min_samples_leaf=5, max_leaf_nodes=15, max_depth=5, learning_rate=0.05, random_state=random_state)),
            ("Ridge_a0.008", Ridge(alpha=0.008)),
            ("XGB_Abs", XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, objective='reg:absoluteerror', random_state=random_state, n_jobs=-1, tree_method='hist')),
            ("LGBM_Abs", LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.05, objective='regression_l1', random_state=random_state, n_jobs=-1, verbose=-1)),
            ("CatBoost_Abs", CatBoostRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, loss_function='MAE', random_state=random_state, verbose=0)),
        ]


        
        #searching algorithm for ridge regression
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingRandomSearchCV
        search = HalvingRandomSearchCV(
            Ridge(), 
            param_distributions={'alpha': np.logspace(-5, 2, 100)}, 
            min_resources=10, 
            factor=1.5,
            cv=TimeSeriesSplit(n_splits=5), 
            scoring='neg_mean_absolute_error', 
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X, y)
        best_alpha = search.best_params_['alpha']
        print(f"Best Ridge alpha: {best_alpha:.6f}")
        models_config.append((f"Ridge_a{best_alpha}", Ridge(alpha=best_alpha)))

        #searching algorithm for lasso regression
        search_lasso = HalvingRandomSearchCV(
            Lasso(), 
            param_distributions={'alpha': np.logspace(-5, 2, 100)}, 
            min_resources=10, 
            factor=1.5,
            cv=TimeSeriesSplit(n_splits=5), 
            scoring='neg_mean_absolute_error', 
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        search_lasso.fit(X, y)
        best_alpha_lasso = search_lasso.best_params_['alpha']
        print(f"Best Lasso alpha: {best_alpha_lasso:.6f}")
        models_config.append((f"Lasso_a{best_alpha_lasso}", Lasso(alpha=best_alpha_lasso)))

        #searching algorithm for elasticnet regression
        search_en = HalvingRandomSearchCV(
            ElasticNet(), 
            param_distributions={
                'alpha': np.logspace(-5, 2, 100),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }, 
            min_resources=10, 
            factor=1.5,
            cv=TimeSeriesSplit(n_splits=5), 
            scoring='neg_mean_absolute_error', 
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        
        search_en.fit(X, y)
        best_alpha_en = search_en.best_params_['alpha']
        best_l1_ratio_en = search_en.best_params_['l1_ratio']
        print(f"Best ElasticNet alpha: {best_alpha_en:.6f}")
        print(f"Best ElasticNet l1_ratio: {best_l1_ratio_en:.6f}")
        models_config.append((f"ElasticNet_a{best_alpha_en}_l1_{best_l1_ratio_en}", ElasticNet(alpha=best_alpha_en, l1_ratio=best_l1_ratio_en)))    
        
        model_results = {}
        trained_models = {}

        #train models
        for name, model in models_config:
            # Note: train_and_evaluate_model will reconstruct prices for MAE calculation
            avg_metrics, final_model = train_and_evaluate_model(model, X, y, X_test, name, n_splits=N_SPLITS)
            model_results[name] = avg_metrics
            trained_models[name] = final_model

        #tuning
        if DO_HYPERPARAM_TUNING:
            best_rf = tune_random_forest(X, y)
            metrics_rf, final_rf = train_and_evaluate_model(best_rf, X, y, X_test, "Tuned_RandomForest", n_splits=N_SPLITS)
            model_results["Tuned_RandomForest"] = metrics_rf
            trained_models["Tuned_RandomForest"] = final_rf
            
            best_hgb = tune_hist_gradient_boosting(X, y)
            metrics_hgb, final_hgb = train_and_evaluate_model(best_hgb, X, y, X_test, "Tuned_HGBR", n_splits=N_SPLITS)
            model_results["Tuned_HGBR"] = metrics_hgb
            trained_models["Tuned_HGBR"] = final_hgb

        #select best model
        selector_metric = 'mae'
        best_model_name = min(model_results, key=lambda k: model_results[k][selector_metric])
        best_res = model_results[best_model_name]
        
        print(f"WINNER: {best_model_name}")
        print(f"MAE:    {best_res['mae']:.5f}")

        
        #comparison table
        print("\nModel Comparison")
        print(f"{'Model':<20} | {'MAE':<10}")
        print("-" * 58)
        for name, res in model_results.items():
            print(f"{name:<20} | {res['mae']:.5f}")
        
        #ensemble based on inverse MAE
        print("\nGenerating Ensemble Predictions")
        all_preds = []
        weights = []
        
        #use top 1 models
        sorted_models = sorted(model_results.items(), key=lambda x: x[1][selector_metric])
        top_k = 1
        top_models = sorted_models[:top_k]
        
        print(f"Ensembling Top {top_k} Models: {[m[0] for m in top_models]}")
        
        for name, res in top_models:
            model = trained_models[name]

            #since boosting and forest models are bad at predicting trends they havent seen yet we predict log returns
            pred_log_returns = model.predict(X_test)
            
            #reconstruct price
            if 'lag_1' in X_test.columns:
                prev_close_test = X_test['lag_1']
            elif 'close_lag_1' in X_test.columns:
                 prev_close_test = X_test['close_lag_1']
            else:
                print("WARNING: lag_1 column not found in X_test. Using fallback if possible.")
                prev_close_test = X_test['lag_1']

            pred_prices = prev_close_test * np.exp(pred_log_returns)
            
            all_preds.append(pred_prices)
            weights.append(1.0 / (res[selector_metric] + 1e-9))
            
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        weighted_preds = np.average(all_preds, axis=0, weights=weights)

        #final submission
        submission_df = pd.DataFrame()
        full_df.to_csv("full_df.csv", index=False)
        if id_col in test_df.columns:
            submission_df[id_col] = test_df[id_col]
        else:
            submission_df[id_col] = range(len(test_df))
            
        submission_df["Prediction"] = weighted_preds
        submission_df.to_csv(submission_path, index=False)
        print(f"\nSubmission saved to {submission_path}")
        
        #final validation against real data
        if has_real_data:
            print(f"{'Model':<25} | {'Test MAE':<10} | {'Test MAPE':<10}")
            print("-" * 60)
            
            real_vals = test_df[target_col].values
            mask = ~np.isnan(real_vals)
            y_true_final = real_vals[mask]
            
            if len(y_true_final) > 0:
                best_model_name = "Ensemble"
                best_model_mae = float('inf')
                best_model_preds = weighted_preds
                
                y_pred_ensemble = weighted_preds[mask]
                ens_mae = mean_absolute_error(y_true_final, y_pred_ensemble)
                ens_mape = mean_absolute_percentage_error(y_true_final, y_pred_ensemble)
                print(f"{'Ensemble':<25} | {ens_mae:.5f}   | {ens_mape:.2%}")
                
                if ens_mae < best_model_mae:
                    best_model_mae = ens_mae
                    best_model_preds = weighted_preds
                
                #validate individual models
                for name, model in trained_models.items():
                    #predict log return
                    pred_log = model.predict(X_test)
                    
                    #reconstruct price 
                    if 'lag_1' in X_test.columns:
                        prev = X_test['lag_1']
                    elif 'close_lag_1' in X_test.columns:
                        prev = X_test['close_lag_1']
                    else:
                        prev = X_test.iloc[:, 0]
                    
                    pred_price = prev * np.exp(pred_log)
                    
                    y_pred_indiv_valid = pred_price.values[mask]
                    
                    mae_indiv = mean_absolute_error(y_true_final, y_pred_indiv_valid)
                    mape_indiv = mean_absolute_percentage_error(y_true_final, y_pred_indiv_valid)
                    
                    print(f"{name:<25} | {mae_indiv:.5f}   | {mape_indiv:.2%}")
                    
                    if mae_indiv < best_model_mae:
                        best_model_mae = mae_indiv
                        best_model_name = name
                        best_model_preds = pred_price.values 
                
                print("="*50)
                print(f"BEST MODEL: {best_model_name} (MAE: {best_model_mae:.5f})")
                
                #make submission the actual best model
                submission_df["Prediction"] = best_model_preds
                submission_df.to_csv(submission_path, index=False)
                print(f"submission updated with predictions from {best_model_name}")

            else:
                print("Error")

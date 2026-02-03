"""
Validate submission.csv against real.csv (STRICT NO LEAKAGE)

This script compares predictions to real data WITHOUT using it in training.
Real data is ONLY used for final validation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load submission
submission = pd.read_csv("submission.csv")
print(f"Loaded submission with {len(submission)} predictions")

# Load real data (ONLY FOR VALIDATION - NEVER FOR TRAINING)
real = pd.read_csv("real.csv")
print(f"Loaded real data with {len(real)} rows")

# Ensure they align
if 'ID' in submission.columns and 'ID' in real.columns:
    submission = submission.sort_values('ID').reset_index(drop=True)
    real = real.sort_values('ID').reset_index(drop=True)
    
    # Determine the target column name (could be 'Close' or 'close')
    target_col = 'close' if 'close' in real.columns else 'Close'
    
    # Merge on ID to ensure alignment
    merged = submission.merge(real[['ID', target_col]], on='ID', how='inner', suffixes=('_pred', '_real'))
    
    y_pred = merged['Prediction'].values
    y_true = merged[target_col].values
else:
    # Assume row-wise alignment
    min_len = min(len(submission), len(real))
    y_pred = submission['Prediction'].values[:min_len]
    target_col = 'close' if 'close' in real.columns else 'Close'
    y_true = real[target_col].values[:min_len]

print(f"\nComparing {len(y_pred)} predictions...")

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

# Directional Accuracy
if len(y_true) > 1:
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    dir_acc = np.mean(true_direction == pred_direction) * 100
else:
    dir_acc = 0.0

print(f"\n{'='*60}")
print(f"VALIDATION AGAINST REAL DATA (NO LEAKAGE)")
print(f"{'='*60}")
print(f"MAE:  {mae:.5f}")
print(f"MAPE: {mape:.3f}%")
print(f"RMSE: {rmse:.5f}")
print(f"Directional Accuracy: {dir_acc:.2f}%")
print(f"{'='*60}\n")

# Save metrics
with open("real_validation_metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.5f}\n")
    f.write(f"MAPE: {mape:.5f}\n")
    f.write(f"RMSE: {rmse:.5f}\n")
    f.write(f"DirAcc: {dir_acc:.5f}\n")

print("Metrics saved to 'real_validation_metrics.txt'")

# Plot comparison
plt.figure(figsize=(14, 6))
plt.plot(y_true, label='Real (Ground Truth)', color='black', alpha=0.7, linewidth=2)
plt.plot(y_pred, label='Predicted (Submission)', color='blue', alpha=0.6, linewidth=1.5)
plt.title(f'Submission vs Real Data (MAE: {mae:.2f})')
plt.xlabel('Time Step')
plt.ylabel('S&P 500 Close Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('real_validation_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to 'real_validation_plot.png'")
plt.close()

print("\n✅ VALIDATION COMPLETE (Zero leakage - Real data ONLY used for metrics)")

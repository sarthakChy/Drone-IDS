import numpy as np
import pandas as pd
import warnings

def extract_window_features(df: pd.DataFrame) -> dict:
    """
    Extract robust window-level statistical + dynamic features.
    This version is "NaN-safe" and suitable for real-time inference.
    """
    features = {}

    # Suppress warnings that occur when a slice is all-NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for col in df.columns:
            # Ensure data is float, handle potential infs
            col_data = df[col].astype(float).values
            
            # Check if all data for this window is NaN (can happen)
            if np.all(np.isnan(col_data)):
                features[f"{col}_mean"] = np.nan
                features[f"{col}_std"] = np.nan
                features[f"{col}_min"] = np.nan
                features[f"{col}_max"] = np.nan
                features[f"{col}_slope"] = np.nan
                features[f"{col}_range"] = np.nan
                features[f"{col}_diff_mean"] = np.nan
                features[f"{col}_diff_std"] = np.nan
                continue # Skip to next column

            # ===== Basic Stats (nan-safe) =====
            features[f"{col}_mean"] = np.nanmean(col_data)
            features[f"{col}_std"] = np.nanstd(col_data)
            features[f"{col}_min"] = np.nanmin(col_data)
            features[f"{col}_max"] = np.nanmax(col_data)

            # Slope (last - first, finding first/last non-nan)
            try:
                # Find the first and last valid (non-NaN) indices
                valid_indices = np.where(~np.isnan(col_data))[0]
                first_valid_idx = valid_indices[0]
                last_valid_idx = valid_indices[-1]
                
                # Calculate slope based on the first and last *valid* points
                features[f"{col}_slope"] = col_data[last_valid_idx] - col_data[first_valid_idx]
            except IndexError: 
                # This happens if all data was NaN, which our first check handles,
                # but it's good practice to keep it.
                features[f"{col}_slope"] = np.nan

            # Range (based on nan-safe min/max)
            features[f"{col}_range"] = np.nanmax(col_data) - np.nanmin(col_data)

            # First-order difference (nan-safe)
            diff = np.diff(col_data) # np.diff propagates NaNs
            if len(diff[~np.isnan(diff)]) > 0: # Only calc if there are valid diffs
                features[f"{col}_diff_mean"] = np.nanmean(diff)
                features[f"{col}_diff_std"] = np.nanstd(diff)
            else:
                # This happens if there are no two consecutive valid numbers
                features[f"{col}_diff_mean"] = np.nan
                features[f"{col}_diff_std"] = np.nan

    return features
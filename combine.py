import pandas as pd
import os

# --- Configuration ---
FILES_TO_LOAD = [
    "parsed_data/normal_corridor_base_1_data.csv",
    "parsed_data/normal_corridor_base_2_data.csv",
    "parsed_data/normal_corridor_base_3_data.csv",
    "parsed_data/normal_corridor_base_4_data.csv",
    "parsed_data/normal_corridor_base_5_data.csv",
    "parsed_data/normal_corridor_base_6_data.csv",
    "parsed_data/normal_corridor_base_7_data.csv",
    "parsed_data/normal_corridor_base_8_data.csv",
    "parsed_data/normal_corridor_base_9_data.csv",
    "parsed_data/normal_corridor_base_10_data.csv",
    "parsed_data/normal_corridor_base_11_data.csv",
    "parsed_data/normal_corridor_base_12_data.csv",
    "parsed_data/normal_corridor_base_13_data.csv",
    "parsed_data/normal_corridor_base_14_data.csv",
    "parsed_data/normal_corridor_base_15_data.csv",

    "parsed_data/attack_corridor_base_1_data.csv",
    "parsed_data/attack_corridor_base_4_data.csv",
    "parsed_data/attack_corridor_base_6_data.csv",
    "parsed_data/attack_corridor_base_7_data.csv",
    "parsed_data/attack_corridor_base_8_data.csv",
    "parsed_data/attack_corridor_base_14_data.csv",
    "parsed_data/attack_corridor_base_15_data.csv",
    "parsed_data/attack_corridor_base_16_data.csv",
    "parsed_data/attack_corridor_base_17_data.csv",
    "parsed_data/attack_corridor_base_18_data.csv"
]

OUTPUT_CSV_FILE = "my_master_dataset.csv"
OUTPUT_MASTER_FULL = "my_master_dataset_full.csv"  # full master with metadata
# --- End of Configuration ---


def prepare_data_for_ml(csv_files, output_csv, output_full):
    if not csv_files:
        print("Error: FILES_TO_LOAD is empty.")
        return

    print(f"Found {len(csv_files)} CSV files to process...")

    df_list = []
    for f in csv_files:
        if not os.path.exists(f):
            print(f"Warning: File not found, skipping: {f}")
            continue
        try:
            print(f"Loading {f}...")
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read {f}. Error: {e}")

    if not df_list:
        print("Error: No dataframes were loaded. Aborting.")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    print(f"\nSuccessfully combined all files. Total rows: {len(master_df)}")
    print("Label distribution in combined data:")
    if 'label' in master_df.columns:
        print(master_df['label'].value_counts(normalize=True))
    else:
        print("No 'label' column found in combined data.")

    # Ensure attack_type and attack_params_json exist and have sane defaults
    if 'attack_type' not in master_df.columns:
        master_df['attack_type'] = 'none'
    else:
        master_df['attack_type'] = master_df['attack_type'].fillna('none')

    if 'attack_params_json' not in master_df.columns:
        master_df['attack_params_json'] = "{}"
    else:
        master_df['attack_params_json'] = master_df['attack_params_json'].fillna("{}")

    # Save a full master that keeps everything (for inspection / auditing)
    try:
        master_df.to_csv(output_full, index=False)
        print(f"Saved full master CSV with metadata to: {output_full}")
    except Exception as e:
        print(f"Warning: could not save full master CSV. {e}")

    # --- 2. Feature Selection (numeric features used for model training) ---
    features = [
        'att_roll',
        'att_pitch',
        'att_yaw',
        'pos_lat',
        'pos_lon',
        'pos_alt_rel',
        'pos_vx',
        'pos_vy',
        'pos_vz',
        'nav_roll',
        'nav_pitch',
        'nav_alt_error',
        'sys_load',
        'vib_x',
        'vib_y',
        'vib_z'
    ]

    target = 'label'

    # Validate that the feature columns exist in master_df; if any missing, warn and remove them from features
    available_features = [f for f in features if f in master_df.columns]
    missing = [f for f in features if f not in master_df.columns]
    if missing:
        print(f"Warning: The following expected feature columns are missing and will be ignored: {missing}")
    features = available_features

    # Keep identifiers + selected features + target
    columns_to_keep = ['flight_id', 'timestamp'] + features
    if target in master_df.columns:
        columns_to_keep.append(target)
    else:
        print("Warning: target column 'label' not found; result will not contain a target column.")

    ml_df = master_df[columns_to_keep].copy()

    print(f"Selected {len(features)} features for the ML dataset: {features}")

    # --- 3. Data Cleaning ---
    initial_rows = len(ml_df)
    # Drop rows with NaNs in the numeric features (only if there are numeric features selected)
    if features:
        ml_df.dropna(subset=features, inplace=True)
    final_rows = len(ml_df)

    if initial_rows > final_rows:
        print(f"Dropped {initial_rows - final_rows} rows with missing values in features.")

    if final_rows == 0:
        print("Error: All data was dropped. Please check parsed CSVs and feature list.")
        return

    # --- 4. Save outputs ---
    try:
        ml_df.to_csv(output_csv, index=False)
        print(f"\nSuccess! ML-ready dataset saved to '{output_csv}' with {final_rows} rows.")
    except Exception as e:
        print(f"Error: Could not save the final CSV. {e}")


if __name__ == '__main__':
    print("--- Starting ML Data Preparation ---")
    prepare_data_for_ml(FILES_TO_LOAD, OUTPUT_CSV_FILE, OUTPUT_MASTER_FULL)
    print("--- Script Finished ---")


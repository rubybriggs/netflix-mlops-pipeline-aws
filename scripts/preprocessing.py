import os
import sys
import traceback
import pandas as pd
import time

output_path = '/opt/ml/processing/output/'
os.makedirs(output_path, exist_ok=True)

try:
    print("STARTING Phase 2 Feature Engineering...")
    
    # 1. Find and load data
    input_files = os.listdir('/opt/ml/processing/input/')
    csv_files = [f for f in input_files if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in input directory!")
    
    print(f"Reading file: {csv_files[0]}")
    df = pd.read_csv(os.path.join('/opt/ml/processing/input/', csv_files[0]), dtype=str)
    print(f"Loaded {len(df)} records.")
    
    # 2. Clean data and sanitize strings to prevent Feature Store Unicode validation errors
    df = df.fillna('')
    
    def sanitize_text(text):
        if not isinstance(text, str):
            return str(text)
        # Remove newlines, tabs, carriage returns
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Remove non-ASCII characters (like em-dashes, smart quotes) that trigger Feature Store regex bugs
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    # Apply sanitization to all text-based columns
    text_cols = ['title', 'description', 'listed_in', 'director', 'cast', 'country', 'rating', 'duration']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(sanitize_text)
    
    # 3. Feature Store Mandatory Columns
    print("Adding Feature Store mandatory columns...")
    df['record_id'] = df['show_id'].fillna(df.index.to_series().astype(str))
    df['event_time'] = time.time()
    
    # 4. Save processed data
    output_file = os.path.join(output_path, 'processed_data.csv')
    print(f"Saving to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print("Phase 2 Feature Engineering COMPLETED SUCCESSFULLY.")

except Exception as e:
    error_msg = f"ERROR: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"
    print(error_msg)
    with open(os.path.join(output_path, 'error_log.txt'), 'w') as f:
        f.write(error_msg)
    sys.exit(1)
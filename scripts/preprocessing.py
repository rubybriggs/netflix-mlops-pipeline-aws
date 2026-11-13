import os
import sys
import traceback
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer

output_path = '/opt/ml/processing/output/'
os.makedirs(output_path, exist_ok=True)

try:
    print("STARTING Phase 2 Feature Engineering (with embeddings)...")

    input_files = os.listdir('/opt/ml/processing/input/')
    csv_files = [f for f in input_files if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in input directory!")

    df = pd.read_csv(os.path.join('/opt/ml/processing/input/', csv_files[0]), dtype=str)
    print(f"Loaded {len(df)} records.")

    # 1. Sanitize text FIRST (ASCII + strip newlines) so embeddings & Feature Store are clean
    def sanitize_text(text):
        if not isinstance(text, str):
            return str(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text.strip()

    for col in ['title', 'description', 'listed_in', 'director', 'cast',
                'country', 'rating', 'duration']:
        if col in df.columns:
            df[col] = df[col].apply(sanitize_text)
    df = df.fillna('')

    # 2. Semantic text embeddings via TF-IDF (100 features) on sanitized text
    print("Generating TF-IDF embeddings...")
    df['content_text'] = df['title'] + " " + df['description'] + " " + df['listed_in']
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(df['content_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=[f'tfidf_feat_{i}' for i in range(100)])
    df_final = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    print(f"Generated 100 embedding features. Total columns: {len(df_final.columns)}")

    # 3. Feature Store mandatory columns
    df_final['record_id'] = df_final['show_id'].fillna(df_final.index.to_series().astype(str))
    df_final['event_time'] = time.time()

    df_final.to_csv(os.path.join(output_path, 'processed_data.csv'), index=False)
    print("Phase 2 Feature Engineering COMPLETED SUCCESSFULLY.")

except Exception as e:
    error_msg = f"ERROR: {str(e)}\n\nTRACEBACK:\n{traceback.format_exc()}"
    print(error_msg)
    with open(os.path.join(output_path, 'error_log.txt'), 'w') as f:
        f.write(error_msg)
    sys.exit(1)
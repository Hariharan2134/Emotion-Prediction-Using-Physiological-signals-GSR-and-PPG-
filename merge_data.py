import pandas as pd

# Define file paths
annotations_path = '/content/drive/My Drive/annotations/sub111.csv'
physiological_path = '/content/drive/My Drive/physiological/sub_2.csv'

merged_csv_path = '/content/merged_data.csv'
chunk_size = 5000

merged_df = pd.DataFrame()

annotations_iter = pd.read_csv(annotations_path, chunksize=chunk_size)
physiological_iter = pd.read_csv(physiological_path, chunksize=chunk_size)

for annotations_chunk, physiological_chunk in zip(annotations_iter, physiological_iter):
    chunk_merge = pd.merge(
        physiological_chunk[['bvp', 'gsr', 'video']],
        annotations_chunk[['valence', 'arousal', 'video']],
        on='video',
        how='inner'
    )
    merged_df = pd.concat([merged_df, chunk_merge], ignore_index=True)

merged_df.to_csv(merged_csv_path, index=False)
print(f"Merged data saved to {merged_csv_path}")

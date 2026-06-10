from src.extract import extract_data
from src.transform import transform_data
from src.load import load_data

RAW_PATH_FILE = "data/raw/dataset.csv"
CLEANED_PATH_FILE = "data/processed/spotify_tracks_cleaned.csv"

def main():

    raw_data = extract_data(RAW_PATH_FILE)
    print(f"Extracted {len(raw_data)} rows.")

    cleaned_data = transform_data(raw_data)
    print(f"Cleaned {len(cleaned_data)} rows.")

    load_data(cleaned_data, CLEANED_PATH_FILE)
    print(f"Loaded data as a .csv file.")

if __name__ == "__main__":
    main()

import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# Config - edit these as needed
IN_CSV = "data/IMDB-Dataset.csv"
OUT_DIR = "data/imdb_binary"

SEED = 42
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10

# Column names in the CSV
COL_RATING = "Ratings"
COL_TEXT = "Reviews"
COL_MOVIE = "Movies"


def clean_text(text):
    """Remove control chars and normalize whitespace"""
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)  # control chars
    text = re.sub(r"\s+", " ", text).strip()  # multiple spaces
    return text


def rating_to_label(rating):
    """
    Convert to binary labels:
    - 1-4 -> 0 (negative)
    - 7-10 -> 1 (positive)
    - 5-6 -> None (dropping neutral ratings)
    """
    try:
        rating = float(rating)
    except (ValueError, TypeError):
        return None
    
    if 1.0 <= rating <= 4.0:
        return 0
    elif 7.0 <= rating <= 10.0:
        return 1
    else:
        return None


def main():
    # Load data
    df = pd.read_csv(IN_CSV)
    print(len(df), "reviews before cleaning")
    
    # Keep only needed columns and rename
    df = df[[COL_RATING, COL_TEXT, COL_MOVIE]].copy()
    df.rename(columns={
        COL_RATING: "rating",
        COL_TEXT: "text",
        COL_MOVIE: "movie"
    }, inplace=True)
    
    # Clean text and create labels
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["rating"].apply(rating_to_label)
    
    # Filter out unusable rows
    df = df[df["label"].notna()].copy()  # drop neutral ratings
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() >= 10].copy()  # drop short reviews
    df.drop_duplicates(subset=["movie", "text"], inplace=True)  # remove dupes
    
    print(len(df), "reviews after cleaning")
    
    # Split by movie - prevent model from learning movie-specific words in different splits
    groups = df["movie"].astype(str)
    
    # First split: train vs (val+test)
    splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=SEED)
    train_idx, temp_idx = next(splitter.split(df, groups=groups))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    
    # Second split: val vs test
    val_proportion = VAL_SIZE / (1.0 - TRAIN_SIZE)
    temp_groups = temp_df["movie"].astype(str)
    
    splitter = GroupShuffleSplit(n_splits=1, train_size=val_proportion, random_state=SEED)
    val_idx, test_idx = next(splitter.split(temp_df, groups=temp_groups))
    
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    # Save
    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    #check for movie leakage
    train_movies = set(train_df["movie"])
    val_movies = set(val_df["movie"])
    test_movies = set(test_df["movie"])
    
    leakage = (train_movies & val_movies) | (train_movies & test_movies) | (val_movies & test_movies)
    if leakage:
        print(f"WARNING: Movie leakage found: {list(leakage)[:5]}")
    else:
        print("No movie leakage")


if __name__ == "__main__":
    main()
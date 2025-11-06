import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

# Load the original dataset
original_data = pd.read_csv("./original.csv")

# Rewrite Class Labels from 0 and 1 to true and fake
original_data["Classe"] = original_data["Classe"].replace({0: "fake", 1: "true"})

# Rename Columns
original_data = original_data.rename(
    columns={
        "Noticia": "text",
        "Classe": "classificacao",
        "Categoria": "categoria",
    }
)

# Drop Unnecessary Columns
original_data = original_data[["text", "classificacao", "categoria"]]

# Ensure categoria has no NaNs and tiny categories are handled before stratify
original_data["categoria"] = original_data["categoria"].fillna("missing")
original_data = original_data[original_data["categoria"] != "missing"]

# Inspect counts to detect tiny categories that would break stratify
counts = original_data["categoria"].value_counts()
tiny = counts[counts < 2]  # classes with fewer than 2 samples

# Remove tiny categories temporarily
if not tiny.empty:
    logging.info(
        "Removing tiny categories for stratified split: %s",
        tiny.index.tolist(),
    )
    tiny_df = original_data[original_data["categoria"].isin(tiny.index)]
    original_data = original_data[~original_data["categoria"].isin(tiny.index)]

# Stratified Train-Test Split
train_data, test_data = train_test_split(
    original_data,
    test_size=0.2,
    random_state=42,
    stratify=original_data["categoria"],
)

# Log the distribution of categories in train and test sets
category_counts = original_data["categoria"].value_counts()
category_percents = 100 * category_counts / original_data.shape[0]

logging.info("-------------------------------------------")
logging.info(
    "Original dataset category distribution:\n%s (%s percent)",
    category_counts,
    category_percents,
)
logging.info("-------------------------------------------")
logging.info(
    "Train set category distribution:\n%s (%s percent)",
    train_data["categoria"].value_counts(),
    100 * train_data["categoria"].value_counts() / train_data.shape[0],
)
logging.info("-------------------------------------------")
logging.info(
    "Test set category distribution:\n%s (%s percent)",
    test_data["categoria"].value_counts(),
    100 * test_data["categoria"].value_counts() / test_data.shape[0],
)

# Log the distribution of classes in train and test sets
class_counts = original_data["classificacao"].value_counts()
class_percents = 100 * class_counts / original_data.shape[0]

logging.info("-------------------------------------------")
logging.info(
    "Original dataset class distribution:\n%s (%s percent)",
    class_counts,
    class_percents,
)
logging.info("-------------------------------------------")
logging.info(
    "Train set class distribution:\n%s (%s percent)",
    train_data["classificacao"].value_counts(),
    100 * train_data["classificacao"].value_counts() / train_data.shape[0],
)
logging.info("-------------------------------------------")
logging.info(
    "Test set class distribution:\n%s (%s percent)",
    test_data["classificacao"].value_counts(),
    100 * test_data["classificacao"].value_counts() / test_data.shape[0],
)

# Add back tiny categories to train set
if not tiny.empty:
    train_data = pd.concat([train_data, tiny_df], ignore_index=True)

# Save the preprocessed datasets
train_data.to_csv("./train.csv", index=False)
test_data.to_csv("./test.csv", index=False)

logging.info("-------------------------------------------")
logging.info("Preprocessing complete. Train and test datasets saved.")

import pandas as pd

train_df = pd.read_csv(
    "../data/fakebr/claim_normalization/gpt-5-nano_2025-11-06_18-00-24/train.csv"
)
test_df = pd.read_csv(
    "../data/fakebr/claim_normalization/gpt-5-nano_2025-11-06_18-00-24/test.csv"
)

total_df = pd.concat([train_df, test_df], ignore_index=True)

# Calculate average text length
total_df["text_length"] = total_df["text"].apply(len)
average_length = total_df["text_length"].mean()
print(f"Average text length: {average_length:.2f} characters")

# Calculate average number of words per text
total_df["word_count"] = total_df["text"].apply(lambda x: len(x.split()))
average_words = total_df["word_count"].mean()
print(f"Average number of words per text: {average_words:.2f} words")

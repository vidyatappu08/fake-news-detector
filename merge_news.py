import pandas as pd

# Load both CSVs
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add label columns
fake_df["label"] = "FAKE"
real_df["label"] = "REAL"

# Combine them
df = pd.concat([fake_df, real_df])

# Keep only text and label columns
df = df[["text", "label"]]

# Save to a new CSV file
df.to_csv("news.csv", index=False)

print("✅ news.csv file created successfully!")

import pandas as pd

# Load original dataset (skip junk rows)
df = pd.read_csv("data.csv", skiprows=2)

# Drop '#' column if present
if '#' in df.columns:
    df = df.drop(columns=['#'])

# Generate synthetic dataset by row sampling (preserves relationships!)
synthetic_data = df.sample(n=3000, replace=True, random_state=42)

# Save
synthetic_data.to_csv("synthetic_claims_dataset.csv", index=False)
print("✅ Fixed synthetic dataset saved as 'synthetic_claims_dataset.csv'")

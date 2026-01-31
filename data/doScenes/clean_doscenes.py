import pandas as pd

# Load the CSV file
df = pd.read_csv("entire_doscenes.csv")

# Drop rows where the "Instruction" column is blank or NaN
df = df.dropna(subset=["Instruction"])
df = df[df["Instruction"].str.strip().astype(bool)]

# Convert "Scene Number" to numeric type (to avoid '1', '10', '2' sorting)
df["Scene Number"] = pd.to_numeric(df["Scene Number"], errors="coerce")

# Drop any rows where Scene Number isn't a valid number
df = df.dropna(subset=["Scene Number"])

# Sort by numeric value of Scene Number
df = df.sort_values(by="Scene Number", ascending=True)

# Save the cleaned and sorted file
df.to_csv("annotated_doscenes.csv", index=False)
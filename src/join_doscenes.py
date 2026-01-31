import pandas as pd
import os

# Define the path where the CSV files are stored
directory = '../doScenes/Annotations'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Read and concatenate all CSV files
dataframes = []
for file in csv_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all the dataframes vertically (stack on top of each other)
combined_df = pd.concat(dataframes, ignore_index=True)

# 🔹 Remove rows where 'Instruction' is missing or blank
combined_df = combined_df.dropna(subset=['Instruction'])                     # remove NaN
combined_df = combined_df[combined_df['Instruction'].astype(str).str.strip() != ""]  # remove empty strings

# Convert 'Scene Number' to numeric and sort
combined_df['Scene Number'] = pd.to_numeric(combined_df['Scene Number'], errors='coerce')
combined_df = combined_df.sort_values(by='Scene Number').reset_index(drop=True)

# Find missing scene numbers
scene_numbers = combined_df['Scene Number'].unique()
all_scene_numbers = set(range(1, 1111))
missing_scene_numbers = all_scene_numbers - set(scene_numbers)

# Print the missing scene numbers
if missing_scene_numbers:
    print(f"Missing scene numbers: {sorted(missing_scene_numbers)}")
else:
    print("No scene numbers are mssing.")

print(f"Total doScene Annotation Missing: {len(missing_scene_numbers)}")

# Save the sorted dataframe to a new CSV file
combined_df.to_csv('sorted_combined_annotations.csv', index=False)

print(f"Sorted dataset saved as 'sorted_combined_annotations.csv' with {len(combined_df)} rows.")

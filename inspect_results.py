import pandas as pd

# Load the csv
df = pd.read_csv('multi_model_benchmark_final.csv')

# Configure pandas to show full column width so we can read the text
pd.set_option('display.max_colwidth', None)

# Print the first few rows for the Triagic model and the Ground Truth
# Also print the Baseline prediction to compare
# Print the second error for Triagic model
print("--- Triagic Error Detail (2nd Error) ---")
error_rows = df[(df['model'] == 'Triagic_Curriculum_Symbolic') & (df['exact_match'] == 0.0)]
if len(error_rows) > 1:
    error_row = error_rows.iloc[1]
    print(f"Question: {error_row['question']}")
    print(f"Type: {error_row['type']}")
    print(f"Truth: {error_row['truth']}")
    print(f"Prediction: {error_row['prediction']}")
    print(f"Cleaned Truth in Prediction?: {str(error_row['truth']).lower() in str(error_row['prediction']).lower()}")
else:
    print("Only one error found.")



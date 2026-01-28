
from datasets import load_dataset
import pandas as pd

def inspect_dataset():
    print("Loading VQA-RAD dataset...")
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded. Size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print("\nSample Item Structure:")
    for key, value in sample.items():
        if key == 'image':
            print(f" - {key}: [Image Object] {value}")
        else:
            print(f" - {key}: {value}")

    # Convert to pandas for quick peek at more rows
    df = pd.DataFrame(dataset.select(range(10)))
    
    # Add derived 'Type' column similar to benchmark script
    df['type'] = df['answer'].apply(lambda x: "CLOSED" if str(x).lower() in ['yes', 'no'] else "OPEN")
    
    print("\nFirst 10 rows with derived types:")
    # Drop image column for printing text
    cols = [c for c in df.columns if c != 'image']
    print(df[cols].to_string())

if __name__ == "__main__":
    inspect_dataset()

import pandas as pd

def parse_line(line):
    # Split the line into word and vector parts
    parts = line.split()
    if len(parts) < 2:
        # Skip lines that do not have at least one word and one vector component
        print(f"Skipping invalid line: {line.strip()}")
        return None, None
    word = parts[0].strip(':')
    # Convert the rest of the parts to floats
    try:
        vector = list(map(float, parts[1:]))
    except ValueError as e:
        print(f"Error converting vector components to float: {e}")
        return None, None
    return word, vector

def create_dataframe(file_path):
    # Initialize lists to store words and their vectors
    words = []
    vectors = []
    
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            word, vector = parse_line(line)
            if word is not None and vector is not None:
                words.append(word)
                vectors.append(vector)
    
    if not words:
        raise ValueError("No valid data was found in the file.")
    
    # Create a DataFrame
    df = pd.DataFrame(vectors)
    
    # Insert the 'word' column at the beginning
    df.insert(0, 'word', words)
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df

# Path to the text file
file_path = 'MaleVectors.txt'

try:
    # Create the DataFrame
    df = create_dataframe(file_path)
    
    # Save DataFrame to CSV
    df.to_csv('MaleVectors.csv', index=False)
    
    # Print the first few rows of the DataFrame
    print(df.head())
except Exception as e:
    print(f"An error occurred: {e}")

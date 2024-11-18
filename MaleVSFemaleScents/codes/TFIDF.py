import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf(file1, file2, output_file):
    # Read content from the two files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        doc1 = f1.read()
        doc2 = f2.read()

    # Combine the documents into a list
    documents = [doc1, doc2]

    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the documents and transform the documents to TF-IDF matrices
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Convert the TF-IDF matrix to a dense format
    tfidf_dense = tfidf_matrix.todense()

    # Convert dense matrix to a DataFrame with words as index
    df_tfidf = pd.DataFrame(tfidf_dense.T, index=feature_names, columns=[f'Document 1', f'Document 2'])

    # Save the DataFrame to a CSV file
    df_tfidf.to_csv(output_file)

    return df_tfidf

# Example usage:
file1 = 'FemaleUnsorted.txt'
file2 = 'MaleUnsorted.txt'
output_file = 'tfidf_output.csv'

# Calculate TF-IDF and save to CSV
tfidf_df = calculate_tfidf(file1, file2, output_file)

# Print the DataFrame
print(tfidf_df)


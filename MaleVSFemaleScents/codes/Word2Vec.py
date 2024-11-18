import gensim.downloader as api
import nltk
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Ensure NLTK's punkt tokenizer is downloaded
nltk.download('punkt')

# Load the pre-trained Word2Vec model
print("Downloading the pre-trained Word2Vec model (word2vec-google-news-300)...")
model = api.load("word2vec-google-news-300")
print("Model loaded successfully.")

# Function to process a single line of text
def process_line(line):
    words = word_tokenize(line.lower())  # Tokenize the line into words
    word_vectors = []
    for word in words:
        if word in model:  # Check if the word is in the Word2Vec model
            word_vectors.append((word, model[word]))
    return word_vectors

# Process the input text file
input_file = 'MaleSorted.txt'
output_file = 'MaleVectors.txt'

print(f"Processing file: {input_file}")
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        word_vectors = process_line(line)
        for word, vector in word_vectors:
            vector_str = ' '.join(map(str, vector))  # Convert vector to a string
            outfile.write(f"{word}: {vector_str}\n")
        outfile.write("\n")  # Add a newline between processed lines

print(f"Word vectors written to: {output_file}")

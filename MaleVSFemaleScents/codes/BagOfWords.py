import string

def create_unique_bag_of_words(input_file, output_file):
    # Read the content of the file
    with open(input_file, 'r') as file:
        text = file.read()
    
    # Normalize the text: convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split the text into words
    words = text.split()
    
    # Remove duplicates by converting the list to a set, then back to a list
    unique_words = sorted(set(words))
    
    # Write the unique words to a new file
    with open(output_file, 'w') as file:
        for word in unique_words:
            file.write(f"{word}\n")

# Example usage
input_file = 'MaleSorted.txt'  # Replace with the path to your input file
output_file = 'BoW_Male.txt'  # Replace with the path to your output file
create_unique_bag_of_words(input_file, output_file)

def remove_duplicates_and_sort(input_file, output_file):
    # Read lines from the input file
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Remove duplicates and sort the lines alphabetically
    unique_sorted_lines = sorted(set(lines))

    # Write the unique, sorted lines to the output file
    with open(output_file, 'w') as outfile:
        outfile.writelines(unique_sorted_lines)

# Example usage:
input_file = 'MaleUnsorted.txt'
output_file = 'MaleSorted.txt'
remove_duplicates_and_sort(input_file, output_file)


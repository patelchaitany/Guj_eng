import csv

# Assuming the TSV file is in directory1 with your Python script
tsv_file_path = "guj_Gujr.tsv"  # Replace with your actual TSV filename
output_tsv_path = "refined.tsv"

# Count total rows and get first three rows
total_rows = 0
first_three_rows = []

punctuation_to_remove = '",!.@\''

# Count total rows and get first three rows
total_rows = 0
first_three_rows = []

try:
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')

        for i, row in enumerate(tsv_reader):
            total_rows += 1
            if i < 3:
                # Process each field in the row to remove punctuation from start and end
                cleaned_row = [field.strip(punctuation_to_remove) for field in row]
                first_three_rows.append(cleaned_row)

    print(f"Total number of rows in {tsv_file_path}: {total_rows}")

    with open(output_tsv_path, 'w', newline='', encoding='utf-8') as output_file:
        tsv_writer = csv.writer(output_file, delimiter='\t')
        for row in first_three_rows:
            tsv_writer.writerow(row)

    print(f"First three rows saved to {output_tsv_path} with punctuation removed")

except FileNotFoundError:
    print(f"Error: File '{tsv_file_path}' not found")
except Exception as e:
    print(f"An error occurred: {str(e)}")


import sys
import os
import csv



def is_valid_file(file_name):
    """Ensure that the input_file exists."""
    if not os.path.exists(file_name):
        print("The file '{}' does not exist!".format(file_name))
        sys.exit(1)


def is_valid_csv(file_name, row_limit):
    """
    Ensure that the # of rows in the input_file
    is greater than the row_limit.
    """
    row_count = 0
    for row in csv.reader(open(file_name)):
        row_count += 1
    # Note: You could also use a generator expression
    # and the sum() function to count the rows:
    # row_count = sum(1 for row in csv.reader(open(file_name)))
    if row_limit > row_count:
        print(
            "The 'row_count' of '{}' is > the number of rows in '{}'!"
            .format(row_limit, file_name)
        )
        sys.exit(1)


def parse_file(input_file,output_path,output_file,row_limit):
    """
    Splits the CSV into multiple files or chunks based on the row_limit.
    Then create new CSV files.
    """
    is_valid_file(input_file)

    is_valid_csv(input_file,row_limit)


    # Read CSV, split into list of lists
    with open(input_file, 'r') as input_csv:
        datareader = csv.reader(input_csv)
        all_rows = []
        for row in datareader:
            all_rows.append(row)

        # Remove header
        header = all_rows.pop(0)

        # Split list of list into chunks
        current_chunk = 1
        for i in range(0, len(all_rows), row_limit):  # Loop through list
            chunk = all_rows[i:i + row_limit]  # Create single chunk

            current_output = os.path.join(  # Create new output file
                output_path,
                "{}-{}.csv".format(output_file, current_chunk)
            )

            # Add header
            chunk.insert(0, header)

            # Write chunk to output file
            with open(current_output, 'w') as output_csv:
                writer = csv.writer(output_csv)
                writer = writer.writerows(chunk)

            # Output info
            print("")
            print("Chunk # {}:".format(current_chunk))
            print("Filepath: {}".format(current_output))
            print("# of rows: {}".format(len(chunk)))

            # Create new chunk
            current_chunk += 1


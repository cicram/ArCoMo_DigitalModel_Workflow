def reverse_branch_segments(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # Markers to identify the start and end of segments
    segment_start_marker = 'Px         Py         Pz'
    in_segment = False
    new_lines = []
    segment_lines = []

    for line in lines:
        if segment_start_marker in line:  # Start of a new segment
            if in_segment and segment_lines:  # If already in a segment, reverse its lines before starting a new one
                new_lines.extend(segment_lines[::-1])
                segment_lines = []
            in_segment = True
            new_lines.append(line)  # Add the segment start marker to new lines
        elif in_segment:
            if line.strip() == "":  # End of the current segment
                in_segment = False
                new_lines.extend(segment_lines[::-1])  # Reverse the current segment's lines and add them to new lines
                segment_lines = []
                new_lines.append(line)  # Add the empty line following the segment
            else:
                segment_lines.append(line)  # Add current line to the segment lines
        else:
            new_lines.append(line)  # Add non-segment lines directly to new lines

    if in_segment and segment_lines:  # Ensure the last segment is also reversed if the file ends without an empty line
        new_lines.extend(segment_lines[::-1])

    # Write the modified content to a new file
    with open(output_filename, 'w') as output_file:
        output_file.writelines(new_lines)

# Example usage
input_filename = 'ArCoMo_Data/ArCoMo1/ArCoMo1_centerline_orig_changed.txt'
output_filename = 'ArCoMo_Data/ArCoMo1/ArCoMo1_centerline.txt'
reverse_branch_segments(input_filename, output_filename)

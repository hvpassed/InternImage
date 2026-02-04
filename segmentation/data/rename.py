import os

# Define the directory containing the files
directory = '/data/home/cwk/workplace/mf/data/val/masks'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    # Check if the file ends with '_m.png'
    if filename.endswith('_m.png'):
        # Create the new filename by removing '_m'
        new_filename = filename.replace('_m.png', '.png')
        # Get the full path of the old and new filenames
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
#!/bin/bash

# Set the directory path
DIR="/workspace/data/safety/video_official"

# Iterate over all files in the directory
for filepath in "$DIR"/*; do
    # Check if it's a file (not a directory)
    if [ -f "$filepath" ]; then
        # Extract the filename from the full path
        filename=$(basename "$filepath")
        
        # Execute the python script with the filename as an argument
        python track_with_depth_jtop_engine.py --input "$filename"
    fi
done


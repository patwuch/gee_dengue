#!/bin/bash

# List all the .ipynb files in the current directory
echo "Available Jupyter Notebooks:"
select notebook in *.ipynb; do
    if [ -n "$notebook" ]; then
        echo "Launching $notebook..."
        # Launch Jupyter Notebook with the selected file
        jupyter notebook "$notebook" --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        break
    else
        echo "Invalid selection, please choose a valid notebook."
    fi
done

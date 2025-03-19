#!/bin/bash

# Check if conda environment is activated
echo 'Checking environment:'
which python  # Should point to the python from the conda environment
which jupyter  # Should point to the jupyter installed in the conda environment

# List notebooks and prompt the user
echo 'Select a notebook to run:'
select notebook in *.ipynb; do
  if [[ -n "$notebook" ]]; then
    echo "Launching $notebook..."
    jupyter notebook "$notebook" --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    break
  else
    echo 'Invalid selection. Try again.'
  fi
done

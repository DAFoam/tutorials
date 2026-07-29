#!/usr/bin/env bash

if [ -z "$WM_PROJECT" ]; then
  echo "OpenFOAM environment not found, forgot to source the OpenFOAM bashrc?"
  exit 1
fi

echo "Running API tests"

# Make the primal run fast for every discovered Python run script.
find . -type f -name 'run*.py' -exec sed -i '/"primalMinResTol":/c\    "primalMinResTol": 0.9,' {} \;

# Discover case directories that contain both the exact preprocessing script
# and at least one Python run script in the same folder.
mapfile -d '' case_dirs < <(
  find . -type f -name 'preProcessing.sh' -print0 | while IFS= read -r -d '' preprocessing_script; do
    case_dir=$(dirname "$preprocessing_script")
    if find "$case_dir" -maxdepth 1 -type f -name 'run*.py' | grep -q .; then
      printf '%s\0' "$case_dir"
    fi
  done | sort -zu
)

if [ ${#case_dirs[@]} -eq 0 ]; then
  echo "No test cases found."
  exit 0
fi

for case_dir in "${case_dirs[@]}"; do
  echo "Discovered case: ${case_dir#./}"

  # Collect all Python run scripts that live beside preProcessing.sh.
  mapfile -t run_scripts < <(find "$case_dir" -maxdepth 1 -type f -name 'run*.py' | sort)

  if [ ${#run_scripts[@]} -eq 0 ]; then
    echo "Skipping ${case_dir#./}: no run scripts found."
    continue
  fi

  (
    cd "$case_dir" || exit 1

    echo "Running preProcessing.sh in ${case_dir#./}"
    ./preProcessing.sh || exit 1

    # Run every discovered Python driver after preprocessing completes.
    for run_script in "${run_scripts[@]}"; do
      script_name=$(basename "$run_script")
      echo "Running ${case_dir#./}/${script_name}"
      python "$script_name" -task=run_model || exit 1
    done
  ) || exit 1
done

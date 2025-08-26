#!/bin/bash

# --- Configuration ---
# The base directory where your sampled data is located.
DATA_DIR="data_sampled"

# The directory where all evaluation output files will be saved.
OUTPUT_DIR="evaluation_results"

# The Anthropic model to use for the evaluation.
EVALUATION_MODEL="claude-3-5-haiku-20241022"

# The number of parallel workers for the Python script.
MAX_WORKERS=50
# --- End of Configuration ---


# Exit immediately if a command exits with a non-zero status.
set -e

# Check if the main data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found."
    echo "Please make sure this script is in the same parent directory as '$DATA_DIR',"
    echo "or update the DATA_DIR variable in this script."
    exit 1
fi

# Create the output directory if it doesn't already exist
mkdir -p "$OUTPUT_DIR"

echo "Starting evaluations..."
echo "Reading from: $DATA_DIR"
echo "Writing to:   $OUTPUT_DIR"
echo "----------------------------------------"


# Find all review model directories (e.g., 4o, o3, etc.)
# This command safely lists only the directories inside the reviews folder.
for model_path in "$DATA_DIR/reviews"/*/; do
    MODEL=$(basename "$model_path")
    echo "--- Processing Model: $MODEL ---"

    # Find all venue directories for this specific model
    for venue_path in "$DATA_DIR/reviews/$MODEL"/*/; do
        VENUE=$(basename "$venue_path")
        echo "  - Processing Venue: $VENUE"

        # The ground truth can be either 'accepted' or 'rejected'
        for STATUS in accepted rejected; do
            REVIEWS_DIR="$DATA_DIR/reviews/$MODEL/$VENUE/$STATUS"
            GROUND_TRUTH_DIR="$DATA_DIR/flawed_papers/$VENUE/$STATUS"

            # Check if both the reviews and ground truth directories exist for this status
            if [ -d "$REVIEWS_DIR" ] && [ -d "$GROUND_TRUTH_DIR" ]; then
                echo "    - Evaluating status: $STATUS"

                # Create a clean, descriptive name for the output file
                OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_${VENUE}_${STATUS}_evaluated.json"

                echo "      Running evaluation command..."
                
                # Execute the python script with the constructed paths
                python evaluate_reviews_with_anthropic.py \
                    --reviews_dir "$REVIEWS_DIR" \
                    --ground_truth_dir "$GROUND_TRUTH_DIR" \
                    --output_file "$OUTPUT_FILE" \
                    --max_workers "$MAX_WORKERS" \
                    --model_name "$EVALUATION_MODEL" \
                    --verbose
                
                echo "      Evaluation complete. Output saved to $OUTPUT_FILE"
                echo "----------------------------------------"

            else
                # This message will appear if, for example, a venue has 'accepted' papers but no 'rejected' ones.
                echo "    - Skipping status '$STATUS' (directory not found)."
            fi
        done
    done
done

echo "--- All evaluations finished successfully! ---"
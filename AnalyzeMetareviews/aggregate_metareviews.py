import json
import os
import re
from collections import defaultdict

def load_json_from_file(filepath):
    """
    Loads JSON data from a single file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data, or None if an error occurs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {filepath}: {e}")
        return None

def aggregate_flaw_data(base_directory, output_directory):
    """
    Scans a directory for JSON files, extracts all flaw data objects,
    and concatenates them into new JSON files grouped by model, venue, and year.
    The output is organized into subdirectories for each model.

    Args:
        base_directory (str): The path to the root directory containing the JSON files.
        output_directory (str): The path to the directory where aggregated files will be saved.
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(output_directory)}")

    # A dictionary to hold aggregated flaw objects, e.g., {('gpt-4o', 'neurips', '2023'): [flaw1, flaw2]}
    aggregated_data = defaultdict(list)

    print(f"Scanning for data in: {os.path.abspath(base_directory)}")
    if not os.path.exists(base_directory):
        print(f"Warning: Base directory not found at {base_directory}")
        return

    # Walk through the directory structure
    for root, _, files in os.walk(base_directory):
        # Skip the root directory itself from model name extraction
        if os.path.abspath(root) == os.path.abspath(base_directory):
            continue

        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)

                # --- Extract Model, Venue, and Year from the path ---
                
                # Extract Model Name
                # This assumes the model name is the first directory inside BASE_DIRECTORY
                try:
                    rel_path = os.path.relpath(root, base_directory)
                    model_name = rel_path.split(os.sep)[0]
                except IndexError:
                    print(f"Warning: Could not determine model name from path, skipping: {filepath}")
                    continue

                # Extract Venue and Year
                match = re.search(r'(iclr|nips|neurips|icml)(\d{4})', root, re.IGNORECASE)
                if not match:
                    print(f"Warning: Could not determine venue and year from path, skipping: {filepath}")
                    continue

                venue = match.group(1).lower()
                year = match.group(2)

                # Normalize NeurIPS naming
                if venue == 'nips':
                    venue = 'neurips'

                # Load the individual JSON file
                data = load_json_from_file(filepath)
                if data:
                    # The data is structured as {paper_id: [flaw_objects]}
                    # We iterate through all paper IDs and add their flaw objects to our aggregate list.
                    for paper_id, flaws in data.items():
                        if isinstance(flaws, list):
                            # Extend the main list with the list of flaw objects for this paper
                            aggregation_key = (model_name, venue, year)
                            aggregated_data[aggregation_key].extend(flaws)

    if not aggregated_data:
        print("Warning: No flaw data was found. Check the directory structure and file content.")
        return

    # Write the aggregated data to new JSON files inside model-specific folders
    for (model, venue, year), all_flaws in aggregated_data.items():
        # Create a specific directory for the model if it doesn't exist
        model_output_dir = os.path.join(output_directory, model)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Define the output filename and path
        output_filename = f"{venue}_{year}_aggregated_flaws.json"
        output_filepath = os.path.join(model_output_dir, output_filename)

        print(f"Writing {len(all_flaws)} flaw records to {output_filepath}")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(all_flaws, f, indent=4)
        except IOError as e:
            print(f"Error writing to file {output_filepath}: {e}")

if __name__ == '__main__':
    # The root directory where the original JSON files are located.
    # It should contain subdirectories for each model, venue, and year.
    BASE_DIRECTORY = './metareviews'

    # The directory where the combined JSON files will be saved.
    OUTPUT_DIRECTORY = './aggregated_metareviews'

    aggregate_flaw_data(BASE_DIRECTORY, OUTPUT_DIRECTORY)

    print("\nAggregation complete.")

import os
import random
import shutil
import argparse
import json

def sample_data(source_base_dir, dest_base_dir, num_samples_per_category, update_metareviews_only=False):
    """
    Samples papers, reviews, and metareviews. Can also update metareviews for an existing sample.

    Args:
        source_base_dir (str): The path to the source data directory.
        dest_base_dir (str): The path to the destination directory for the sampled data.
        num_samples_per_category (int): The number of papers to sample from EACH of
                                        the 'accepted' and 'rejected' folders.
        update_metareviews_only (bool): If True, skips paper/review sampling and only
                                         processes metareviews for an existing sample.
    """
    # --- Setup source and destination paths ---
    flawed_papers_src = os.path.join(source_base_dir, 'flawed_papers')
    reviews_src = os.path.join(source_base_dir, 'reviews')
    metareviews_src = os.path.join(source_base_dir, 'metareviews')

    flawed_papers_dest = os.path.join(dest_base_dir, 'flawed_papers')
    reviews_dest = os.path.join(dest_base_dir, 'reviews')
    metareviews_dest = os.path.join(dest_base_dir, 'metareviews')

    all_sampled_paper_ids = set()

    if not update_metareviews_only:
        # --- Part 1: Sample Flawed Papers and Copy Reviews ---
        print("--- Part 1: Sampling Flawed Papers and Reviews ---")
        os.makedirs(flawed_papers_dest, exist_ok=True)
        os.makedirs(reviews_dest, exist_ok=True)

        try:
            conference_dirs = [d for d in os.listdir(flawed_papers_src) if os.path.isdir(os.path.join(flawed_papers_src, d))]
        except FileNotFoundError:
            print(f"Error: The source directory '{flawed_papers_src}' was not found.")
            print("Please check the --input_dir path.")
            return

        for conf_dir in conference_dirs:
            if 'ICML2024' in conf_dir:
                print(f"Skipping {conf_dir} as requested.")
                continue

            print(f"Processing Venue: {conf_dir}...")
            conf_src_path = os.path.join(flawed_papers_src, conf_dir)
            
            sampled_for_this_venue = []
            
            for status_folder in ['accepted', 'rejected']:
                print(f"  - Looking in '{status_folder}' folder...")
                subdir_path = os.path.join(conf_src_path, status_folder)

                if os.path.exists(subdir_path):
                    papers_in_subdir = [p for p in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, p))]
                    if not papers_in_subdir:
                        print(f"    No papers found in '{status_folder}'.")
                        continue

                    num_to_sample = min(num_samples_per_category, len(papers_in_subdir))
                    print(f"    Found {len(papers_in_subdir)} papers. Sampling {num_to_sample}.")
                    
                    papers_sampled_from_subdir = random.sample(papers_in_subdir, num_to_sample)
                    
                    for paper_name in papers_sampled_from_subdir:
                        all_sampled_paper_ids.add(paper_name)
                        sampled_for_this_venue.append((paper_name, status_folder))
                else:
                    print(f"    Directory not found: {subdir_path}")

            if not sampled_for_this_venue:
                print(f"  No papers were sampled for {conf_dir}. Skipping.")
                continue

            print(f"\n  Copying files for {len(sampled_for_this_venue)} sampled papers from {conf_dir}...")
            for paper_name, original_subdir in sampled_for_this_venue:
                # Copy flawed paper directory
                src_paper_path = os.path.join(conf_src_path, original_subdir, paper_name)
                dest_paper_path = os.path.join(flawed_papers_dest, conf_dir, original_subdir, paper_name)
                os.makedirs(os.path.dirname(dest_paper_path), exist_ok=True)
                if os.path.exists(src_paper_path):
                    shutil.copytree(src_paper_path, dest_paper_path)

                # Copy corresponding review directories
                if os.path.exists(reviews_src):
                    for model_dir in os.listdir(reviews_src):
                        model_src_path = os.path.join(reviews_src, model_dir)
                        if os.path.isdir(model_src_path):
                            review_conf_dir_src = os.path.join(model_src_path, conf_dir, original_subdir, paper_name)
                            if os.path.exists(review_conf_dir_src):
                                review_conf_dir_dest = os.path.join(reviews_dest, model_dir, conf_dir, original_subdir, paper_name)
                                os.makedirs(os.path.dirname(review_conf_dir_dest), exist_ok=True)
                                shutil.copytree(review_conf_dir_src, review_conf_dir_dest)
    else:
        # If updating, populate paper IDs from the existing destination folder
        print("--- Update Mode: Identifying existing sampled papers ---")
        if not os.path.exists(flawed_papers_dest):
            print(f"Error: Cannot update metareviews because the destination folder '{flawed_papers_dest}' does not exist.")
            return
        
        for dirpath, dirnames, _ in os.walk(flawed_papers_dest):
            # The paper IDs are the directory names at the deepest level
            for dirname in dirnames:
                # A simple check to see if it looks like a paper ID folder
                if '_' in dirname and len(dirname) > 10:
                     all_sampled_paper_ids.add(dirname)


    # --- Part 2: Sample Metareviews ---
    print("\n--- Part 2: Processing Metareviews ---")
    if not os.path.exists(metareviews_src):
        print("Skipping: 'metareviews' directory not found in source.")
    elif not all_sampled_paper_ids:
        print("Skipping: No sampled paper IDs found to filter metareviews by.")
    else:
        print(f"Found {len(all_sampled_paper_ids)} unique paper IDs to filter metareviews by.")
        # Use os.walk to recursively find all .json files in the metareviews directory
        for dirpath, _, filenames in os.walk(metareviews_src):
            for filename in filenames:
                if filename.endswith('.json'):
                    src_json_path = os.path.join(dirpath, filename)
                    
                    # Determine the correct destination path to preserve the subdirectory structure
                    relative_path = os.path.relpath(dirpath, metareviews_src)
                    dest_dir = os.path.join(metareviews_dest, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_json_path = os.path.join(dest_dir, filename)
                    
                    print(f"  - Processing metareview file: {src_json_path}")
                    
                    try:
                        with open(src_json_path, 'r', encoding='utf-8') as f:
                            source_data = json.load(f)
                        
                        # Filter the JSON data based on the sampled paper IDs
                        filtered_data = {
                            paper_id: evals
                            for paper_id, evals in source_data.items()
                            if paper_id in all_sampled_paper_ids
                        }
                        
                        if filtered_data:
                            print(f"    - Writing {len(filtered_data)} sampled entries to {dest_json_path}")
                            with open(dest_json_path, 'w', encoding='utf-8') as f:
                                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
                        else:
                            print(f"    - No matching sampled papers found in {filename}. Skipping.")

                    except (json.JSONDecodeError, IOError) as e:
                        print(f"    - Warning: Could not process file {src_json_path}. Error: {e}")

    print("\n--- Script complete! ---")
    print(f"Data is located in: {dest_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample papers, reviews, and metareviews from a dataset.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The source directory containing 'flawed_papers', 'reviews', and optionally 'metareviews'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The destination directory where the sampled data will be saved or updated."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="The number of papers to sample from EACH of the 'accepted' and 'rejected' folders. Default is 15."
    )
    parser.add_argument(
        "--update_metareviews",
        action="store_true",
        help="If specified, skips paper/review sampling and only adds/updates metareviews for the existing sample in the output directory."
    )

    args = parser.parse_args()

    if not args.update_metareviews and os.path.exists(args.output_dir):
        print(f"Error: Destination directory '{args.output_dir}' already exists.")
        print("Please remove it or choose a different output directory before running in sampling mode.")
    else:
        sample_data(args.input_dir, args.output_dir, args.num_samples, args.update_metareviews)

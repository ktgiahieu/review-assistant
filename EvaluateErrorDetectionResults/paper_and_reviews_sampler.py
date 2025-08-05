import os
import random
import shutil
import argparse

def sample_data(source_base_dir, dest_base_dir, num_samples):
    """
    Samples data from the source directory and copies it to the destination directory.

    Args:
        source_base_dir (str): The path to the source data directory.
        dest_base_dir (str): The path to the destination directory for the sampled data.
        num_samples (int): The number of papers to sample from each conference/year.
    """
    flawed_papers_src = os.path.join(source_base_dir, 'flawed_papers')
    reviews_src = os.path.join(source_base_dir, 'reviews')

    flawed_papers_dest = os.path.join(dest_base_dir, 'flawed_papers')
    reviews_dest = os.path.join(dest_base_dir, 'reviews')

    # Create destination directories if they don't exist
    os.makedirs(flawed_papers_dest, exist_ok=True)
    os.makedirs(reviews_dest, exist_ok=True)

    # Get all conference/year directories from flawed_papers
    try:
        conference_dirs = [d for d in os.listdir(flawed_papers_src) if os.path.isdir(os.path.join(flawed_papers_src, d))]
    except FileNotFoundError:
        print(f"Error: The source directory '{flawed_papers_src}' was not found.")
        print("Please check the --input_dir path.")
        return

    for conf_dir in conference_dirs:
        # Skip the specified ICML2024 directory
        if 'ICML2024' in conf_dir:
            print(f"Skipping {conf_dir} as requested.")
            continue

        print(f"Processing {conf_dir}...")
        conf_src_path = os.path.join(flawed_papers_src, conf_dir)
        
        # We assume papers are in 'accepted' or 'rejected' subdirectories
        paper_subdirs = ['accepted', 'rejected']
        all_papers = []
        for subdir in paper_subdirs:
            subdir_path = os.path.join(conf_src_path, subdir)
            if os.path.exists(subdir_path):
                papers_in_subdir = [p for p in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, p))]
                # Store the paper name and its original subdir path
                all_papers.extend([(p, subdir) for p in papers_in_subdir])

        if not all_papers:
            print(f"  No papers found in {conf_dir}. Skipping.")
            continue
            
        # Sample the papers
        num_to_sample = min(num_samples, len(all_papers))
        sampled_papers = random.sample(all_papers, num_to_sample)
        print(f"  Sampling {len(sampled_papers)} papers from {conf_dir}.")

        # Copy sampled flawed papers and their corresponding reviews
        for paper_name, original_subdir in sampled_papers:
            # 1. Copy the flawed paper directory
            src_paper_path = os.path.join(conf_src_path, original_subdir, paper_name)
            dest_conf_path = os.path.join(flawed_papers_dest, conf_dir, original_subdir)
            os.makedirs(dest_conf_path, exist_ok=True)
            dest_paper_path = os.path.join(dest_conf_path, paper_name)
            
            if os.path.exists(src_paper_path):
                shutil.copytree(src_paper_path, dest_paper_path)
            else:
                print(f"  Warning: Source paper path not found: {src_paper_path}")
                continue

            # 2. Copy the corresponding review directories from all models
            for model_dir in os.listdir(reviews_src):
                model_src_path = os.path.join(reviews_src, model_dir)
                if os.path.isdir(model_src_path):
                    review_conf_dir_src = os.path.join(model_src_path, conf_dir, original_subdir, paper_name)
                    
                    if os.path.exists(review_conf_dir_src):
                        review_conf_dir_dest = os.path.join(reviews_dest, model_dir, conf_dir, original_subdir, paper_name)
                        # The parent directory is created when copying the flawed paper, but this ensures it exists
                        os.makedirs(os.path.dirname(os.path.dirname(review_conf_dir_dest)), exist_ok=True)
                        shutil.copytree(review_conf_dir_src, review_conf_dir_dest)
                    else:
                        # It's possible a model didn't generate a review for every paper
                        pass
    
    print("\nSampling complete!")
    print(f"Sampled data is located in: {dest_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample papers and reviews from a dataset.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The source directory containing the 'flawed_papers' and 'reviews' folders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The destination directory where the sampled data will be saved."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="The number of papers to sample from each conference/year venue. Default is 15."
    )

    args = parser.parse_args()

    # Ensure the destination directory doesn't exist to avoid merging with old data
    if os.path.exists(args.output_dir):
        print(f"Error: Destination directory '{args.output_dir}' already exists.")
        print("Please remove it or choose a different output directory before running.")
    else:
        sample_data(args.input_dir, args.output_dir, args.num_samples)
import os
import subprocess
import sys
import argparse

def run_for_all_datasets(folder_path, algorithm, mode, k, all_seeds):
    # Get all subdirectories in the given folder and remove "_dataset" suffix if present
    datasets = [d.removesuffix("_dataset") for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # Construct the command using the current Python interpreter
        command = [
            sys.executable,  # Use the current Python interpreter
            "scripts/fast_dawid_skene.py",
            "--dataset", dataset,
            "--algorithm", algorithm,
            "--mode", mode,
            "--k", str(k)
        ]
        
        if all_seeds:
            command.append("-a")
        
        # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing dataset {dataset}: {e}")
        
        print(f"Finished processing dataset: {dataset}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Fast Dawid-Skene for all datasets in a folder')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing datasets')
    parser.add_argument('--algorithm', type=str, choices=['DS', 'FDS', 'H', 'MV'], required=True,
                        help='Algorithm to use - DS: Dawid-Skene, FDS: Fast-Dawid Skene, H: Hybrid, MV: Majority Voting')
    parser.add_argument('--mode', default='aggregate', type=str, choices=['aggregate', 'test'],
                        help='The mode to run this program - aggregate: obtain aggregated dataset, test: aggregate data and compare with ground truths. Default is aggregate')
    parser.add_argument('--k', default=0, type=int,
                        help='Number of annotators to use. Default is 0')
    parser.add_argument('-a', '--all', action='store_true',
                        help='Run 10 seeds', dest='all_seeds', default=False)
    
    args = parser.parse_args()
    
    try:
        run_for_all_datasets(args.folder_path, args.algorithm, args.mode, args.k, args.all_seeds)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        input("Press Enter to exit...")

    sys.exit(0)
    

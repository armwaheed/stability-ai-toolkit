# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# huggingface_upload.py
# Pytorch to ONNX Model Conversion
#
# Created by Waheed Brown on April 24, 2025

"""
Script to upload a folder (containing your large model files) to a Hugging Face Hub repository,
while skipping files in any .cache folder.

Usage:
    python upload_model.py armwaheed/stable-diffusion-3.5-medium-onnx [path/to/model/folder] [--commit_message "Your commit message"] [--token YOUR_HF_TOKEN]

If no folder is specified, the current directory is used.
"""

import os
import argparse
from huggingface_hub import upload_folder

def create_ignore_files(folder: str) -> None:
    """
    Create a .huggingfaceignore file (if it does not exist) that tells the uploader to ignore
    the .cache folder, and also create a simple .gitattributes to track common large-file extensions
    using Git LFS (adjust these settings as needed).
    """
    # Create .huggingfaceignore to ignore all contents of any .cache/ folder
    hf_ignore_path = os.path.join(folder, ".huggingfaceignore")
    if not os.path.exists(hf_ignore_path):
        with open(hf_ignore_path, "w") as ignore_file:
            ignore_file.write(".cache/\n")
        print(f"Created {hf_ignore_path} to ignore the .cache folder.")
    else:
        print(f"{hf_ignore_path} already exists. Please verify it ignores unwanted files.")
    
    # (Optional) Create .gitattributes for Git LFS tracking of large files.
    # Adjust the patterns to match your file types (for example .onnx, .bin, etc.)
    gitattributes_path = os.path.join(folder, ".gitattributes")
    if not os.path.exists(gitattributes_path):
        with open(gitattributes_path, "w") as attr_file:
            attr_file.write("*.onnx filter=lfs diff=lfs merge=lfs -text\n")
            attr_file.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        print(f"Created {gitattributes_path} to track large files with Git LFS.")
    else:
        print(f"{gitattributes_path} already exists. Please confirm it meets your needs.")

def main():
    parser = argparse.ArgumentParser(
        description="Upload a folder of model files to a Hugging Face Hub repository, "
                    "skipping any files under .cache/"
    )
    parser.add_argument(
        "repo_id",
        help="The Hugging Face Hub repository id (for example: armwaheed/stable-diffusion-3.5-medium-onnx)"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="The local folder containing the model files (default: current directory)."
    )
    parser.add_argument(
        "--commit_message",
        default="Upload model files",
        help="Commit message for the upload."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token. If not specified, the script will use your environment configuration."
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.folder)

    # Create/ensure ignore files (to skip .cache and optionally set up Git LFS tracking)
    create_ignore_files(model_dir)

    print(f"Uploading folder:\n  {model_dir}\nto repository:\n  {args.repo_id}")
    
    try:
        # The ignore_regex below ignores any file whose relative path starts with ".cache/"
        upload_folder(
            folder_path=model_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            token=args.token
        )
        print("Upload completed successfully.")
    except Exception as e:
        print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()


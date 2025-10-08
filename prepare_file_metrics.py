import argparse
import re
import os
from pathlib import Path
import stitch

def main(args):
    val_src = args.val_folder
    dest_folder_gt = args.dest_folder_gt
    grp_regex = re.compile(args.grp_regex)
    src_folder = args.src_folder

    patient_id = set()
    for file in os.listdir(val_src):
        match: re.Match = grp_regex.match(file)
        if not match:
            print(f"Skipping {file} as it does not match the regex")
            continue

        patient_id.add(match.group(1))

    for patient in os.listdir(src_folder):
        if patient not in patient_id:
            continue
        if (dest_folder_gt / f'{patient}.nii.gz').exists():
            continue
        os.symlink(src_folder / patient / 'GT.nii.gz', dest_folder_gt / f'{patient}.nii.gz')

    args.dest_folder = args.dest_folder_val
    stitch.main(args)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='File path parameters')
    parser.add_argument('--val_folder', type=Path, required=True)
    parser.add_argument('--src_folder', type=Path, required=True)
    parser.add_argument('--grp_regex', type=str, required=True)
    parser.add_argument('--data_folder', type=Path, required=True)
    parser.add_argument('--dest_folder_gt', type=Path, required=True)
    parser.add_argument('--dest_folder_val', type=Path, required=True)
    parser.add_argument('--source_scan_pattern', type=str, required=True,
                        help="The pattern to get the original scan. This is used to get the correct metadata")
    parser.add_argument('--num_classes', type=int, default=5)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())

import argparse
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from postprocessing import postprocess

thresholds = [0, 20, 100, 500, 1000, 2000]

def main(args):

    src_folder = args.src_folder
    dest_folder = args.dest_folder
    threshold = {cls: args.threshold for cls in range(1, 5) if args.threshold > 0}

    os.makedirs(dest_folder, exist_ok=True)

    for patient_file in os.listdir(src_folder):
        if not patient_file.endswith('.nii.gz'):
            continue

        pred_path = src_folder / patient_file

        pred = nib.load(pred_path)
        pred = np.asarray(pred.dataobj)

        config = {
            'cca': True,
            'cca_threshold': threshold
        }
        processed_pred = postprocess(pred, config)

        nib.save(
            nib.Nifti1Image(processed_pred,
                            header=pred.header,
                            affine=pred.affine,
                            dtype=pred.get_data_dtype()),
            (dest_folder / patient_file)
        )

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='File path parameters')
    parser.add_argument('--src_folder', type=Path, required=True)
    parser.add_argument('--dest_folder', type=Path, required=True)
    parser.add_argument('--threshold', type=int, required=True)

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())

import argparse
import os
from pathlib import Path
import nibabel as nib
import numpy as np
from postprocessing import postprocess

def main(args):

    src_folder = args.src_folder
    dest_folder = args.dest_folder
    if args.threshold is not None:
        if args.threshold == - 1:
            threshold = {
                3: 20,
                4: 20
            }
        else:
            threshold = {
                cls: args.threshold
                for cls in range(1, 5) if args.threshold > 0
            }
    else:
        threshold = None

    os.makedirs(dest_folder, exist_ok=True)

    for patient_file in os.listdir(src_folder):
        if not patient_file.endswith('.nii.gz'):
            continue

        pred_path = src_folder / patient_file

        pred_nib = nib.load(pred_path)
        pred = np.asarray(pred_nib.dataobj)

        config = {
            'cca': True,
            'cca_threshold': threshold,
            'fill_holes': True
        }
        processed_pred = postprocess(pred, config)

        nib.save(
            nib.Nifti1Image(processed_pred,
                            header=pred_nib.header,
                            affine=pred_nib.affine,
                            dtype=pred_nib.get_data_dtype()),
            (dest_folder / patient_file)
        )

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='File path parameters')
    parser.add_argument('--src_folder', type=Path, required=True)
    parser.add_argument('--dest_folder', type=Path, required=True)
    parser.add_argument('--threshold', type=int, required=False)
    parser.add_argument('--connectivity', type=int, required=False)
    parser.add_argument('--radius', type=int, required=False)
    parser.add_argument('--ignore_classes_fill_holes', type=int, nargs='+', required=False, default=[])
    parser.add_argument('--ignore_classes_cca', type=int, nargs='+', required=False, default=[])

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())

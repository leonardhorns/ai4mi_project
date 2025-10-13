import argparse
import os
import re
from collections import defaultdict
import nibabel as nib
import numpy as np
import skimage.transform
from tqdm import tqdm
from pprint import pprint

parser = argparse.ArgumentParser(description="Stitch predicted 2D segmentation class probabilities back into 3D volumes.")
parser.add_argument("--data_folders", type=str, nargs='+', required=True, help="Directory containing the 2D slices of predicted probabilities.")
parser.add_argument("--views", type=str, nargs='+', required=True, help="Views of data_folders (in the same order). Options: axial, sagittal, coronal.")
parser.add_argument("--dest_folder", type=str, required=True, help="Directory to save the stitched 3D volumes.")
parser.add_argument("--grp_regex", type=str, required=True, help="Regex pattern to group slices into volumes, containing a capturing group for the patient id.")
parser.add_argument("--source_scan_pattern", type=str, default=None, help="Pattern to identify source scans, containing '{id_}' as a placeholder for the patient id from the regex.")


view_axes = {
    'sagittal': 0,
    'coronal': 1,
    'axial': 2,
}

def drop_idx(t: tuple, idx: int) -> tuple:
    return t[:idx] + t[idx+1:]


def main(args):
    data_base_paths = args.data_folders
    for path in data_base_paths:
        if not os.path.isdir(path):
            raise ValueError(f"{path} does not exist os is not a directory")
    if len(data_base_paths) != len(args.views):
        raise ValueError("Number of data folders must match number of views")
    
    pattern = re.compile(args.grp_regex)
    patient_predictions = [defaultdict(list) for _ in data_base_paths]
    for data_base_path, prediction_dict in zip(data_base_paths, patient_predictions):
        for f in os.listdir(data_base_path):
            if (match := pattern.match(f)) is not None:
                prediction_dict[match.group(1)].append(f)
    patient_ids = set(patient_predictions[0].keys())
    assert all(patient_ids == set(d.keys()) for d in patient_predictions), "The grouping regex does not yield the same patient ids across all data folders"
    
    dest_base_path = args.dest_folder
    os.makedirs(dest_base_path, exist_ok=True)
    
    stitch_axes = [view_axes[v] for v in args.views]
    for patient_id in tqdm(patient_ids):
        affine, header = None, None
        first_folder_files = sorted(patient_predictions[0][patient_id])
        first_pred_path = os.path.join(data_base_paths[0], first_folder_files[0])
        n_classes = int(np.load(first_pred_path, allow_pickle=False).shape[0])
        
        if args.source_scan_pattern is not None:
            gt_path = args.source_scan_pattern.replace("{id_}", patient_id)
            gt = nib.load(gt_path)
            gt_shape = gt.get_fdata().shape
            affine, header = gt.affine, gt.header
        else:
            gt_shape = [0, 0, 0]
            dims = (len(prediction_dict[patient_id]) for prediction_dict in patient_predictions)
            for dim, axis in zip(dims, stitch_axes):
                gt_shape[axis] = dim
            gt_shape = tuple(gt_shape)
        
        stitched = np.zeros((n_classes,) + gt_shape, dtype=np.float32)
        for data_base_path, prediction_dict, stitch_axis in zip(data_base_paths, patient_predictions, stitch_axes):
            prediction_slice_names = prediction_dict[patient_id]
            prediction_paths = (os.path.join(data_base_path, f) for f in sorted(prediction_slice_names))
            
            for i, p in enumerate(prediction_paths):
                sl = np.load(p, allow_pickle=False).astype(np.float32)
                sl = skimage.transform.resize(sl, drop_idx(stitched.shape, stitch_axis+1), order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
                if stitch_axis == 0:
                    stitched[:, i, :, :] += sl
                elif stitch_axis == 1:
                    stitched[:, :, i, :] += sl
                else:
                    stitched[:, :, :, i] += sl
        
        combined_preds = stitched.argmax(axis=0).astype(np.uint8)
        prediction_nii = nib.Nifti1Image(combined_preds, affine, header)
        dest_path = os.path.join(dest_base_path, f"{patient_id}.nii.gz")
        nib.save(prediction_nii, dest_path)


if __name__ == "__main__":
    args = parser.parse_args()
    pprint(args)
    main(args)

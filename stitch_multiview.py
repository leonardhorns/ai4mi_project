import argparse
import os
import re
from collections import defaultdict
import nibabel as nib
import numpy as np
import skimage.transform
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Stitch predicted 2D segmentation class probabilities back into 3D volumes.")
parser.add_argument("--data_folders", type=str, nargs='+', required=True, help="Directory containing the 2D slices of predicted probabilities.")
parser.add_argument("--views", type=str, nargs='+', required=True, help="Views of data_folders (in the same order). Options: axial, sagittal, coronal.")
parser.add_argument("--dest_folder", type=str, required=True, help="Directory to save the stitched 3D volumes.")
parser.add_argument("--grp_regex", type=str, required=True, help="Regex pattern to group slices into volumes, containing a capturing group for the patient id.")
parser.add_argument("--source_scan_pattern", type=str, required=True, help="Pattern to identify source scans, containing '{id_}' as a placeholder for the patient id from the regex.")


view_axes = {
    'sagittal': 0,
    'coronal': 1,
    'axial': 2,
}

def drop_idx(t: tuple, idx: int) -> tuple:
    return t[:idx] + t[idx+1:]

if __name__ == "__main__":
    args = parser.parse_args()
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
    
    for patient_id in tqdm(patient_ids):
        gt_path = args.source_scan_pattern.replace("{id_}", patient_id)
        gt = nib.load(gt_path)
        gt_shape = gt.get_fdata().shape
        
        view_probs = []
        stitch_axes = (view_axes[v] for v in args.views)
        for data_base_path, prediction_dict, stitch_axis in zip(data_base_paths, patient_predictions, stitch_axes):
            prediction_slice_names = prediction_dict[patient_id]
            prediction_paths = (os.path.join(data_base_path, f) for f in sorted(prediction_slice_names))
            predictions = (np.load(p) for p in prediction_paths)
            stitched = np.stack(list(predictions), axis=stitch_axis + 1) # dimension 0 is prediction class
            stitched = skimage.transform.resize(stitched, (stitched.shape[0],) + gt_shape, order=0, preserve_range=True, anti_aliasing=False)
            view_probs.append(stitched)
        
        combined_probs = np.mean(np.stack(view_probs, axis=0), axis=0)
        combined_preds = combined_probs.argmax(axis=0).astype(np.uint8)
        # combined_preds = (skimage.transform.resize(combined_preds, gt_shape, order=0, preserve_range=True, anti_aliasing=False)).astype(np.uint8)
        prediction_nii = nib.Nifti1Image(combined_preds, gt.affine, gt.header)
        dest_path = os.path.join(dest_base_path, f"{patient_id}.nii.gz")
        nib.save(prediction_nii, dest_path)

import numpy as np
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure, iterate_structure


def postprocess(volume, config):

    if not isinstance(volume, np.ndarray):
        volume = volume.cpu().numpy()

    if config.get('cca', False):
        connected_components = get_connected_components(volume)
        final_volume = np.zeros_like(volume)

        for c, labeled in connected_components.items():
            components_size = np.bincount(labeled.flatten())
            if c in config['cca_threshold']:
                #for esophagus pick size > 500
                filtered_component = np.where(components_size[1:] > config['cca_threshold'][c])[0] + 1
                mask = np.isin(labeled, filtered_component)
            else:
                #pick largest component for other organs
                largest_component = components_size[1:].argmax() + 1
                mask = (labeled == largest_component)

            final_volume[mask] = c

        volume = final_volume

    if config.get('fill_holes', False):
        filled_volume = np.zeros_like(volume)
        structure = generate_binary_structure(len(volume.shape), config.get('connectivity', 1))
        structure = iterate_structure(structure, config.get('radius', 1))
        for cls in range(1, volume.max() + 1):
            mask = (volume == cls)
            mask_filled = binary_fill_holes(mask, structure=structure)
            filled_volume[mask_filled] = cls

        volume = filled_volume

    return volume

def get_connected_components(volume):

    connected_components = {}
    structure = np.ones((3, 3, 3))
    for c in range(1, volume.max() + 1):
        mask = (volume == c)
        labeled, _ = label(mask, structure)

        connected_components[c] = labeled

    return connected_components


def dice_score_3d(pred, target, smooth=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def multiclass_dice(pred, target, num_classes):
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        dice = dice_score_3d(pred_c, target_c)
        dice_scores.append(float(dice))
    return dice_scores

def print_dice_scores(dice_scores):
    class_names = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]

    print("🩻 Dice Scores per Class")
    print("| Class       | Dice Score |")
    print("|--------------|------------:|")

    for name, score in zip(class_names, dice_scores):
        print(f"| {name:<12} | {score:.4f} |")

    # Compute mean (excluding background)
    mean_dice = np.mean(dice_scores[1:])
    print(f"\nMean (excluding background): {mean_dice:.4f}")

# def apply_morphology(segmentation_mask, operation, radius):
#
#     mask = segmentation_mask.cpu().numpy()
#     processed = np.zeros_like(mask)
#     selem = morphology.ball(radius)
#     for c in range(num_classes):
#         mask = one_hot[c].astype(bool)
#         if 'opening' in operation:
#             mask = morphology.binary_opening(mask, selem)
#         if 'closing' in operation:
#             mask = morphology.binary_closing(mask, selem)
#         processed[c] = mask
#
#     return torch.tensor(processed).to(segmentation_mask.device)
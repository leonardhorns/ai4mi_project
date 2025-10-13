import numpy as np
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure, iterate_structure


def postprocess(volume, config):

    if not isinstance(volume, np.ndarray):
        volume = volume.cpu().numpy()

    if config.get('cca', False):
        print('Performing CCA')
        connected_components = get_connected_components(volume)
        final_volume = np.zeros_like(volume)

        for c, labeled in connected_components.items():
            if c in config.get('ignore_classes_cca', []):
                print(f'Skipping CCA for class {c}')
                final_volume[volume == c] = c
                continue
            components_size = np.bincount(labeled.flatten())
            if c in config['cca_threshold']:
                print(f'Performing CCA for class {c} with threshold {config["cca_threshold"][c]}')
                filtered_component = np.where(components_size[1:] > config['cca_threshold'][c])[0] + 1
                mask = np.isin(labeled, filtered_component)
            else:
                print(f'Performing CCA for class {c} with largest component')
                largest_component = components_size[1:].argmax() + 1
                mask = (labeled == largest_component)

            final_volume[mask] = c

        volume = final_volume

    if config.get('fill_holes', False):
        print(f'Performing filling holes with connectivity {config.get("connectivity", 1)} and radius {config.get("radius", 1)}')
        filled_volume = np.zeros_like(volume)
        structure = generate_binary_structure(len(volume.shape), config.get('connectivity', 1))
        structure = iterate_structure(structure, config.get('radius', 1))
        for cls in range(1, volume.max() + 1):
            if cls in config.get('ignore_classes_fill_holes', []):
                print(f'Skipping fill_holes for class {cls}')
                filled_volume[volume == cls] = cls
                continue

            mask = (volume == cls)
            mask_filled = binary_fill_holes(mask, structure=structure)
            filled_volume[mask_filled] = cls

        print(f'Check equal {np.array_equal(volume, filled_volume)}')
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

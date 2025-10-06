import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)
from skimage import morphology

def postprocess(model, img, config):

    model.eval()
    pred_logits = model(img)
    pred_probs = F.softmax(1 * pred_logits, dim=1)

    if config.get('probability_smoothing'):
        pred_probs = probability_smooth(pred_probs.cpu().numpy(), sigma=config['sigma'])
        pred_probs = torch.tensor(pred_probs).to(img.device)

    pred_one_hot = probs2one_hot(pred_probs)

    if config.get('morphology'):
        pred_one_hot = apply_morphology(pred_one_hot, operation=config['operation'], radius=config['radius'])

        conflict_mask = (pred_one_hot.sum(axis=0) > 1)
        pred_one_hot = torch.argmax(pred_one_hot, axis=0)
        conflict_idx = torch.where(conflict_mask)
        pred_one_hot[conflict_idx] = torch.argmax(probs[:conflict_idx[0], conflict_idx[1], conflict_idx[2]])

    pred_class = torch.argmax(pred_one_hot, dim=0)

    return pred_class, pred_one_hot

def probability_smooth(prob_map, sigma=1):

    return np.stack([gaussian_filter(prob_map[c], sigma=sigma) for c in range(prob_map.shape[0])], axis=0)

def apply_morphology(segmentation_mask, operation, radius):

    mask = segmentation_mask.cpu().numpy()
    processed = np.zeros_like(mask)
    selem = morphology.ball(radius)
    for c in range(num_classes):
        mask = one_hot[c].astype(bool)
        if 'opening' in operation:
            mask = morphology.binary_opening(mask, selem)
        if 'closing' in operation:
            mask = morphology.binary_closing(mask, selem)
        processed[c] = mask

    return torch.tensor(processed).to(segmentation_mask.device)

import torch
import lpips


def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax


def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]


# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)


def init_lpips_model():
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()
    return LPIPS


def compute_lpips_loss(LPIPS, image, gt_image, mask2d, K=2):
    bbox = mask_to_bbox(mask2d)
    cropped_image = crop_using_bbox(image, bbox)
    cropped_gt_image = crop_using_bbox(gt_image, bbox)
    rendering_patches = divide_into_patches(cropped_image[None, ...], K)
    gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
    lpips_loss = LPIPS(rendering_patches.squeeze()*2-1,gt_patches.squeeze()*2-1).mean()
    return lpips_loss


def is_large_mask(mask2d):
    """
    check if the patch sizes are larger than 32x32 (16x16 minimum size for LPIPS loss & K=2)
    """
    if not torch.any(mask2d):
        return False
    x_min, y_min, x_max, y_max = mask_to_bbox(mask2d)
    if x_max - x_min < 32 or y_max - y_min < 32:
        return False
    return True
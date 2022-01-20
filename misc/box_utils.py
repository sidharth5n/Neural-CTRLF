import torch
import torchvision

def apply_box_transform(boxes, trans):
    """
    Parameters
    ----------
    boxes : torch.tensor of shape (N, M, 4) or (N, 4)
    trans : torch.tensor of shape (N, M, 4) or (N, 4)

    Returns
    -------
    out : torch.tensor of shape (N, M, 4) or (N, 4)
    """

    assert boxes.shape[-1] == 4, 'Last dim of boxes must be 4'
    assert trans.shape[-1] == 4, 'Last dim of trans must be 4'

    xa, ya, wa, ha = torch.split(boxes, 1, -1)
    tx, ty, tw, th = torch.split(trans, 1, -1)

    x = tx * wa + xa
    y = ty * ha + ya
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha

    out = torch.cat([x, y, w, h], dim = -1)
    
    return out

def clip_boxes(boxes, bounds, format):
    """
    Clip bounding boxes to a specified region.
    
    Parameters
    ----------
    boxes         : torch.Tensor of shape (N, 4) or (N, M, 4)
                    Coordinates in the specified format
    bounds        : dict
                    Bounds of the image (inclusive)
                    x_min, x_max, y_min, y_max
    format        : str
                    Format of the boxes. One of 'xyxy', 'cxcywh' or 'xywh'.
    
    Returns
    -------
    boxes_clipped : torch.tensor of shape (N, 4) or (N, M, 4)
                    Clipped boxes in the input format.
    valid         : torch.tensor of shape (N, ) or (N, M)
                    Whether box is geometrically valid
    """
    if format == 'x1y1x2y2':
        boxes_clipped = boxes.clone()
    elif format in ['xcycwh', 'xywh']:
        boxes_clipped = torchvision.ops.box_convert(boxes, format, 'xyxy')
    else:
        raise Exception(f'Unrecognized box format {format}')

    # Now we can actually clip the boxes
    boxes_clipped[..., 0].clamp_(bounds['x_min'], bounds['x_max'] - 1)
    boxes_clipped[..., 1].clamp_(bounds['y_min'], bounds['y_max'] - 1)
    boxes_clipped[..., 2].clamp_(bounds['x_min'] + 1, bounds['x_max'])
    boxes_clipped[..., 3].clamp_(bounds['y_min'] + 1, bounds['y_max'])

    validx = torch.gt(boxes_clipped[..., 2], boxes_clipped[..., 0])
    validy = torch.gt(boxes_clipped[..., 3], boxes_clipped[..., 1])
    valid = validx * validy

    # Convert to the same format as the input
    boxes_clipped = torchvision.ops.box_convert(boxes_clipped, 'xyxy', format)
    
    return boxes_clipped, valid

def invert_box_transform(anchor_boxes, target_boxes):
    """
    # -- Used to compute box regression targets from GT boxes
    """
    xa = anchor_boxes[:, 0]
    ya = anchor_boxes[:, 1]
    wa = anchor_boxes[:, 2]
    ha = anchor_boxes[:, 3]

    xt = target_boxes[:, 0]
    yt = target_boxes[:, 1]
    wt = target_boxes[:, 2]
    ht = target_boxes[:, 3]

    transform = torch.zeros_like(target_boxes)
    transform[:, 0] = (xt - 1 + xa) / wa
    transform[:, 1] = (yt - 1 + ya) / ha
    transform[:, 2] = torch.log(wt / wa)
    transform[:, 3] = torch.log(ht / wa)

    return transform

def box_to_affine(box, H, W):
    """
    -- box_to_affine converts boxes of shape (B, 4) to affine parameter
    -- matrices of shape (B, 2, 3);
    """
    assert box.ndim == 2, 'Expected 2D input'
    B = box.shape[0]
    assert box.shape[1] == 4, 'Expected input of shape B x 4'

    xc = box[:, 0]
    yc = box[:, 1]
    w = box[:, 2]
    h = box[:, 3]

    affine = box.new_zeros((B, 2, 3))
    affine[:, 0, 0] = h / H
    affine[:, 0, 2] = (2*yc - H - 1) / (H - 1)
    affine[:, 1, 1] = w / W
    affine[:, 1, 2] = (2*xc - W - 1) / (W - 1)

    return affine
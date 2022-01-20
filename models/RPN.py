import itertools
import torch
import torch.nn as nn
import torchvision

from misc import box_utils

class RPN(nn.Module):
    def __init__(self, opt):
        super(RPN, self).__init__()
        self.anchor_generator = AnchorGenerator(opt.anchor_widths, opt.anchor_heights)
        self.head = torchvision.models.detection.rpn.RPNHead(opt.input_dim, len(opt.anchor_widths) * len(opt.anchor_heights))
        
    def forward(self, img, feats):
        """
        Parameters
        ----------
        img        : torch.tensor of shape (B, C, H, W)
                     Input image
        feats      : torch.tensor of shape (B, C', H', W')
                     Feature map extracted from the image

        Returns
        -------
        boxes      : torch.tensor of shape (B, K*H*W, 4)
                     
        anchors    : torch.tensor of shape (B, K*H*W, 4)
        transforms : torch.tensor of shape (B, K*H*W, 4)
        scores     : torch.tensor of shape (B, K*H*W, 1)
        """
        scores, boxes = self.head(feats) #(N,K,H,W), (N,4*K,H,W)
        boxes = permute_and_flatten(boxes, 4) # (N,K*H*W,4)
        scores = permute_and_flatten(scores, 1) #(N,K*H*W,1)
        anchors = self.anchor_generator(img, feats) # (N,K*H*W,4) #(N,4*K,H,W)
        # anchors = permute_and_flatten(anchors, 4) # (N,K*H*W,4)
        transforms = box_utils.apply_box_transform(anchors, boxes) # (N,K*H*W,4)
        return boxes, anchors, transforms, scores

class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and image sizes.
    The module support computing anchors at multiple widths and heights per 
    feature map.
    
    widths and heights can have an arbitrary number of elements and 
    AnchorGenerator will output a set of len(widths) * len(heights) anchors
    per spatial location for feature map.

    Parameters
    ----------
    widths  : list or tuple
              Widths of the anchor boxes
    heights : list or tuple
              Heights of the anchor boxes
    """

    def __init__(self, widths = (30, 90, 150, 210, 300), heights = (20, 40, 60)):
        super().__init__()
        self.widths = widths
        self.heights = heights
        self.cell_anchors = self.generate_cells(widths, heights)

    def generate_cells(self, widths, heights, dtype = torch.float32):
        """
        For every (width, height) combination, output a zero-centered anchor with those values.

        Returns
        ----------
        base_anchors : torch.tensor of shape (M*N, 4)
        """
        combinations = list(itertools.product(widths, heights))
        base = torch.as_tensor(combinations, dtype = dtype).split(1, dim = 1)
        return base

    def grid_anchors(self, grid_sizes, strides):
        """
        Generates anchors over the grid.

        Parameters
        ----------
        grid_sizes : list or tuple of length 2
                     Size of feature map (H,W)
        stride     : list or tuple of length 2
                     Stride in the image space (stride_y, stride_x)

        Returns
        -------
        anchors    : torch.tensor of shape (HWK, 4)
                     Anchor coorindates in (xc, yc, w, h) format
        """
        anchors = []
        cell_anchors = self.cell_anchors

        grid_height, grid_width = grid_sizes
        stride_height, stride_width = strides
        device = cell_anchors.device

        # For output anchor, compute [x_center, y_center]
        shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
        shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1) #(HW)
        shift_y = shift_y.reshape(-1) #(HW)
        shifts = torch.stack((shift_x, shift_y), dim=1).unsqueeze(1) #(HW,1,2)
        shifts = shifts.expand(-1, cell_anchors.shape[0], -1)
        cell_anchors = cell_anchors.unsqueeze(0).expand(shifts.shape[0], -1, -1)
        anchors = torch.cat([shifts, cell_anchors], dim = 2).reshape(-1, 4)
        return anchors

    def forward(self, image, feature_map):
        """
        Current implementation works for a batch size (B) of 1 only.

        Parameters
        ----------
        image       : torch.tensor of shape (B, C, H, W)
                      Input image
        feature_map : torch.tensor of shape (B, C', H', W')
                      Feature map extracted from the image
        
        Returns
        -------
        anchors     : torch.tensor of shape (B, K*H*W, 4)
                      Anchors various positions of the image
        """
        assert image.shape[0] == 1 and feature_map.shape[0] == 1, "AnchorGenerator requires batch size of 1"
        grid_size = feature_map.shape[-2:]
        image_size = image.shape[-2:]
        device = feature_map.device
        stride = [torch.tensor(image_size[0] // grid_size[0], dtype = torch.int64, device = device), 
                  torch.tensor(image_size[1] // grid_size[1], dtype = torch.int64, device = device)]
        self.anchors = self.anchors.to(feature_map)
        anchors = self.grid_anchors(grid_size, stride)
        return anchors.unsqueeze(0)

def permute_and_flatten(x, d):
    """
    Input a tensor of shape N x (D * k) x H x W
    Reshape and permute to output a tensor of shape N x (k * H * W) x D 

    Parameters
    ----------
    x : torch.tensor of shape (N, D*K, H, W)
    d : int
        Feature dimension

    Returns
    -------
    x : torch.tensor of shape (N, K*H*W, D)
    """
    n, _, h, w = x.shape
    x = x.reshape((n, -1, d, h, w))
    x = x.permute((0, 3, 4, 1, 2))
    x = x.reshape(n, -1, d)
    return x
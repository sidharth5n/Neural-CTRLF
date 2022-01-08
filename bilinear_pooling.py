import torch
import torch.nn as nn
from torch.nn import functional as F

import box_utils


##############################################################
# CHECK WHETHER THIS CAN BE REPLACED WITH roi_pool FROM PYTORCH
# https://pytorch.org/vision/stable/ops.html
###############################################################
class BilinearRoiPooling(nn.Module):
    """
    BilinearRoiPooling is a layer that uses bilinear sampling to pool featurs for a
    region of interest (RoI) into a fixed size.
    The constructor takes inputs height and width, both integers giving the size to
    which RoI features should be pooled. For example if RoI feature maps are being
    fed to VGG-16 fully connected layers, then we should have height = width = 7.
    WARNING: The bounding box coordinates given in the forward pass should be in
    the coordinate system of the input image used to compute the feature map, NOT
    in the coordinate system of the feature map. To properly compute the forward
    pass, the module needs to know the size of the input image; therefore the method
    setImageSize(image_height, image_width) must be called before each forward pass.
    """

    def __init__(self, height, width):
        super(BilinearRoiPooling, self).__init__()
        self.height = height
        self.width = width
        # -- Grid generator converts matrices to sampling grids of shape
        # -- (B, height, width, 2).
        # self.grid_generator = nn.AffineGridGeneratorBHWD(height, width)
        # self.batch_bilinear_sampler = nn.BatchBilinearSamplerBHWD()

    def forward(self, feats, boxes, img_height, img_width):
        """
        Parameters
        ----------
        feats       : torch.tensor of shape (C, H, W)
                      Convolutional feature map
        boxes       : torch.tensor of shape (B, 4)
                      Bounding box coordinates in (xc, yc, w, h) format. The bounding 
                      box coordinates should be in the coordinate system of the original 
                      image, NOT the convolutional feature map.
        img_height  : int
                      Height of image
        img_width   : int
                      Width of image
        
        Returns
        -------
        roi_feature : torch.tensor of shape (B, C, HH, WW)
                      Pooled features for the region of interest where HH and WW are
                      self.height and self.width
        """
        affine = box_utils.box_to_affine(boxes, img_height, img_width) # (B,2,3)
        # grid = self.grid_generator(affine)
        shape = (affine.shape[0], feats.shape[0], self.height, self.width)
        grid = F.affine_grid(affine, shape) # (B,HH,WW,2)
        feats = feats.transpose(0, 1).transpose(1, 2) # (H,W,C)
        # out = self.batch_bilinear_sampler(feats, grid)
        # roi_feature = out.transpose(2, 3).transpose(1, 2)
        roi_feature = F.grid_sample(feats, grid) # (B,C,HH,WW)
        return roi_feature

###########################################
# NOT REQUIRED, PYTORCH HAS THIS IMPLEMENTED
############################################
class AffineGridGeneratorBHWD(nn.Module):
    """
    #    AffineGridGeneratorBHWD(height, width) :
#    AffineGridGeneratorBHWD:updateOutput(transformMatrix)
#    AffineGridGeneratorBHWD:updateGradInput(transformMatrix, gradGrids)
#    AffineGridGeneratorBHWD will take 2x3 an affine image transform matrix (homogeneous 
#    coordinates) as input, and output a grid, in normalized coordinates* that, once used
#    with the Bilinear Sampler, will result in an affine transform.
#    AffineGridGenerator 
#    - takes (B,2,3)-shaped transform matrices as input (B=batch).
#    - outputs a grid in BHWD layout, that can be used directly with BilinearSamplerBHWD
#    - initialization of the previous layer should biased towards the identity transform :
#       | 1  0  0 |
#       | 0  1  0 |
#    *: normalized coordinates [-1,1] correspond to the boundaries of the input image. 
# ]]
    """
    def __init__(self, height, width):
        super(AffineGridGeneratorBHWD, self).__init__()
        assert height > 1 and width > 1, "Height and width should be > 1"
        self.height = height
        self.width = width
        
        self.baseGrid = torch.Tensor(height, width, 3)
        for i in range(self.height):
            self.baseGrid[i, :, 0] = -1 + i / (self.height - 1) * 2
        for j in range(self.width):
            self.baseGrid[:, j, 1] = -1 + j / (self.width - 1) * 2
        self.baseGrid[:, :, 3] = 1

        self.baseGrid = self.baseGrid.view(-1, 3)

    def forward(self, transformMatrix):
        """
        Parameters
        ----------
        transformMatrix : torch.tensor of shape (B, 2, 3) or (2, 3)

        Returns
        -------
        affine_grid : torch.tensor of shape (B, H*W, 2)
        """
        affine_grid = torch.matmul(self.baseGrid, transformMatrix.transpose(1,2))
        return affine_grid
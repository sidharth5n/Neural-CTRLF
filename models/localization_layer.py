import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.box_sampler import BoxSampler
from models.RPN import RPN
from misc import box_utils

"""
[[


Before each forward pass, you need to call the setImageSize method to set the size
of the underlying image for that forward pass. During training, you also need to call
the setGroundTruth method to set the ground-truth boxes and sequnces:
- gt_boxes: 1 x B1 x 4 array of ground-truth region boxes
- gt_embeddings: 1 x B1 x L array of ground-truth labels for regions
After each forward pass, the instance variable stats will be populated with useful
information; in particular stats.losses has all the losses.
If you set the instance variable timing to true, then stats.times will contain
times for all forward and backward passes.
--]]"""

class LocalizationLayer(nn.Module):
    """
    A LocalizationLayer wraps up all of the complexities of detection regions and
    using a spatial transformer to attend to their features. Used on its own, it can
    be used for learnable region proposals; it can also be plugged into larger modules
    to do region proposal + classification (detection) or region proposal + captioning
    (dense captioning).
    """
    def __init__(self, opt):
        super(LocalizationLayer, self).__init__()
        # Computes region proposals from conv features
        self.rpn = RPN(opt)
        # Performs positive / negative sampling of region proposals
        self.box_sampler = BoxSampler(opt)
        # Whether to ignore out-of-bounds boxes for sampling at training time
        self.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
        # Used to track image size; must call setImageSize before each forward pass
        self.image_width = None
        self.image_height = None
        self.test_clip_boxes = opt.clip_final_boxes
        self.test_nms_thresh = opt.rpn_nms_thresh
        self.test_max_proposals = opt.num_proposals

    def forward(self, *args, **kwargs):
        if self.train:
            return self._train(*args, **kwargs)
        else:
            return self._test(*args, **kwargs)

    def _test(self, cnn_features, img_height, img_width, **kwargs):
        """
        Parameters
        ----------
        cnn_features  : torch.tensor of shape (B, C, H, W)
                        Features extracted from CNN
        img_height    : int
                        Height of the image
        img_width     : int
                        Width of the image
        
        Returns
        -------
        roi_features  : torch.tensor of shape (M, C, HH, WW)
                        RoI pooled features
        rpn_boxes_nms : torch.tensor of shape (M, 4)
                        Bounding box of RoIs in (xc, yc, w, h) format
        """
        # Compute region proposals
        rpn_boxes, _, _, rpn_scores = self.rpn(cnn_features) #(B,K*H*W,4),_,_,(B,K*H*W,1)
        # Maybe clip boxes to image boundary
        if self.test_clip_boxes:
            bounds = {'xmin' : 0, 'ymin' : 0, 'xmax' : img_width - 1, 'ymax' : img_height - 1}
            rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'cxcywh')
            # Clamp parallel arrays only to valid boxes (not oob of the image)
            def clamp_data(data):
                # data should be 1 x kHW x D
                # valid is byte of shape kHW
                assert data.ndim() == 3
                return data[valid]
            
            rpn_boxes = clamp_data(rpn_boxes) # (V,4)
            rpn_scores = clamp_data(rpn_scores) #(V,1)
        
        # Convert rpn boxes from (xc, yc, w, h) format to (x1, y1, x2, y2)
        rpn_boxes_x1y1x2y2 = torchvision.ops.box_convert(rpn_boxes, 'cxcywh', 'xyxy')
        verbose = False
        if verbose:
            print(f'Applying NMS on {rpn_boxes.shape[1]} RPN boxes with {self.test_nms_thresh} threshold.')
        # Perform NMS on the RPN boxes
        idx = torchvision.ops.nms(rpn_boxes_x1y1x2y2, F.sigmoid(rpn_scores.squeeze(1)), self.test_nms_thresh)
        # Clip no. of proposals from NMS
        if kwargs.get('num_proposals', self.test_max_proposals) > 0:
            idx = idx[:kwargs.get('num_proposals', self.test_max_proposals)]
        # Keep only the NMS resultant boxes
        rpn_boxes_nms = rpn_boxes[idx] #(M,4)
        if verbose:
            print(f'After NMS there are {rpn_boxes_nms.shape[0]} boxes')

        return rpn_boxes_nms

    def _train(self, cnn_features, gt_boxes, gt_embeddings, gt_labels, img_height, img_width):
        """
        Parameters
        ----------
        cnn_features      : torch.tensor of shape (B, C, H, W)
                            Features extracted from CNN
        img_height        : int
                            Height of the image
        img_width         : int
                            Width of the image
        gt_boxes          : torch.tensor of shape (B, P, 4)
                            Ground truth bounding boxes in (xc, yc, w, h) format
        gt_embeddings     : torch.tensor of shape (B, P, E)
                            Ground truth embeddings of the labels in each box
        gt_labels         : torch.tensor of shape (B, P)
                            Ground truth labels of the boxes

        Returns
        -------
        roi_features      : torch.tensor of shape (P'+N', C, HH, WW)
                            RoI pooled features
        roi_boxes         : torch.tensor of shape (P'+N', 4)
                            Bounding box of RoIs in (xc, yc, w, h) format
        pos_target_boxes  : torch.tensor of shape (P', 4)
                            Bounding box of targets in (xc, yc, w, h) format
        pos_target_labels : torch.tensor of shape (P', E)
                            Embedding of each target
        label_injection   : torch.tensor of shape (P', )
                            Each element is {-1, 1}. If 1, mismatching pair has been injected in target
        pos_scores        : torch.tensor of shape (P', 2)
                            Box or not box score for each positive box
        neg_scores        : torch.tensor of shape (N', 2)
                            Box or not box score for each negative boxes
        pos_trans         : torch.tensor of shape (P', 4)

        pos_trans_targets : torch.tensor of shape (P', 4)
                            Transformation from anchor box to target box
        """
        B = cnn_features.shape[0]
        assert B == 1, 'Only minibatches with N = 1 are supported'

        # Compute region proposals
        rpn_boxes, rpn_anchors, rpn_trans, rpn_scores = self.rpn(cnn_features) #(B,K*H*W,4),(B,K*H*W,4),(B,K*H*W,4),(B,K*H*W,1)
        # If boxes outside image bounds are to be removed
        if self.train_remove_outbounds_boxes:
            bounds = {'x_min' : 0, 'y_min' : 0, 'x_max' : img_width - 1, 'y_max' : img_height - 1}    
        else:
            bounds = None
        # Sample positive and negative boxes and inject mismatching positive target
        pos_data, pos_target_data, neg_data, label_injection = self.box_sampler([rpn_boxes, rpn_anchors, rpn_trans, rpn_scores], [gt_boxes, gt_embeddings], gt_labels, bounds)
        # Unpack positive data
        pos_boxes, pos_anchors, pos_trans, pos_scores = pos_data #(P',4),(P',4),(P',4),(P',1)
        # Unpack target data
        pos_target_boxes, pos_target_labels = pos_target_data #(P',4),(P',E)
        # Unpack negative data (only scores matter)
        neg_boxes, _, _, neg_scores = neg_data #(N',4),(N',4),(N',4),(N',1)
        # Concatentate pos_boxes and neg_boxes into roi_boxes
        roi_boxes = torch.cat([pos_boxes, neg_boxes], dim = 0) #(P'+N',4)
        # Compute targets for RPN bounding box regression
        ####################################################################
        # SHOULD ALL THE BELOW STATEMENTS BE IN with.torch.no_grad() ???
        #####################################################################
        pos_trans_targets = box_utils.invert_box_transform(pos_anchors, pos_target_boxes)
        # # -- DIRTY DIRTY HACK: To prevent the loss from blowing up, replace boxes
        # # -- with huge pos_trans_targets with ground-truth
        # max_trans = torch.abs(pos_trans_targets).max(1)[0]
        # max_trans_mask = torch.gt(max_trans, 10).expand(pos_trans_targets.shape)
        # mask_sum = max_trans_mask.sum() / 4
        # if mask_sum > 0:
        #     print(f'WARNING: Masking out {mask_sum} boxes in LocalizationLayer')
        #     pos_trans[max_trans_mask] = 0
        #     pos_trans_targets[max_trans_mask] = 0

        return roi_boxes, pos_target_boxes, pos_target_labels, label_injection, pos_scores, neg_scores, pos_trans, pos_trans_targets



######################################################################
# CHECK WHETHER THIS CAN BE REPLACED WITH AnchorGenerator FROM PYTORCH
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py
######################################################################
# class MakeAnchors(nn.Module):
#     def __init__(self, x0, y0, sx, sy, anchors):
#         super().__init__()
#         self.x0 = x0
#         self.y0 = y0
#         self.sx = sx
#         self.sy = sy
#         self.anchors = anchors
    
#     def forward(self, x):
#         """
#         x : torch.tensor of shape (N, K*D, H, W)

#         Returns
#         -------
#         anchors : torch.tensor of shape (N, 4*K, H, W)
#         """
#         n, _, h, w = x.shape
#         k = self.anchors.shape[1]
        
#         x_c = torch.arange(w, dtype = x.dtype, device = x.device)
#         x_c = x_c * self.sx + self.x0
        
#         y_c = torch.arange(h, dtype = x.dtype, device = x.device)
#         y_c = y_c * self.sy + self.y0

#         x_c = x_c.view(1, 1, 1, w).expand(n, k, h, w)
#         y_c = y_c.view(1, 1, h, 1).expand(n, k, h, w)
#         W = self.anchors[0].view(1, k, 1, 1).expand(n, k, h, w)
#         H = self.anchors[1].view(1, k, 1, 1).expand(n, k, h, w)
        
#         anchors = torch.cat([x_c, y_c, W, H], dim = 1)
        
#         return anchors
    

import torch
import torch.nn as nn
import torchvision

import box_utils

require 'torch'
require 'nn'

require 'ctrlfnet.modules.OurCrossEntropyCriterion'

require 'ctrlfnet.modules.RegularizeLayer'

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
        self.box_sampler_helper = BoxSamplerHelper(batch_size = opt.sampler_batch_size,
                                                   low_thresh = opt.sampler_low_thresh,
                                                   high_thresh = opt.sampler_high_thresh,
                                                   vocab_size = opt.vocab_size,
                                                   biased_sampling = opt.biased_sampling)
        # -- Interpolates conv features for each RoI
        self.roi_pooling = BilinearRoiPooling(opt.output_height, opt.output_width)
        # Whether to ignore out-of-bounds boxes for sampling at training time
        self.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
        # Used to track image size; must call setImageSize before each forward pass
        self.image_width = None
        self.image_height = None
        self.test_clip_boxes = opt.clip_final_boxes
        self.test_nms_thresh = opt.rpn_nms_thresh
        self.test_max_proposals = opt.num_proposals

    # -- This needs to be called before each forward pass
    def setImageSize(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

# --[[
# This needs to be called before every training-time forward pass.
# Inputs:
# - gt_boxes: 1 x B1 x 4 array of ground-truth region boxes
# - gt_labels: B1 dimensional array of ground-truth labels for regions
# - gt_embeddings: 1 x B1 x E array of ground-truth embeddings for regions
# --]]
    def setGroundTruth(self, gt_boxes, gt_embeddings, gt_labels):
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.gt_embeddings = gt_embeddings
        self._called_forward_gt = False
        self._called_backward_gt = False

    def forward(self, cnn_features, img_height, img_width):
        """
        Parameters
        ----------
        cnn_features     : 1 x C x H x W array of CNN features
        
        Returns
        -------
        roi_features     : (pos + neg) x D x HH x WW array of features for RoIs;
                           roi_features[{{1, pos}}] gives the features for the positive RoIs
                           and the rest are negatives.
        roi_boxes        : (pos + neg) x 4 array of RoI box coordinates (xc, yc, w, h);
                           roi_boxes[{{1, pos}}] gives the coordinates for the positive boxes
                           and the rest are negatives.
        gt_boxes_sample  : pos x 4 array of ground-truth region boxes corresponding to
                           sampled positives. This will be an empty Tensor at test-time.
        gt_labels_sample : pos x L array of ground-truth labels corresponding to sampled
                           positives. This will be an empty Tensor at test-time.
        """
        self.setImageSize(self, img_height, img_width)
        if self.train:
            return self._train(cnn_features)
        else:
            return self._test(cnn_features)

    def _test(self, cnn_features):
        
        rpn_boxes, rpn_anchors, rpn_trans, rpn_scores = self.rpn(cnn_features)
        num_boxes = rpn_boxes.shape[1]

        # Maybe clip boxes to image boundary
        if self.test_clip_boxes:
            bounds = {'xmin' : 1, 'ymin' : 1, 'xmax' : self.image_width, 'ymax' : self.image_height}
            rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'cxcywh')

            # Clamp parallel arrays only to valid boxes (not oob of the image)
            def clamp_data(data):
                # data should be 1 x kHW x D
                # valid is byte of shape kHW
                assert data.ndim() == 3
                return data[valid].unsqueeze(0)
            
            rpn_boxes = clamp_data(rpn_boxes)
            rpn_anchors = clamp_data(rpn_anchors)
            rpn_trans = clamp_data(rpn_trans)
            rpn_scores = clamp_data(rpn_scores)
            num_boxes = rpn_boxes.shape[1]
        
        # Convert rpn boxes from (xc, yc, w, h) format to (x1, y1, x2, y2)
        rpn_boxes_x1y1x2y2 = torchvision.ops.box_convert(rpn_boxes, 'cxcywh', 'xyxy')

        # Convert objectness positive / negative scores to probabilities
        rpn_scores_exp = torch.exp(rpn_scores)
        pos_exp = rpn_scores_exp[0, :, 0]
        neg_exp = rpn_scores_exp[0, :, 1]
        scores = pos_exp / (pos_exp + neg_exp)
        
        # Use NMS indices to pull out corresponding data from RPN
        verbose = False
        if verbose:
            print(f'Before RPN NMS there are {num_boxes} boxes.')
            print(f'Using NMS threshold {self.test_nms_thresh}.')
        idx = torchvision.ops.nms(rpn_boxes_x1y1x2y2, scores, self.test_nms_thresh)
        if self.test_max_proposals > 0:
            idx = idx[:self.test_max_proposals]

        # All these are being converted from (1, B2, D) to (B3, D)
        # where B2 are the number of boxes after boundary clipping and B3
        # is the number of boxes after NMS
        rpn_boxes_nms = rpn_boxes[:, idx].squeeze(0)
        rpn_anchors_nms = rpn_anchors[:, idx].squeeze(0)
        rpn_trans_nms = rpn_trans[:, idx].squeeze(0)
        rpn_scores_nms = rpn_scores[:, idx].squeeze(0)
        scores_nms = scores[idx]

        if verbose:
            print(f'After NMS there are {rpn_boxes_nms.shape[0]} boxes')

        ##############################################################
        # CHECK WHETHER THIS CAN BE REPLACED WITH roi_pool FROM PYTORCH
        # https://pytorch.org/vision/stable/ops.html
        ###############################################################
        # Use roi pooling to get features for boxes
        roi_features = self.roi_pooling(cnn_features[0], rpn_boxes_nms, self.image_height, self.image_width)

        return roi_features, rpn_boxes_nms, None, None

    def _train(self, cnn_features):
        """
        Returns
        -------
        roi_features, roi_boxes, pos_target_boxes, pos_target_labels
        """
        assert(self.gt_boxes and self.gt_embeddings and not self._called_forward_gt,
              'Must call setGroundTruth before training-time forward pass')
        gt_boxes, gt_embeddings = self.gt_boxes, self.gt_embeddings
        self._called_forward_gt = True

        N = cnn_features.shape[0]
        assert N == 1, 'Only minibatches with N = 1 are supported'
        B1 = gt_boxes.shape[1]
        assert gt_boxes.ndim() == 3 and gt_boxes.shape[0] == N and gt_boxes.shape[2] == 4, 'gt_boxes must have shape (N, B1, 4)'
        assert gt_embeddings.ndim() == 3 and gt_embeddings.shape[0] == N and gt_embeddings.shape[1] == B1, 'gt_embeddings must have shape (N, B1, L)'

        # Run the RPN forward
        rpn_boxes, rpn_anchors, rpn_trans, rpn_scores = self.rpn(cnn_features)

        if self.opt.train_remove_outbounds_boxes:
            image_height, image_width = None, None
            bounds = {'x_min' : 1, 'y_min' : 1, 'x_max' : self.image_width, 'y_max' : self.image_height}
            self.box_sampler_helper.setBounds(bounds)

        # Run the sampler forward
        pos_data, pos_target_data, neg_data, self.y = self.box_sampler_helper([rpn_boxes, rpn_anchors, rpn_trans, rpn_scores], [gt_boxes, gt_embeddings], self.gt_labels)
        # Unpack pos data
        pos_boxes, pos_anchors, pos_trans, pos_scores = pos_data
        # Unpack target data
        pos_target_boxes, pos_target_labels = pos_target_data
        # Unpack neg data (only scores matter)
        neg_boxes, _, _, neg_scores = neg_data
        # Concatentate pos_boxes and neg_boxes into roi_boxes
        roi_boxes = torch.cat([pos_boxes, neg_boxes], dim = 0)
        ##############################################################
        # CHECK WHETHER THIS CAN BE REPLACED WITH roi_pool FROM PYTORCH
        # https://pytorch.org/vision/stable/ops.html
        ###############################################################
        # Run the RoI pooling forward for positive boxes
        roi_features = self.roi_pooling(cnn_features[0], roi_boxes, self.image_height, self.image_width)
        # Compute targets for RPN bounding box regression
        pos_trans_targets = box_utils.invert_box_transform(pos_anchors, pos_target_boxes)
        # -- DIRTY DIRTY HACK: To prevent the loss from blowing up, replace boxes
        # -- with huge pos_trans_targets with ground-truth
        max_trans = torch.abs(pos_trans_targets).max(1)[0]
        max_trans_mask = torch.gt(max_trans, 10).expand(pos_trans_targets.shape)
        mask_sum = max_trans_mask.sum() / 4
        if mask_sum > 0:
            print(f'WARNING: Masking out {mask_sum} boxes in LocalizationLayer')
            pos_trans[max_trans_mask] = 0
            pos_trans_targets[max_trans_mask] = 0

        return roi_features, roi_boxes, pos_target_boxes, pos_target_labels


class RPN(nn.Module):
    def __init__(self, opt):
        super(RPN, self).__init__()
        # Set up anchor sizes
        anchors = torch.Tensor([[30, 20], [90, 20], [150, 20], [210, 20], [300, 20],
                                [30, 40], [90, 40], [150, 40], [210, 40], [300, 40],
                                [30, 60], [90, 60], [150, 60], [210, 60], [300, 60]]).t()
        anchors = anchors * opt.anchor_scale
        num_anchors = anchors.shape[1]

        # Add an extra conv layer and a ReLU
        self.conv = nn.Sequential(nn.Conv2d(opt.input_dim, opt.rpn_num_filters, opt.rpn_filter_size, 1, opt.rpn_filter_size // 2),
                                  nn.ReLU())
        # Branch to produce box coordinates for each anchor
        self.bbox_pred = nn.Conv2d(opt.rpn_num_filters, 4*num_anchors, 1, 1)
        ######################################################################
        # CHECK WHETHER THIS CAN BE REPLACED WITH AnchorGenerator FROM PYTORCH
        # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py
        ######################################################################
        x0, y0, sx, sy = opt.field_centers
        self.anchor_generator = MakeAnchors(x0, y0, sx, sy, anchors)   
        # Branch to produce box / not box scores for each anchor
        self.cls_logits = nn.Conv2d(opt.rpn_num_filters, 2*num_anchors, 1, 1)

    def forward(self, feats):
        """
        Parameters
        ----------
        feats      : torch.tensor of shape (N, C, H, W)

        Returns
        -------
        boxes      : torch.tensor of shape (N, K*H*W, 4)
        anchors    : torch.tensor of shape (N, K*H*W, 4)
        transforms : torch.tensor of shape (N, K*H*W, 4)
        scores     : torch.tensor of shape (N, K*H*W, 2)
        """
        t = self.conv(feats) # (N,R,H,W)
        y = self.bbox_pred(t) # (N,4*K,H,W)
        boxes = permute_and_flatten(y, 4) # (N,K*H*W,4)
        anchors = self.anchor_generator(y) # (N,4*K,H,W)
        anchors = permute_and_flatten(anchors, 4) # (N,K*H*W,4)
        transforms = box_utils.apply_box_transform(anchors, boxes) # (N,K*H*W,4)
        scores = self.cls_logits(t) #(N,2*K,H,W)
        scores = permute_and_flatten(scores, 2) #(N,K*H*W,2)
        return boxes, anchors, transforms, scores

######################################################################
# CHECK WHETHER THIS CAN BE REPLACED WITH AnchorGenerator FROM PYTORCH
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py
######################################################################
class MakeAnchors(nn.Module):
    def __init__(self, x0, y0, sx, sy, anchors):
        self.x0 = x0
        self.y0 = y0
        self.sx = sx
        self.sy = sy
        self.anchors = anchors
    
    def forward(self, x):
        """
        x : torch.tensor of shape (N, K*D, H, W)

        Returns
        -------
        anchors : torch.tensor of shape (N, 4*K, H, W)
        """
        with torch.no_grad():
            n, _, h, w = x.shape
        k = self.anchors.shape[1]
        
        x_c = torch.arange(w, dtype = x.dtype, device = x.device)
        x_c = x_c * self.sx + self.x0
        y_c = torch.arange(h, dtype = x.dtype, device = x.device)
        y_c = y_c * self.sy + self.y0

        x_c = x_c.view(1, 1, 1, w).expand(n, k, h, w)
        y_c = x_c.view(1, 1, 1, w).expand(n, k, h, w)
        w = self.anchors[0].view(1, k, 1, 1).expand(n, k, h, w)
        h = self.anchors[1].view(1, k, 1, 1).expand(n, k, h, w)
        anchors = torch.cat([x_c, y_c, w, h], dim = 1)
        
        return anchors
    
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
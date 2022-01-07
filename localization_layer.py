require 'torch'
require 'nn'

require 'ctrlfnet.modules.OurCrossEntropyCriterion'
require 'ctrlfnet.modules.BoxSamplerHelper'
require 'ctrlfnet.modules.RegularizeLayer'
require 'ctrlfnet.modules.MakeAnchors'

local box_utils = require 'ctrlfnet.box_utils'
local utils = require 'ctrlfnet.utils'

"""
[[
A LocalizationLayer wraps up all of the complexities of detection regions and
using a spatial transformer to attend to their features. Used on its own, it can
be used for learnable region proposals; it can also be plugged into larger modules
to do region proposal + classification (detection) or region proposal + captioning\
(dense captioning).
Input:
- cnn_features: 1 x C x H x W array of CNN features
Returns: List of:
- roi_features: (pos + neg) x D x HH x WW array of features for RoIs;
  roi_features[{{1, pos}}] gives the features for the positive RoIs
  and the rest are negatives.
- roi_boxes: (pos + neg) x 4 array of RoI box coordinates (xc, yc, w, h);
  roi_boxes[{{1, pos}}] gives the coordinates for the positive boxes
  and the rest are negatives.
- gt_boxes_sample: pos x 4 array of ground-truth region boxes corresponding to
  sampled positives. This will be an empty Tensor at test-time.
- gt_labels_sample: pos x L array of ground-truth labels corresponding to sampled
  positives. This will be an empty Tensor at test-time.
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


# -- Forward declaration; defined below
local build_rpn


class LocalizationLayer(nn.Module):

    def __init__(self, opt = {}):
        super(LocalizationLayer, self).__init__()

        opt.input_dim = utils.getopt(opt, 'input_dim')
        opt.output_height = utils.getopt(opt, 'output_height')
        opt.output_width = utils.getopt(opt, 'output_width')

        # -- list x0, y0, sx, sy
        opt.field_centers = utils.getopt(opt, 'field_centers')

        opt.backend = utils.getopt(opt, 'backend', 'cudnn')
        opt.rpn_filter_size = utils.getopt(opt, 'rpn_filter_size', 3)
        opt.rpn_num_filters = utils.getopt(opt, 'rpn_num_filters', 256)
        opt.zero_box_conv = utils.getopt(opt, 'zero_box_conv', true)
        opt.std = utils.getopt(opt, 'std', 0.01)
        opt.anchor_scale = utils.getopt(opt, 'anchor_scale', 1.0)

        opt.sampler_batch_size = utils.getopt(opt, 'sampler_batch_size', 256)
        opt.sampler_high_thresh = utils.getopt(opt, 'sampler_high_thresh', 0.5)
        opt.sampler_low_thresh = utils.getopt(opt, 'sampler_low_thresh', 0.25)
        opt.train_remove_outbounds_boxes = utils.getopt(opt, 'train_remove_outbounds_boxes', 1)

        utils.ensureopt(opt, 'mid_box_reg_weight')
        utils.ensureopt(opt, 'mid_objectness_weight')
        
        opt.box_reg_decay = utils.getopt(opt, 'box_reg_decay', 0)
        self.opt = opt
        self.nets = {}

        # -- Computes region proposals from conv features
        self.nets.rpn = build_rpn(opt)

        # -- Performs positive / negative sampling of region proposals
        self.nets.box_sampler_helper = nn.BoxSamplerHelper{
                                          batch_size=opt.sampler_batch_size,
                                          low_thresh=opt.sampler_low_thresh,
                                          high_thresh=opt.sampler_high_thresh,
                                          vocab_size=opt.vocab_size,
                                          biased_sampling=opt.biased_sampling,
                                      } 

        # -- Interpolates conv features for each RoI
        self.nets.roi_pooling = nn.BilinearRoiPooling(opt.output_height, opt.output_width)

        # -- Used to track image size; must call setImageSize before each forward pass
        self.image_width = None
        self.image_height = None

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
        self._called_forward_gt = false
        self._called_backward_gt = false

    def setTestArgs(args):
        args = args or {}
        self.test_clip_boxes = utils.getopt(args, 'clip_boxes', true)
        self.test_nms_thresh = utils.getopt(args, 'nms_thresh', 0.7)
        self.test_max_proposals = utils.getopt(args, 'max_proposals', 300)

    def forward(cnn_features):
      if self.train:
          return self:_forward_train(cnn_features)
      else:
          return self:_forward_test(cnn_features)

# -- Sets external region proposals to use for for training/testing
# -- input is a N x 4 list of N region proposals
    def set_region_proposals(input):
        self.region_proposals = input

    def _forward_test(cnn_features):
        local arg = {
          clip_boxes = self.test_clip_boxes,
          nms_thresh = self.test_nms_thresh,
          max_proposals = self.test_max_proposals
        }

        # -- Make sure that setImageSize has been called
        assert(self.image_height and self.image_width and not self._called_forward_size,
              'Must call setImageSize before each forward pass')
        self._called_forward_size = true

        rpn_boxes, rpn_anchors, rpn_trans, rpn_boxes = self.nets.rpn(cnn_features)
        num_boxes = rpn_boxes.shape[1]

        # -- Maybe clip boxes to image boundary
        if arg.clip_boxes:
            bounds = [1, 1, self.image_width, self.image_height]
            rpn_boxes, valid = box_utils.clip_boxes(rpn_boxes, bounds, 'xcycwh')

            # -- Clamp parallel arrays only to valid boxes (not oob of the image)
            def clamp_data(data)
                # -- data should be 1 x kHW x D
                # -- valid is byte of shape kHW
                assert(data:size(1) == 1, 'must have 1 image per batch')
                assert(data:dim() == 3)
                local mask = valid:view(1, -1, 1):expandAs(data)
                return data[mask]:view(1, -1, data:size(3))
            end
            rpn_boxes = clamp_data(rpn_boxes)
            rpn_anchors = clamp_data(rpn_anchors)
            rpn_trans = clamp_data(rpn_trans)
            rpn_scores = clamp_data(rpn_scores)
            num_boxes = rpn_boxes:size(2)
        end
        
        # -- Convert rpn boxes from (xc, yc, w, h) format to (x1, y1, x2, y2)
        rpn_boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(rpn_boxes)

        # -- Convert objectness positive / negative scores to probabilities
        rpn_scores_exp = torch.exp(rpn_scores)
        pos_exp = rpn_scores_exp[0, :, 0]
        neg_exp = rpn_scores_exp[0, :, 1]
        scores = pos_exp / (pos_exp + neg_exp)
        
        verbose = False
        if verbose:
            print('in LocalizationLayer forward_test')
            print(string.format('Before NMS there are %d boxes', num_boxes))
            print(string.format('Using NMS threshold %f', arg.nms_thresh))
        end

        # -- Run NMS and sort by objectness score
        boxes_scores = scores.new(num_boxes, 5)
        boxes_scores[:, :4] = rpn_boxes_x1y1x2y2
        boxes_scores[:, 4] = scores
        
        
        if arg.max_proposals == -1:
            idx = box_utils.nms(boxes_scores, arg.nms_thresh)
        else:
            idx = box_utils.nms(boxes_scores, arg.nms_thresh, arg.max_proposals)        

        # -- Use NMS indices to pull out corresponding data from RPN
        # -- All these are being converted from (1, B2, D) to (B3, D)
        # -- where B2 are the number of boxes after boundary clipping and B3
        # -- is the number of boxes after NMS
        local rpn_boxes_nms = rpn_boxes:index(2, idx)[1]
        local rpn_anchors_nms = rpn_anchors:index(2, idx)[1]
        local rpn_trans_nms = rpn_trans:index(2, idx)[1]
        local rpn_scores_nms = rpn_scores:index(2, idx)[1]
        local scores_nms = scores:index(1, idx)

        if verbose:
            print(string.format('After NMS there are %d boxes', rpn_boxes_nms:size(1)))
        end

        # -- Use roi pooling to get features for boxes
        self.nets.roi_pooling:setImageSize(self.image_height, self.image_width)
        roi_features = self.nets.roi_pooling:forward{cnn_features[1], rpn_boxes_nms}
        
        if self.dump_vars:
            local vars = self.stats.vars or {}
            vars.test_rpn_boxes_nms = rpn_boxes_nms
            vars.test_rpn_anchors_nms = rpn_anchors_nms
            vars.test_rpn_scores_nms = scores:index(1, idx)
            self.stats.vars = vars

        return roi_features, rpn_boxes_nms, None, None
        # -- return roi_features, rpn_boxes_nms, scores_nms

    def _forward_train(cnn_features):
        """
        Returns
        -------
        roi_features, roi_boxes, pos_target_boxes, pos_target_labels
        """
        assert(self.gt_boxes and self.gt_embeddings and not self._called_forward_gt,
              'Must call setGroundTruth before training-time forward pass')
        gt_boxes, gt_embeddings = self.gt_boxes, self.gt_embeddings
        self._called_forward_gt = True

        # -- Make sure that setImageSize has been called
        assert(self.image_height and self.image_width and not self._called_forward_size,
              'Must call setImageSize before each forward pass')
        self._called_forward_size = true

        N = cnn_features.shape[0]
        assert N == 1, 'Only minibatches with N = 1 are supported'
        B1 = gt_boxes.shape[1]
        assert gt_boxes.ndim() == 3 and gt_boxes.shape[0] == N and gt_boxes.shape[2] == 4, 'gt_boxes must have shape (N, B1, 4)'
        assert gt_embeddings.ndim() == 3 and gt_embeddings.shape[0] == N and gt_embeddings.shape[1] == B1, 'gt_embeddings must have shape (N, B1, L)'

        # -- Run the RPN forward
        rpn_boxes, rpn_anchors, rpn_trans, rpn_scores = self.nets.rpn(cnn_features)

        if self.opt.train_remove_outbounds_boxes == 1:
            image_height, image_width = None, None
            local bounds = {x_min=1, y_min=1, x_max=self.image_width, y_max=self.image_height
            }
            self.nets.box_sampler_helper:setBounds(bounds)

        # -- Run the sampler forward
        self.nets.box_sampler_helper:set_labels(self.gt_labels)
        pos_data, pos_target_data, neg_data = self.nets.box_sampler_helper(self.rpn_out, {gt_boxes, gt_embeddings}}

        self.y = self.nets.box_sampler_helper.y --TODO:change to return this from the box_sampler_helper directly
      
        # -- Unpack pos data
        pos_boxes, pos_anchors = pos_data[0], pos_data[1]
        pos_trans, pos_scores = pos_data[2], pos_data[3]
        
        # -- Unpack target data
        pos_target_boxes, pos_target_labels = unpack(pos_target_data)

        # -- Unpack neg data (only scores matter)
        neg_boxes = neg_data[0]
        neg_scores = neg_data[3]

        num_pos, num_neg = pos_boxes.shape[0], neg_scores.shape[0]

        # -- Concatentate pos_boxes and neg_boxes into roi_boxes
        roi_boxes = torch.cat([pos_boxes, neg_boxes], dim = 0)

        # -- Run the RoI pooling forward for positive boxes
        self.nets.roi_pooling:setImageSize(self.image_height, self.image_width)
        roi_features = self.nets.roi_pooling(cnn_features[1], roi_boxes)

        # -- Compute targets for RPN bounding box regression
        pos_trans_targets = invert_box_transform(pos_anchors, pos_target_boxes)

        # -- DIRTY DIRTY HACK: To prevent the loss from blowing up, replace boxes
        # -- with huge pos_trans_targets with ground-truth
        max_trans = torch.abs(pos_trans_targets):max(2)
        max_trans_mask = torch.gt(max_trans, 10):expandAs(pos_trans_targets)
        mask_sum = max_trans_mask:sum() / 4
        if mask_sum > 0 then
            print(f'WARNING: Masking out {mask_sum} boxes in LocalizationLayer')
            pos_trans[max_trans_mask] = 0
            pos_trans_targets[max_trans_mask] = 0

        return roi_features, roi_boxes, pos_target_boxes, pos_target_labels

# -- RPN returns {boxes, anchors, transforms, scores}

class RPN(nn.Module):
    def __init__(self, opt):
    #   -- Set up anchor sizes
        if opt.anchors:
            self.anchors = opt.anchors
        else:
            self.anchors = torch.Tensor([[30, 20], [90, 20], [150, 20], [210, 20], [300, 20],
                                         [30, 40], [90, 40], [150, 40], [210, 40], [300, 40],
                                         [30, 60], [90, 60], [150, 60], [210, 60], [300, 60]]).t()
        self.anchors = self.anchors * opt.anchor_scale
        
        num_anchors = anchors.shape[1]

        # -- Add an extra conv layer and a ReLU
        self.rpn = nn.Sequential(nn.Conv2d(opt.input_dim, opt.rpn_num_filters, opt.rpn_filter_size, 1, opt.rpn_filter_size // 2),
                                 nn.ReLU())

        # -- Branch to produce box coordinates for each anchor
        # -- This branch will return {boxes, {anchors, transforms}}
        self.box_branch = nn.Sequential(nn.Conv2d(opt.rpn_num_filters, 4*num_anchors, 1, 1),
                                        nn.RegularizeLayer(opt.box_reg_decay)) # CHECK HOW TO HOW IMPLEMENT IN PYTORCH
        
        
        x0, y0, sx, sy = opt.field_centers
        self.seq = nn.Sequential(nn.MakeAnchors(x0, y0, sx, sy, anchors),
                                 nn.ReshapeBoxFeatures(num_anchors))
        
        self.n2 = nn.ReshapeBoxFeatures(num_anchors)        

        # -- Branch to produce box / not box scores for each anchor
        rpn_branch = nn.Sequential(nn.Conv2d(opt.rpn_num_filters, 2*num_anchors, 1, 1),
                                   nn.ReshapeBoxFeatures(num_anchors))

        # -- Concat and flatten the branches
        local concat = nn.ConcatTable()
        concat:add(box_branch)
        concat:add(rpn_branch)

        rpn:add(concat)
        rpn:add(nn.FlattenTable())

        return rpn, anchors
    end

    def forward(self, x):
        x = self.rpn(x)
        y = self.box_branch(x)
        a = self.seq(y)
        b = self.n2(y)
        a_ = apply_box_transform(x)
        b_ = apply_box_transform(x)
        z = self.rpn_branch(x)
        {{{a_, {b_}}, {a, {b}}}, z}
        {{{a_, {b_}}, {b_, b}}, z}

class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels: int, num_filters: int, filter_size: int, num_anchors: int) -> None:
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, filter_size, 1, filter_size // 2)
        self.bbox_pred = nn.Conv2d(num_filters, 4*num_anchors, 1, 1)
        self.cls_logits = nn.Conv2d(num_filters, 2*num_anchors, 1, 1),

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

class MakeAnchors(nn.Module):
    def __init__(self, x0, y0, sx, sy, anchors):
        self.x0 = x0
        self.y0 = y0
        self.sx = sx
        self.sy = sy
        self.anchors = anchors
    
    def forward(self, x):
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
        
class ReshapeBoxFeatures(nn.Module):
    def __init__(self, k):
        super(ReshapeBoxFeatures, self).__init__()
        self.k = k
    
    def forward(self, x):
        """
        Input a tensor of shape N x (D * k) x H x W
        Reshape and permute to output a tensor of shape N x (k * H * W) x D 
        """
        n, _, h, w = x.shape
        d = x.shape[1] // self.k
        x = x.reshape((n, self.k, h, w, d))
        x = x.permute((0, 1, 3, 4, 2))
        x = x.reshape(n, -1, d)
        return x

class BilinearRoiPooling(nn.Module):

    def __init__(self, height, width):
        super(BilinearRoiPooling, self).__init__()
        self.height = height
        self.width = width
        
        # -- box_branch is a net to convert box coordinates of shape (B, 4)
        # -- to sampling grids of shape (B, height, width)
        # -- Grid generator converts matrices to sampling grids of shape
        # -- (B, height, width, 2).
        self.grid_generator = nn.AffineGridGeneratorBHWD(height, width))
        self.sampler = nn.BatchBilinearSamplerBHWD()

        # -- Set these by calling setImageSize
        self.image_height = None
        self.image_width = None

    def setImageSize(self, image_height, image_width)
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, feats, boxes):
        assert self.image_height and self.image_width, 'Must call setImageSize before each forward pass'
        feats = feats.transpose(0, 1).transpose(1, 2)
        affine = box_to_affine(boxes, self.image_height, self.image_width)
        grids = self.grid_generator(affine)
        roi_features = self.sampler(feats, grids)
        roi_features = roi_features.transpose(2, 3).transpose(1, 2)
        return roi_features


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

    affine = torch.zeros((B, 2, 3), device = box.device)
    affine[:, 0, 0] = h / H
    affine[:, 0, 2] = (2*yc - H - 1) / (H - 1)
    affine[:, 1, 1] = w / W
    affine[:, 1, 2] = (2*xc - W - 1) / (W - 1)

    return affine

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

    transform = target_boxes.zeros_like()
    transform[:, 0] = (xt - 1 + xa) / wa
    transform[:, 1] = (yt - 1 + ya) / ha
    transform[:, 2] = torch.log(wt / wa)
    transform[:, 3] = torch.log(ht / wa)

    return transform
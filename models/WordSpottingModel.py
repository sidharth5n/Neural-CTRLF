from platform import python_branch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.PreActResNet import PreActResNet34
# from models import PreActResNet
from models.localization_layer import LocalizationLayer
from misc import box_utils

class WordSpottingModel(nn.Module):

    def __init__(self, opt):
        super(WordSpottingModel, self).__init__()
        # Threshold for performing NMS on final boxes
        self.final_nms_thresh = opt.final_nms_thresh
        # Break CNN into three parts - feature extractor, recognition base and linear layer
        cnn = torchvision.models.resnet34(pretrained = True) # PreActResNet34()
        cnn = list(cnn.children())
        for i, layer in enumerate(cnn):
            for param in layer.parameters():
                param.requires_grad = False
        # First 6 layers act as the feature extractor
        self.feat_cnn = nn.Sequential(*cnn[:6]) #cnn.model[:6])
        # The final layers of the CNN is the recognition base
        self.recog_base = nn.Sequential(*cnn[6:-1]) #cnn.model[6:])
        # Linear layer of the CNN
        self.recog_linear = nn.Linear(512, opt.descriptor_size) # cnn.linear
        # -- Figure out the receptive fields of the CNN
        # -- TODO: Should we just hardcode this too per CNN? Answer: yes
        # -- TODO: Is this correct? POSSIBLE BUG
        x0, y0 = 1, 1
        sx, sy = 1, 1
        for i in range(4):
            x0 = x0 + sx / 2
            y0 = y0 + sy / 2
            sx = 2 * sx
            sy = 2 * sy
        # 1.5, 1.5, 2, 2
        # 2.5, 2.5, 4, 4
        # 4.5, 4.5, 8, 8
        # 8.5, 8.5, 16, 16

        opt.field_centers = [x0, y0, sx, sy]
        # Network for region proposal, sampling and pooling
        self.localization_layer = LocalizationLayer(opt)
        # Interpolates conv features for each RoI
        self.roi_pooling = torchvision.ops.RoIPool((opt.output_height, opt.output_width), spatial_scale = 1/8)#BilinearRoiPooling(opt.output_height, opt.output_width)
        # Objectness branch for confidence score of final boxes
        self.objectness_branch = nn.Linear(opt.descriptor_size, 1)
        # Regression branch for regressing from RPN boxes to final boxes
        self.box_reg_branch = nn.Linear(opt.descriptor_size, 4)
        # FC Network for finding word embedding
        self.embedding_net = EmbeddingNet(opt)
    
    def recognition(self, feats, roi_boxes, gt_boxes = None, gt_labels = None):
        """
        Parameters
        ----------
        feats             : torch.tensor of shape (B, C, H, W)
                            Features extracted from CNN
                            RoI pooled features
        roi_boxes         : torch.tensor of shape (P'+N', 4)
                            Bounding box of RoIs in (xc, yc, w, h) format
        gt_boxes          : torch.tensor of shape (P', 4), optional
                            Bounding box of targets in (xc, yc, w, h) format. Default is None.
        gt_labels         : torch.tensor of shape (P', E), optional
                            Embedding of each target. Default is None.
        
        Returns
        -------
        objectness_scores : torch.tensor of shape (P'+N', 1)
                            Score for the final boxes to be positive
        pos_roi_boxes     : torch.tensor of shape (P', 4)
                            Bbox coordinates in (xc, yc, w, h) of the positive boxes
        final_box_trans   : torch.tensor of shape (P', 4)
                            Transformation from RoI boxes to final boxes
        final_boxes       : torch.tensor of shape (P', 4) or None
                            Transformation applied on RoI boxes during testing
        emb_output        : torch.tensor of shape (P', E)
                            Embedding for each RoI
        """
        # Use roi pooling to get features for boxes
        roi_feats = self.roi_pooling(feats, [roi_boxes]) #(P'+N', C, HH, WW)    
        roi_codes = self.recog_base(roi_feats).squeeze(3).squeeze(2)#transpose(1, 2)
        roi_codes = self.recog_linear(roi_codes)
        # Get positive or negative probabilities for final boxes
        objectness_scores = self.objectness_branch(roi_codes)
        # Separate positive features and boxes if GT is available
        pos_roi_codes = roi_codes[:gt_labels.shape[0]] if gt_labels is not None else roi_codes
        if pos_roi_codes.shape[0] > 1:
            print("pos_roi_codes", pos_roi_codes.shape)
        pos_roi_boxes = roi_boxes[:gt_boxes.shape[0]] if gt_boxes is not None else roi_boxes
        # Regress final transformation from RoI boxes to final boxes
        final_box_trans = self.box_reg_branch(pos_roi_codes)
        # Apply transformation on positive RoI boxes
        final_boxes = box_utils.apply_box_transform(pos_roi_boxes, final_box_trans) if not self.train else None
        # Compute embedding for each RoI
        emb_output = self.embedding_net(pos_roi_codes)

        return objectness_scores, pos_roi_boxes, final_box_trans, final_boxes, emb_output
    
    def forward(self, *args, **kwargs):
        if self.train:
            return self._train(*args, **kwargs)
        else:
            return self._test(*args, **kwargs)

    def _train(self, img, gt_boxes, gt_embeddings, gt_labels):
        """
        Parameters
        ----------
        img           : torch.tensor of shape (B, 1, H, W)
                        Mean normalized grayscale image
        gt_boxes      : torch.tensor of shape (B, P, 4)
                        Ground truth bounding boxes in (xc, yc, w, h) format
        gt_embeddings : torch.tensor of shape (B, P, E)
                        Ground truth embeddings of the labels in each box
        gt_labels     : torch.tensor of shape (B, P)
                        Ground truth labels of the boxes

        Returns
        -------
        objectness_scores    : torch.tensor of shape (P'+N', 1)
                               Score for the final boxes to be positive
        pos_roi_boxes        : torch.tensor of shape (P', 4)
                               Bbox coordinates in (xc, yc, w, h) of the positive boxes
        final_box_trans      : torch.tensor of shape (P', 4)
                               Transformation from RoI boxes to final boxes
        emb_output           : torch.tensor of shape (P', E)
                               Embedding for each RoI
        label_injection      : torch.tensor of shape (P', )
                               Each element is {-1, 1}. If 1, mismatching pair has been injected in target
        pos_scores           : torch.tensor of shape (P', 1)
                               Box or not box score for each positive box
        neg_scores           : torch.tensor of shape (N', 1)
                               Box or not box score for each negative boxes
        pos_trans            : torch.tensor of shape (P', 4)
                               
        pos_trans_targets    : torch.tensor of shape (P', 4)
                               Transformation from anchor box to target box
        pos_target_boxes     : torch.tensor of shape (P', 4)
                               Bounding box of targets in (xc, yc, w, h) format
        pos_target_embedding : torch.tensor of shape (P', E)
                               Embedding of each target
        """
        assert img.shape[0] == 1, "Batch size should be 1"
        H, W = img.shape[2:]
        feats = self.feat_cnn(img) #(B,C,H/8,W/8)
        roi_boxes, pos_target_boxes, pos_target_labels, label_injection, pos_scores, neg_scores, pos_trans, pos_trans_targets = self.localization_layer(feats, gt_boxes, gt_embeddings, gt_labels, H, W)
        objectness_scores, pos_roi_boxes, final_box_trans, final_boxes, emb_output = self.recognition(feats, roi_boxes, pos_target_boxes, pos_target_labels)
        return objectness_scores, pos_roi_boxes, final_box_trans, emb_output, label_injection, pos_scores, neg_scores, pos_trans, pos_trans_targets, pos_target_boxes, pos_target_labels
    
    @torch.no_grad()
    def _test(self, img, gt_boxes, region_proposals, **kwargs):
        """
        Parameters
        ----------
        img               : torch.tensor of shape (B, 1, H, W)
                            Mean normalized grayscale image

        Returns
        -------
        objectness_scores : torch.tensor of shape (P'+N', 1)
                            Score for the final boxes to be positive
        final_boxes       : torch.tensor of shape (P', 4)
                            Transformed RoI boxes
        emb_output        : torch.tensor of shape (P', E)
                            Embedding for each RoI
        """
        assert img.shape[0] == 1, "Batch size should be 1"
        H, W = img.shape[2:]
        feats = self.feat_cnn(img) #(B,C,H/8,W/8)
        rpn_boxes = self.localization_layer(feats, H, W, **kwargs) # (M,4)
        if self.batch_size > 0:
            if rpn_boxes.shape[0] > self.batch_size:
                roi_scores, roi_boxes, roi_embeddings = self.function_loop(feats, rpn_boxes)
            else:
                roi_scores, _, _, roi_boxes, roi_embeddings = self.recognition(feats, rpn_boxes) #(M,1), (M,4), (M,4), (M,4), (M,E)    
            if gt_boxes.shape[0] > self.batch_size:
                gt_scores, _, gt_embeddings = self.function_loop(feats, rpn_boxes)
            else:
                gt_scores, _, _, _, gt_embeddings = self.recognition(feats, gt_boxes) #(M,1), (M,4), (M,4), (M,4), (M,E)    
            if region_proposals.shape[0] > self.batch_size:
                rp_scores, rp_boxes, rp_embeddings = self.function_loop(feats, region_proposals)
            else:
                rp_scores, _, _, rp_boxes, rp_embeddings = self.recognition(feats, region_proposals) #(M,1), (M,4), (M,4), (M,4), (M,E)    
        else:
            roi_scores, _, _, roi_boxes, roi_embeddings = self.recognition(feats, rpn_boxes) #(M,1), (M,4), (M,4), (M,4), (M,E)    
            gt_scores, _, _, _, gt_embeddings = self.recognition(feats, gt_boxes) #(M,1), (M,4), (M,4), (M,4), (M,E)    
            rp_scores, _, _, _, rp_embeddings = self.recognition(feats, region_proposals) #(M,1), (M,4), (M,4), (M,4), (M,E)    
        
        roi_boxes = torchvision.ops.box_convert(roi_boxes, 'cxcywh', 'xyxy')
        gt_boxes = torchvision.ops.box_convert(gt_boxes, 'cxcywh', 'xyxy')
        region_proposals = torchvision.ops.box_convert(region_proposals, 'cxcywh', 'xyxy')
        
        verbose = False
        if verbose and self.final_nms_thresh > 0:
            print(f'Applying NMS on {roi_boxes.shape[0]} boxes with {self.final_nms_thresh} threshold.')
        
        # Apply NMS on the final boxes
        if self.final_nms_thresh > 0:
            idx = torchvision.ops.nms(roi_boxes, roi_scores.squeeze(1), self.final_nms_thresh)
            roi_scores = roi_scores[idx]
            roi_boxes = roi_boxes[idx]
            roi_embeddings = roi_embeddings[idx]

        return roi_scores, roi_boxes, roi_embeddings, gt_scores, gt_boxes, gt_embeddings, rp_scores, region_proposals, rp_embeddings

    def function_loop(self, cnn_features, boxes):
        boxes = []
        embeddings = []
        logprobs = []
        for v in torch.split(boxes, self.batch_size, dim = 0):
            lp, _, _, box, emb = self.recognition(cnn_features, v)
            embeddings.append(emb)
            logprobs.append(lp)
            boxes.append(box)
        boxes = torch.cat(boxes, 0)
        embeddings = torch.cat(embeddings, 0)
        logprobs = torch.cat(logprobs, 0)
        return boxes, embeddings, logprobs

class EmbeddingNet(nn.Module):

    def __init__(self, opt):
        super(EmbeddingNet, self).__init__()
        self.layers = nn.Sequential(nn.Linear(opt.descriptor_size, opt.fc_size),
                                    nn.BatchNorm1d(opt.fc_size),
                                    nn.Tanh(),
                                    nn.Linear(opt.fc_size, opt.fc_size),
                                    nn.BatchNorm1d(opt.fc_size),
                                    nn.Tanh(),
                                    nn.Linear(opt.fc_size, opt.fc_size),                                    
                                    nn.BatchNorm1d(opt.fc_size),
                                    nn.Tanh(),
                                    nn.Linear(opt.fc_size, opt.embedding_size))
    
    def forward(self, x):
        if x.shape[0] == 1 and self.train:
            self.eval()
            y = F.normalize(self.layers(x), 1)#2)
            self.train()
            return y
        else:
            return F.normalize(self.layers(x), 1)#2)

class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import box_utils
from presnet import PreActResNet34
from localization_layer import LocalizationLayer

class WordSpottingModel(nn.Module):

    def __init__(self, opt):
        super(WordSpottingModel, self).__init__()
        self.embedding_net = EmbeddingNet(opt)

        cnn = PreActResNet34()
        # Chop the CNN, RPN after module 6 of CNN
        self.feat_cnn = cnn[:6]

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

        opt.field_centers = {x0, y0, sx, sy}
        self.localization_layer = LocalizationLayer(opt)
        # Recognition base network; The final layers of the PResNet and the FC embedding network
        # Produces roi_codes of dimension opt.descriptor_size
        self.recog_base = cnn[6:]
        # Objectness branch; outputs positive / negative probabilities for final boxes
        self.objectness_branch = nn.Linear(opt.descriptor_size, 1)
        # Final box regression branch; regresses from RPN boxes to final boxes
        self.box_reg_branch = nn.Linear(opt.descriptor_size, 4)
        self.finetune_cnn = False
        self.final_nms_thresh = opt.final_nms_thresh
    
    def some_forward(self, roi_feats, roi_boxes, gt_boxes, gt_labels):
        roi_codes = self.recog_base(roi_feats)
        objectness_scores = self.objectness_branch(roi_codes)

        pos_roi_codes = pos_slicer(roi_codes, gt_labels)
        pos_roi_boxes = pos_slicer(roi_boxes, gt_boxes)

        final_box_trans = self.box_reg_branch(pos_roi_codes)
        final_boxes = box_utils.apply_box_transform(pos_roi_boxes, final_box_trans)

        emb_output = self.embedding_net(pos_roi_codes)

        return objectness_scores, pos_roi_boxes, final_box_trans, final_boxes, emb_output
    

    def forward(self, input):
        # -- Make sure the input is (1, 1, H, W)
        assert input.ndim == 4 and input.shape[0] == 1 and input.shape[1] == 1
        H, W = input.shape[2:]

        if self.train:
            assert not self._called_forward, 'Must call setGroundTruth before training-time forward pass')
        
        # self.output = self.net:forward(input)
        feats = self.feat_cnn(input)
        roi_features, roi_boxes, pos_target_boxes, pos_target_labels = self.localization_layer(feats, H, W)
        class_scores, pos_roi_boxes, final_box_trans, final_boxes, emb_output = self.some_forward(roi_features, roi_boxes, pos_target_boxes, pos_target_labels)

        # -- At test-time, apply NMS to final boxes
        verbose = False
        if verbose:
            print(f'Before final NMS there are {self.output[3].shape[0]} boxes')
            print(f'Using NMS threshold of {self.final_nms_thresh}')
        
        if not self.train and self.final_nms_thresh > 0:
            # -- We need to apply the same NMS mask to the final boxes, their
            # -- objectness scores, and the output from the language model
            final_boxes = final_boxes.float()
            class_scores = class_scores.float()
            emb_output = emb_output.float()
            boxes_x1y1x2y2 = torchvision.ops.box_convert(final_boxes, 'cxcywh', 'xyxy')
            idx = torchvision.ops.nms(boxes_x1y1x2y2, class_scores[:, 0], self.final_nms_thresh)
            class_scores = class_scores[idx]
            final_boxes = final_boxes[idx]
            emb_output = emb_output[idx]

        return class_scores, pos_roi_boxes, final_box_trans, final_boxes, emb_output


def pos_slicer(features, gt_features):
    if gt_features.numel() == 0:
        output = features
    else:
        P = gt_features.shape[0]
        assert P <= features.shape[0], "Must have P <= N"
        output = features[{{1, P}}]
    return output


class EmbeddingNet(nn.Module):

    def __init__(self, opt):
        super(EmbeddingNet, self).__init__()
        self.layers = nn.Sequential(nn.Linear(opt.descriptor_size, opt.fc_size),
                                    nn.BatchNorm1D(opt.fc_size),
                                    nn.Tanh(),
                                    nn.Linear(opt.fc_size, opt.fc_size),
                                    nn.BatchNorm1D(opt.fc_size),
                                    nn.Tanh(),
                                    nn.Linear(opt.fc_size, opt.fc_size),
                                    nn.BatchNorm1D(opt.fc_size),
                                    nn.Tanh(),
                                    nn.Linear(opt.fc_size, opt.embedding_size))
    
    def forward(self, x):
        return F.normalize(self.layers(x), 2)

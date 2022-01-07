import torch
import torch.nn as nn
import torch.nn.functional as F

import os


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        model = WordSpottingModel(opt)

        if os.path.exists(opt.checkpoint_start_from):
            model.load_state_dict(torch.load(opt.checkpoint_start_from, map_location = device))

            model.opt.end_objectness_weight = opt.end_objectness_weight
            model.nets.localization_layer.opt.mid_objectness_weight = opt.mid_objectness_weight
            model.nets.localization_layer.opt.mid_box_reg_weight = opt.mid_box_reg_weight
            model.crits.box_reg_crit.w = opt.end_box_reg_weight
            local rpn = model.nets.localization_layer.nets.rpn
            rpn:findModules('nn.RegularizeLayer')[1].w = opt.box_reg_decay
            model.opt.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
            model.opt.embedding_weight = opt.embedding_weight

            # -- TODO: Move into a reset function in BoxSampler
            model.nets.localization_layer.nets.box_sampler_helper.box_sampler.vocab_size = opt.vocab_size
            model.nets.localization_layer.nets.box_sampler_helper.box_sampler.histogram = torch.ones(opt.vocab_size)

    end

        # -- Find all Dropout layers and set their probabilities
        local dropout_modules = model.nets.recog_base:findModules('nn.Dropout')
        for i, dropout_module in ipairs(dropout_modules) do
            dropout_module.p = opt.drop_prob
        end

class WordSpottingModel(nn.Module):

    def __init__(self, opt):
        super(WordSpottingModel, self).__init__()
        self.embedding_net = EmbeddingNet(opt)

        cnn = PreActResNet34()
        # Insert RPN layer after module 6 in the CNN part (size divided by 8)
        opt.output_height, opt.output_width = 8, 20
        fc_dim = opt.descriptor_size

        # Now that we have the indices, actually chop up the CNN.
        self.nets.conv_net1 = nn.Identity()
        self.nets.conv_net2 = cnn[:6]

        self.net:add(self.nets.conv_net1)
        self.net:add(self.nets.conv_net2)

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

        # -- local x0, y0, sx, sy = net_utils.compute_field_centers(conv_full)
        self.opt.field_centers = {x0, y0, sx, sy}

        self.nets.localization_layer = LocalizationLayer(opt)
        self.net:add(self.nets.localization_layer)

        # -- Recognition base network; The final layers of the PResNet and the FC embedding network
        # -- Produces roi_codes of dimension fc_dim.
        self.recog_base = cnn[6:]
        if opt.emb_pretrain != '':
            self.recog_base = cp.model.nets.recog_base

        # -- Objectness branch; outputs positive / negative probabilities for final boxes
        self.objectness_branch = nn.Linear(fc_dim, 1)

        # -- Final box regression branch; regresses from RPN boxes to final boxes
        self.box_reg_branch = nn.Linear(fc_dim, 4)

        self.nets.recog_net = self:_buildRecognitionNet()
        self.net:add(self.nets.recog_net)

        self:training()
        self.finetune_cnn = false
    
    def forward(self, roi_feats, roi_boxes, gt_boxes, gt_labels):
        roi_codes = self.recog_base(roi_feats)
        objectness_scores = self.objectness_branch(roi_codes)

        pos_roi_codes = pos_slicer(roi_codes, gt_labels)
        pos_roi_boxes = pos_slicer(roi_boxes, gt_boxes)

        final_box_trans = self.box_reg_branch(pos_roi_codes)
        final_boxes = apply_box_transform(pos_roi_boxes, final_box_trans)

        emb_output = self.embedding_net(pos_roi_codes)

        return objectness_scores, pos_roi_boxes, final_box_trans, final_boxes, emb_output, gt_boxes, gt_labels
    

    function WordSpottingModel:updateOutput(input)
        # -- Make sure the input is (1, 1, H, W)
        assert input.ndim == 4 and input.shape[0] == 1 and input.shape[1] == 1
        h, w = input.shape[2:]
        self.nets.localization_layer:setImageSize(H, W)

        if self.train then
            assert not self._called_forward, 'Must call setGroundTruth before training-time forward pass')
        
        self.output = self.net:forward(input)

        # -- At test-time, apply NMS to final boxes
        verbose = False
        if verbose:
            print(f'Before final NMS there are {self.output[3].shape[0]} boxes')
            print(f'Using NMS threshold of {self.opt.final_nms_thresh}')
        
        if not self.train and self.opt.final_nms_thresh > 0:
            # -- We need to apply the same NMS mask to the final boxes, their
            # -- objectness scores, and the output from the language model
            final_boxes_float = self.output[3].float()
            class_scores_float = self.output[0].float()
            emb_output_float = self.output[4].float()
            boxes_scores = torch.FloatTensor((final_boxes_float.shape[0], 5))
            boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
            boxes_scores[:, :4] = boxes_x1y1x2y2
            boxes_scores[:, 4] = class_scores_float[:, 0]
            idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)
            output[3] = final_boxes_float:index(1, idx):typeAs(self.output[4])
            output[0] = class_scores_float:index(1, idx):typeAs(self.output[1])
            output[4] = emb_output_float:index(1, idx):typeAs(self.output[5])

        return self.output
    end


def pos_slicer(features, gt_features):
    if gt_features.numel() == 0:
        output = features
    else:
        P = gt_features.shape[0]
        assert P <= features.shape[0], "Must have P <= N"
        output = features[{{1, P}}]
    return output

def apply_box_transform(boxes, trans):

    assert boxes.shape[-1] == 4, 'Last dim of boxes must be 4'
    assert trans.shape[-1] == 4, 'Last dim of trans must be 4'
    
    boxes = boxes.contiguous().view(-1, 4)
    trans = trans.contiguous().view(-1, 4)

    xa, ya, wa, ha = torch.split(boxes, 1, -1)
    tx, ty, tw, th = torch.split(trans, 1, -1)

    x = tx * wa + xa
    y = ty * ha + ya
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha

    out = torch.cat([x, y, w, h], dim = 1)
    
    return out


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

class RecognitionNet(nn.Module):

    def __init__(self, opt):
        super(RecognitionNet, self).__init__()

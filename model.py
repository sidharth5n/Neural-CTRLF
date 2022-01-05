import torch
import torch.nn as nn
import torch.nn.functional as F

class WordSpottingModel(nn.Module):

    def __init__(self, opt):
        super(WordSpottingModel, self).__init__()
        self.embedding_net = EmbeddingNet(opt)

        cnn = PreActResNet34()
        # Insert RPN layer after module 6 in the CNN part (size divided by 8)
        conv_start1, conv_end1 = 1, 4 # these will not be finetuned for efficiency
        conv_start2, conv_end2 = 5, 6 # these will be finetuned possibly
        recog_start, recog_end = 7, 14 # The rest
        opt.input_dim = 128 # number of feature maps as input to the localization layer
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
        self.nets.recog_base = cnn[6:]
        if opt.emb_pretrain != '':
            self.nets.recog_base = cp.model.nets.recog_base

        # -- Objectness branch; outputs positive / negative probabilities for final boxes
        self.nets.objectness_branch = nn.Linear(fc_dim, 1)
        self.nets.objectness_branch.weight:normal(0, opt.std)
        self.nets.objectness_branch.bias:zero()

        # -- Final box regression branch; regresses from RPN boxes to final boxes
        self.nets.box_reg_branch = nn.Linear(fc_dim, 4)
        self.nets.box_reg_branch.weight:zero()
        self.nets.box_reg_branch.bias:zero()

        self.nets.recog_net = self:_buildRecognitionNet()
        self.net:add(self.nets.recog_net)

        # -- Set up Criterions
        self.crits = {}
        self.crits.objectness_crit = nn.LogisticCriterion()
        self.crits.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)
        self.crits.emb_crit = nn.CosineEmbeddingCriterion(opt.cosine_margin)

        self:training()
        self.finetune_cnn = false

def _buildRecognitionNet():
    roi_feats = nn.Identity()()
    roi_boxes = nn.Identity()()
    gt_boxes = nn.Identity()()
    gt_labels = nn.Identity()()

    roi_codes = self.nets.recog_base(roi_feats)
    objectness_scores = self.nets.objectness_branch(roi_codes)

    pos_roi_codes = nn.PosSlicer(){roi_codes, gt_labels}
    pos_roi_boxes = nn.PosSlicer(){roi_boxes, gt_boxes}

    final_box_trans = self.nets.box_reg_branch(pos_roi_codes)
    final_boxes = nn.ApplyBoxTransform(){pos_roi_boxes, final_box_trans}

    emb_output = self.nets.embedding_net(pos_roi_codes)

    # -- Annotate nodes
    roi_codes:annotate{name='recog_base'}
    objectness_scores:annotate{name='objectness_branch'}
    pos_roi_codes:annotate{name='code_slicer'}
    pos_roi_boxes:annotate{name='box_slicer'}
    final_box_trans:annotate{name='box_reg_branch'}

    local inputs = {roi_feats, roi_boxes, gt_boxes, gt_labels}
    local outputs = {
    objectness_scores, pos_roi_boxes, final_box_trans, final_boxes,
    emb_output, gt_boxes, gt_labels,
    }
    local mod = nn.gModule(inputs, outputs)
    mod.name = 'recognition_network'
    return mod
end

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

import torch
import torch.nn as nn

class BoxSampler(nn.Module):

    def __init__(self, options = {}):
        super(BoxSampler, self).__init__()
        self.low_thresh = utils.getopt(options, 'low_thresh', 0.2)
        self.high_thresh = utils.getopt(options, 'high_thresh', 0.75)
        self.batch_size = utils.getopt(options, 'batch_size', 256)
        self.biased_sampling = utils.getopt(options, 'biased_sampling', false)
        self.vocab_size = utils.getopt(options, 'vocab_size')
        
        self.x_min, self.x_max = nil, nil
        self.y_min, self.y_max = nil, nil

        # -- Used for biased sampling scheme
        self.labels = None
        # -- self.vocab_size = 1126
        self.histogram = torch.ones(self.vocab_size)

    def unpack_dims(self, input_boxes, target_boxes)
        N, B1 = input_boxes.shape[:2]
        B2 = target_boxes.shape[1]
  
        assert input_boxes.shape[2] == 4 and target_boxes.shape[2] == 4
        assert target_boxes.shape[0] == N
  
        return N, B1, B2

    def setBounds(self, bounds):
        self.x_min = utils.getopt(bounds, 'x_min', None)
        self.x_max = utils.getopt(bounds, 'x_max', None)
        self.y_min = utils.getopt(bounds, 'y_min', None)
        self.y_max = utils.getopt(bounds, 'y_max', None)

    def set_labels(self, labels):
        self.labels = labels

# --[[
#   Tries to sample boxes so that the most uncommon words in the page are used
# ]]--
    def sample(self, input_idx)
        # -- Get the occurrences of the current labels from the histogram
        local occ = self.histogram:index(1, self.labels)

        # -- Invert the occurrences, words that haven't occurred get a weigh of 1
        occ = torch.cdiv(torch.ones(occ:size()):cuda(), occ)

        # -- Map the weights onto each positive sample
        pos_mask_nonzero = self.pos_mask:nonzero():view(-1)
        local ii = input_idx[self.pos_mask]:view(pos_mask_nonzero:size(1)) -- ii = positive rpn/input boxes
        
        local out = occ:index(1, ii):float()
        return out # -- return it as weighting for the multinomial sampling

    def update_histogram(self, pos_target_idx)
        # -- Update the histogram with which samples with corresponding label were chosen for a particular batch
        local l = self.labels:index(1, pos_target_idx)
        self.histogram:indexCopy(1, l, self.histogram:index(1, l) + 1)

# --[[
#   Inputs:
#   - input: list of two elements:
#     - input_boxes: Tensor of shape (1, B1, 4) giving coordinates of generated
#       box coordinates in (xc, yc, w, h) format
#     - target_boxes: Tensor of shape (1, B2, 4) giving coordinates of target
#       box coordinates in (xc, yc, w, h) format.
#   Returns: List of three elements:
#     - pos_input_idx: LongTensor of shape (P,) where each element is in the
#       range [1, B1] and gives indices into input_boxes for the positive boxes.
#     - pos_target_idx: LongTensor of shape (P,) where each element is in the
#       range [1, B2] and gives indices into target_boxes for the positive boxes.
#     - neg_input_idx: LongTensor of shape (M,) where each element is in the
#       range [1, B1] and gives indices into input_boxes for the negative boxes.
#   Based on the ious between the generated boxes and the target boxes, we sample
#   P positive boxes and M negative boxes, where P + M = batch_size. The ith
#   positive box is given by input_boxes[{1, pos_input_idx[i]}], and it was matched
#   to target_boxes[{1, pos_target_idx[i]}]. The ith negative box is given by
#   input_boxes[{1, neg_input_idx[i]}].
# --]]
def forward(input_boxes, target_boxes):
    N, B1, B2 = unpack_dims(input_boxes, target_boxes)

    ious = box_iou(input_boxes, target_boxes) #-- N x B1 x B2
    input_max_iou, input_idx = torch.max(ious, dim = 2)   #-- N x B1
    target_max_iou, target_idx = torch.max(ious, dim = 1) #-- N x B2
    
    # -- Pick positive and negative boxes based on IoU thresholds
    self.pos_mask = torch.gt(input_max_iou, self.high_thresh) #-- N x B1
    self.neg_mask = torch.lt(input_max_iou, self.low_thresh)  #-- N x B1

    # -- Maybe find the input boxes that fall outside the boundaries
    # -- and exclude them from the pos and neg masks
    if self.x_min and self.y_min and self.x_max and self.y_max:
        # -- Convert from (xc, yc, w, h) to (x1, y1, x2, y2) format to make
        # -- it easier to find boxes that are out of bounds
        local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_boxes)
        x_min_mask = torch.lt(boxes_x1y1x2y2[..., 0], self.x_min)
        y_min_mask = torch.lt(boxes_x1y1x2y2[..., 1], self.y_min)
        x_max_mask = torch.gt(boxes_x1y1x2y2[..., 2], self.x_max)
        y_max_mask = torch.gt(boxes_x1y1x2y2[..., 3], self.y_max)
        self.pos_mask[x_min_mask] = 0
        self.pos_mask[y_min_mask] = 0
        self.pos_mask[x_max_mask] = 0
        self.pos_mask[y_max_mask] = 0
        self.neg_mask[x_min_mask] = 0
        self.neg_mask[y_min_mask] = 0
        self.neg_mask[x_max_mask] = 0
        self.neg_mask[y_max_mask] = 0

    # -- Count as positive each input box that has maximal IoU with each target box,
    # -- even if it is outside the bounds or does not meet the thresholds.
    # -- This is important since things will crash if we don't have at least one
    # -- positive box.
    self.pos_mask.scatter_(1, target_idx, 1)
    self.neg_mask.scatter_(1, target_idx, 0)

    assert N == 1, "Only 1-element minibatches are supported"
    self.pos_mask = self.pos_mask:view(B1):byte()
    self.neg_mask = self.neg_mask:view(B1):byte()

    if self.neg_mask:sum() == 0 then
        # -- There were no negatives; this can happen if all input boxes are either:
        # -- (1) An input box with maximal IoU with a target box
        # -- (2) Out of bounds, therefore clipped
        # -- (3) max IoU to all target boxes is in the range [low_thresh, high_thresh]
        # -- This should be a pretty rare case, but we still need to handle it.
        # -- Ideally this should do something like sort the non-positive in-bounds boxes
        # -- by their max IoU to target boxes and set the negative set to be those with
        # -- minimal IoU to target boxes; however this is complicated so instead we'll
        # -- just sample from non-positive boxes to get negatives.
        # -- We'll also log this event in the __GLOBAL_STATS__ table; if this happens
        # -- regularly then we should handle it more cleverly.

        self.neg_mask:mul(self.pos_mask, -1):add(1) # -- set neg_mask to inverse of pos_mask
        k = 'BoxSampler no negatives'
        old_val = utils.__GLOBAL_STATS__[k] or 0
        utils.__GLOBAL_STATS__[k] = old_val + 1

    pos_mask_nonzero = self.pos_mask:nonzero():view(-1)
    neg_mask_nonzero = self.neg_mask:nonzero():view(-1)

    total_pos = pos_mask_nonzero:size(1)
    total_neg = neg_mask_nonzero:size(1)

    local num_pos = math.min(self.batch_size / 2, total_pos)
    local num_neg = self.batch_size - num_pos

    # -- We always sample positives without replacemet
    pos_p = pos_mask_non_zeros.ones_like()
    if self.biased_sampling:
        pos_p = self:sample(input_idx)

    pos_sample_idx = torch.multinomial(pos_p, num_pos, False)

    # -- We sample negatives with replacement if there are not enough negatives
    # -- to fill out the minibatch
    neg_p = neg_mask_nonzero.ones_like(total_neg)
    neg_replace = (total_neg < num_neg)
    if neg_replace:
        local k = 'BoxSampler negative with replacement'
        local old_val = utils.__GLOBAL_STATS__[k] or 0
        utils.__GLOBAL_STATS__[k] = old_val + 1
    
    neg_sample_idx = torch.multinomial(neg_p, num_neg, neg_replace)
    
    if self.debug_pos_sample_idx:
        pos_sample_idx = self.debug_pos_sample_idx
    
    if self.debug_neg_sample_idx:
        neg_sample_idx = self.debug_neg_sample_idx

    local pos_input_idx = pos_mask_nonzero:index(1, pos_sample_idx)
    local pos_target_idx = input_idx:index(2, pos_input_idx):view(num_pos)
    local neg_input_idx = neg_mask_nonzero:index(1, neg_sample_idx)

    if self.biased_sampling:
        self:update_histogram(pos_target_idx)

    return pos_input_idx, pos_target_idx, neg_input_idx


class BoxSamplerHelper(nn.Module):
    def __init__(self, options)
        super(BoxSamplerHelper, self).__init__()
        if options and options.box_sampler:
            # -- Optional dependency injection for testing
            self.box_sampler = options.box_sampler
        else:
            self.box_sampler = nn.BoxSampler(options)
        end
        
        self.output = {{torch.Tensor()}, {torch.Tensor()}, {torch.Tensor()}}
        self.gradInput = {torch.Tensor()}

        self.num_pos, self.num_neg = nil, nil
        self.pos_input_idx = nil
        self.pos_target_idx = nil
        self.neg_input_idx = nil
        self.y = nil
    end


    def setBounds(self, bounds):
      # -- Just forward to the underlying sampler
      self.box_sampler:setBounds(bounds)

    def set_labels(self, labels):
      self.box_sampler:set_labels(labels)

    # --[[
    #   Input:
      
    #   List of two lists. The first list contains data about the input boxes,
    #   and the second list contains data about the target boxes.
    #   The first element of the first list is input_boxes, a Tensor of shape (N, B1, 4)
    #   giving coordinates of the input boxes in (xc, yc, w, h) format.
    #   All other elements of the first list are tensors of shape (N, B1, Di) parallel to
    #   input_boxes; Di can be different for each element.
    #   The first element of the second list is target_boxes, a Tensor of shape (N, B2, 4)
    #   giving coordinates of the target boxes in (xc, yc, w, h) format.
    #   All other elements of the second list are tensors of shape (N, B2, Dj) parallel
    #   to target_boxes; Dj can be different for each Tensor.
      
    #   Returns a list of three lists:
    #   The first list contains data about positive input boxes. The first element is of
    #   shape (P, 4) and contains coordinates of positive boxes; the other elements
    #   correspond to the additional input data about the input boxes; in particular the
    #   ith element has shape (P, Di).
    #   The second list contains data about target boxes corresponding to positive
    #   input boxes. The first element is of shape (P, 4) and contains coordinates of
    #   target boxes corresponding to sampled positive input boxes; the other elements
    #   correspond to the additional input data about the target boxes; in particular the
    #   jth element has shape (P, Dj).
    #   The third list contains data about negative input boxes. The first element is of
    #   shape (M, 4) and contains coordinates of negative input boxes; the other elements
    #   correspond to the additional input data about the input boxes; in particular the
    #   ith element has shape (M, Di).
    # --]]
    def forward(input_data, target_data):
      
      input_boxes = input_data[0]
      target_boxes = target_data[0]
      N = input_boxes.shape[0]
      assert N == 1, 'Only minibatches of 1 are supported'

      # -- Run the sampler to get the indices of the positive and negative boxes
      pos_input_idx, pos_target_idx, neg_input_idx = self.box_sampler(input_boxes, target_boxes)

      # -- Inject mismatching pairs for the cosine embedding loss here, and save which pairs are mismatched
      n = pos_target_idx.shape[0]
      self.y = torch.ones(n, dtype = pos_target_idx.dtype, device = pos_target_idx.device)
      frac = 5 #-- The fraction of how many positive pairs are kept 3 = 66% positive
      z = torch.randperm(n):lt(n / frac) #--:type(self.pos_target_idx:type())
      self.y[z] = -1
      p = z:clone():double() #--:type(self.pos_target_idx:type())
      p[p:eq(0)] = 1e-14 #-- Kind of sucky solution, but should happen very rarely.
      
      # -- Randomly select other word embeddings from the same page.
      # -- TODO: Should perhaps switch to any word embedding from the dataset, but requires those as input to model / constructor
      # -- self.pos_target_idx[z] = torch.multinomial(p, z:sum(), true):type(self.pos_target_idx:type())
      modified_pos_target_idx = pos_target_idx.clone()
      modified_pos_target_idx[z] = torch.multinomial(p, z:sum(), True):type(pos_target_idx:type())

      # -- Resize the output. We need to allocate additional tensors for the
      # -- input data and target data, then resize them to the right size.
      self.num_pos = pos_input_idx:size(1)
      self.num_neg = neg_input_idx:size(1)
      for i = 1, #input_data do
        # -- The output tensors for additional data will be lazily instantiated
        # -- on the first forward pass, which is probably after the module has been
        # -- cast to the right datatype, so we make sure the new Tensors have the
        # -- same type as the corresponding elements of the input.
          dtype = input_data[i]:type()
          if #self.output[1] < i:
            table.insert(self.output[1], torch.Tensor():type(dtype))
          end
          if #self.output[3] < i then
            table.insert(self.output[3], torch.Tensor():type(dtype))
          end
          local D = input_data[i]:size(3)
          self.output[1][i]:resize(self.num_pos, D)
          self.output[3][i]:resize(self.num_neg, D)
        end
        for i = 1, #target_data do
          local dtype = target_data[i]:type()
          if #self.output[2] < i then
            table.insert(self.output[2], torch.Tensor():type(dtype))
          end
          local D = target_data[i]:size(3)
          self.output[2][i]:resize(self.num_pos, D)
        end

        # -- Now use the indicies to actually copy data from inputs to outputs
        for i = 1, #input_data do
          self.output[1][i]:index(input_data[i], 2, self.pos_input_idx)
          self.output[3][i]:index(input_data[i], 2, self.neg_input_idx)
          -- The call to index adds an extra dimension at the beginning for batch
          -- size, but since its a singleton we just squeeze it out
          local D = input_data[i]:size(3)
          self.output[1][i] = self.output[1][i]:view(self.num_pos, D)
          self.output[3][i] = self.output[3][i]:view(self.num_neg, D)
        end
        for i = 1, #target_data do
          if i == 1 then
            self.output[2][i]:index(target_data[i], 2, self.pos_target_idx)
          elseif i == 2 then
            self.output[2][i]:index(target_data[i], 2, modified_pos_target_idx)
          end    
          local D = target_data[i]:size(3)
          self.output[2][i] = self.output[2][i]:view(self.num_pos, D)
        end

        return self.output
    end

def box_iou(box1, box2):
    """
    boxes are in xc, yc, w, h format

    Parameters
    ----------
    box1 : torch.tensor of shape (N, B1, 4)
    box2 : torch.tensor of shape (N, B2, 4)

    Returns
    -------
    iou : torch.tensor of shape (N, B1, B2)
    """
    N, B1 = box1.shape[:2]
    B2 = box2.shape[1]
    area1 = box1[...,2] * box1[...,3]
    area2 = box2[...,2] * box2[...,3]
    
    area1 = area1.view(N, B1, 1).expand(N, B1, B2)
    area2 = area2.view(N, 1, B2).expand(N, B1, B2)
    
    # Convert boxes to x1, y1, x2, y2 format
    box1 = box_utils.xcycwh_to_x1y1x2y2(box1) # -- N x B1 x 4
    box2 = box_utils.xcycwh_to_x1y1x2y2(box2) # -- N x B2 x 4
    box1 = box1.view(N, B1, 1, 4).expand(N, B1, B2, 4)
    box2 = box2.view(N, 1, B2, 4).expand(N, B1, B2, 4)
    
    x0 = torch.maximum(box1[..., 0], box2[..., 0])
    y0 = torch.maximum(box1[..., 1], box2[..., 1])
    x1 = torch.minimum(box1[..., 2], box2[..., 2])
    y1 = torch.minimum(box1[..., 3], box2[..., 3])
    
    w = torch.maximum(x1 - x0, 0)
    h = torch.maximum(y1 - y0, 0)
    
    intersection = w * h

    iou = intersection / (area1 + area2 - intersection)
    
    return iou
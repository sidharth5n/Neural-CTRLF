import torch
import torch.nn as nn

import box_utils

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

class BoxSampler(nn.Module):

    def __init__(self, opt):
        super(BoxSampler, self).__init__()
        self.low_thresh = opt.sampler_low_thresh
        self.high_thresh = opt.sampler_high_thresh
        self.batch_size = opt.sampler_batch_size
        self.biased_sampling = opt.biased_sampling
        self.vocab_size = utils.getopt(opt, 'vocab_size')
        
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None

        # -- Used for biased sampling scheme
        self.labels = None
        # -- self.vocab_size = 1126
        self.histogram = torch.ones(self.vocab_size)

    def unpack_dims(self, input_boxes, target_boxes):
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

    def sample(self, input_idx):
        """
        Tries to sample boxes so that the most uncommon words in the page are used
        """
        # Get the occurrences of the current labels from the histogram
        occ = self.histogram[(self.histogram == self.labels).nonzero(as_tuple = True)]
        # Invert the occurrences, words that haven't occurred get a weight of 1
        occ = 1 / occ
        # Map the weights onto each positive sample
        ii = input_idx[self.pos_mask.nonzero(as_tuple = True)].view(-1)#-- ii = positive rpn/input boxes
        out = occ[ii]
        return out # -- return it as weighting for the multinomial sampling

    def update_histogram(self, pos_target_idx):
        # -- Update the histogram with which samples with corresponding label were chosen for a particular batch
        l = self.labels[pos_target_idx]
        self.histogram[l] = self.histogram[l] + 1

    def forward(self, input_boxes, target_boxes):
        """
        Based on the ious between the generated boxes and the target boxes, P positive 
        boxes and M negative boxes are sampleed, where P + M = batch_size. The ith
        positive box is given by input_boxes[0, pos_input_idx[i]], and it was matched
        to target_boxes[0, pos_target_idx[i]]. The ith negative box is given by
        input_boxes[0, neg_input_idx[i]].

        Parameters
        ----------
        input_boxes    : torch.tensor of shape (1, B1, 4) 
                         Coordinates of generated box coordinates in (xc, yc, w, h) format
        target_boxes   : torch.tensor of shape (1, B2, 4) 
                         Coordinates of target box coordinates in (xc, yc, w, h) format
        
        Returns
        -------
        pos_input_idx  : torch.LongTensor of shape (P,) 
                         Each element is in the range [1, B1] and gives indices into 
                         input_boxes for the positive boxes.
        pos_target_idx : torch.LongTensor of shape (P,) 
                         Each element is in the range [1, B2] and gives indices into 
                         target_boxes for the positive boxes.
        neg_input_idx  : torch.LongTensor of shape (M,) 
                         Each element is in the range [1, B1] and gives indices into 
                         input_boxes for the negative boxes.
        """
        assert input_boxes.shape[0] == 1 and target_boxes.shape[0] == 1, "Only 1-element minibatches are supported"
        N, B1, B2 = self.unpack_dims(input_boxes, target_boxes)

        ious = box_iou(input_boxes, target_boxes)             # N x B1 x B2
        input_max_iou, input_idx = torch.max(ious, dim = 2)   # N x B1
        target_max_iou, target_idx = torch.max(ious, dim = 1) # N x B2
        
        # Pick positive and negative boxes based on IoU thresholds
        self.pos_mask = torch.gt(input_max_iou, self.high_thresh) # N x B1
        self.neg_mask = torch.lt(input_max_iou, self.low_thresh)  # N x B1

        # -- Maybe find the input boxes that fall outside the boundaries
        # -- and exclude them from the pos and neg masks
        if self.x_min and self.y_min and self.x_max and self.y_max:
            # -- Convert from (xc, yc, w, h) to (x1, y1, x2, y2) format to make
            # -- it easier to find boxes that are out of bounds
            boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(input_boxes)
            x_mask = (boxes_x1y1x2y2[..., 0] < self.x_min) + (boxes_x1y1x2y2[..., 2] > self.x_max)
            y_mask = (boxes_x1y1x2y2[..., 1] < self.y_min) + (boxes_x1y1x2y2[..., 3] > self.y_max)
            self.pos_mask[x_mask + y_mask] = False
            self.neg_mask[x_mask + y_mask] = False

        # -- Count as positive each input box that has maximal IoU with each target box,
        # -- even if it is outside the bounds or does not meet the thresholds.
        # -- This is important since things will crash if we don't have at least one
        # -- positive box.
        self.pos_mask.scatter_(1, target_idx, True)
        self.neg_mask.scatter_(1, target_idx, False)
        
        self.pos_mask = self.pos_mask.view(-1)
        self.neg_mask = self.neg_mask.view(-1)

        if self.neg_mask.sum() == 0:
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

            self.neg_mask = 1 - self.pos_mask # -- set neg_mask to inverse of pos_mask
            k = 'BoxSampler no negatives'
            old_val = utils.__GLOBAL_STATS__[k] or 0
            utils.__GLOBAL_STATS__[k] = old_val + 1

        pos_mask_nonzero = self.pos_mask.squeeze().nonzero(as_tuple = False)
        neg_mask_nonzero = self.neg_mask.squeeze().nonzero(as_tuple = False)

        total_pos = pos_mask_nonzero.shape[0]
        total_neg = neg_mask_nonzero.shape[0]

        num_pos = min(self.batch_size // 2, total_pos)
        num_neg = self.batch_size - num_pos

        # -- We always sample positives without replacemet
        pos_p = pos_mask_nonzero.ones_like()
        if self.biased_sampling:
            pos_p = self.sample(input_idx)

        pos_sample_idx = torch.multinomial(pos_p, num_pos, False)

        # -- We sample negatives with replacement if there are not enough negatives
        # -- to fill out the minibatch
        neg_p = neg_mask_nonzero.ones_like(total_neg)
        neg_replace = (total_neg < num_neg)
        if neg_replace:
            k = 'BoxSampler negative with replacement'
            old_val = utils.__GLOBAL_STATS__[k] or 0
            utils.__GLOBAL_STATS__[k] = old_val + 1
        
        neg_sample_idx = torch.multinomial(neg_p, num_neg, neg_replace)
        
        if self.debug_pos_sample_idx:
            pos_sample_idx = self.debug_pos_sample_idx
        
        if self.debug_neg_sample_idx:
            neg_sample_idx = self.debug_neg_sample_idx

        pos_input_idx = pos_mask_nonzero[pos_sample_idx]
        pos_target_idx = input_idx[:, pos_input_idx].view(num_pos)
        neg_input_idx = neg_mask_nonzero[neg_sample_idx]

        if self.biased_sampling:
            self.update_histogram(pos_target_idx)

        return pos_input_idx, pos_target_idx, neg_input_idx


class BoxSamplerHelper(nn.Module):
    def __init__(self, opts):
        super(BoxSamplerHelper, self).__init__()
        if opts and opts.box_sampler:
            # -- Optional dependency injection for testing
            self.box_sampler = opts.box_sampler
        else:
            self.box_sampler = BoxSampler(opts)

    def setBounds(self, bounds):
        # -- Just forward to the underlying sampler
        self.box_sampler.setBounds(bounds)

    def set_labels(self, labels):
        self.box_sampler.set_labels(labels)

    def forward(self, input_data, target_data, gt_labels):
        """
        Parameters
        ----------
        input_data    : list
                        The first element of the first list is input_boxes, a Tensor of 
                        shape (N, B1, 4) giving coordinates of the input boxes in 
                        (xc, yc, w, h) format. All other elements of the first list are 
                        tensors of shape (N, B1, Di) parallel to input_boxes; Di can be 
                        different for each element.
        target_data   : list
                        The first element of the second list is target_boxes, a Tensor of 
                        shape (N, B2, 4) giving coordinates of the target boxes in 
                        (xc, yc, w, h) format. All other elements of the second list are 
                        tensors of shape (N, B2, Dj) parallel to target_boxes; Dj can be 
                        different for each Tensor.

        Returns
        -------
        positive_list : list
                        The first list contains data about positive input boxes. 
                        The first element is of shape (P, 4) and contains coordinates of 
                        positive boxes; the other elements correspond to the additional 
                        input data about the input boxes; in particular the ith element 
                        has shape (P, Di).
        target_list   : list
                        The second list contains data about target boxes corresponding to 
                        positive input boxes. The first element is of shape (P, 4) and 
                        contains coordinates of target boxes corresponding to sampled 
                        positive input boxes; the other elements correspond to the 
                        additional input data about the target boxes; in particular the
                        jth element has shape (P, Dj).
        negative_list : list
                        The third list contains data about negative input boxes. The first 
                        element is of shape (M, 4) and contains coordinates of negative 
                        input boxes; the other elements correspond to the additional input 
                        data about the input boxes; in particular the ith element has 
                        shape (M, Di).
        y             : torch.tensor of shape (, )
        """
        self.set_labels(gt_labels)
        input_boxes = input_data[0]
        target_boxes = target_data[0]
        N = input_boxes.shape[0]
        assert N == 1, 'Only minibatches of 1 are supported'

        # -- Run the sampler to get the indices of the positive and negative boxes
        pos_input_idx, pos_target_idx, neg_input_idx = self.box_sampler(input_boxes, target_boxes)
        # -- Inject mismatching pairs for the cosine embedding loss here, and save which pairs are mismatched
        n = pos_target_idx.shape[0]
        y = torch.ones(n, dtype = pos_target_idx.dtype, device = pos_target_idx.device)
        frac = 5 #-- The fraction of how many positive pairs are kept 3 = 66% positive
        z = torch.randperm(n, device = pos_target_idx.device) < (n / frac) #--:type(self.pos_target_idx:type())
        y[z] = -1
        p = z.clone().double() #--:type(self.pos_target_idx:type())
        p[p == 0] = 1e-14 #-- Kind of sucky solution, but should happen very rarely.
        
        # -- Randomly select other word embeddings from the same page.
        modified_pos_target_idx = pos_target_idx.clone()
        modified_pos_target_idx[z] = torch.multinomial(p, z.sum(), True).to(pos_target_idx)

        positive_list, negative_list = [], []
        # -- Now use the indicies to actually copy data from inputs to outputs
        for i in range(len(input_data)):
            positive_list.append(input_data[i][0, pos_input_idx])
            negative_list.append(input_data[i][0, neg_input_idx])
        
        target_list = []
        for i in range(len(target_data)):
            if i == 0:
                target_list.append(target_data[i][0, pos_target_idx])
            elif i == 1:
                target_list.append(target_data[i][0, modified_pos_target_idx])
                
        return positive_list, target_list, negative_list, y
import torch
import torch.nn as nn
import torchvision

def box_iou(box1, box2):
    """
    Find IoU of the boxes.

    Parameters
    ----------
    box1 : torch.tensor of shape (N, B1, 4)
           Boxes are in (xc, yc, w, h) format
    box2 : torch.tensor of shape (N, B2, 4)
           Boxes are in (xc, yc, w, h) format

    Returns
    -------
    iou : torch.tensor of shape (N, B1, B2)
          IoU betwen each box in box1 and box2
    """
    N, B1 = box1.shape[:2]
    B2 = box2.shape[1]
    area1 = box1[...,2] * box1[...,3]
    area2 = box2[...,2] * box2[...,3]
    
    area1 = area1.view(N, B1, 1).expand(N, B1, B2)
    area2 = area2.view(N, 1, B2).expand(N, B1, B2)
    
    # Convert boxes to x1, y1, x2, y2 format
    box1 = torchvision.ops.box_convert(box1, 'cxcywh', 'xyxy') # N x B1 x 4
    box2 = torchvision.ops.box_convert(box2, 'cxcywh', 'xyxy') # N x B2 x 4
    box1 = box1.view(N, B1, 1, 4).expand(N, B1, B2, 4)
    box2 = box2.view(N, 1, B2, 4).expand(N, B1, B2, 4)
    
    x0 = torch.maximum(box1[..., 0], box2[..., 0])
    y0 = torch.maximum(box1[..., 1], box2[..., 1])
    x1 = torch.minimum(box1[..., 2], box2[..., 2])
    y1 = torch.minimum(box1[..., 3], box2[..., 3])
    
    w = torch.maximum(x1 - x0, torch.zeros_like(x1))
    h = torch.maximum(y1 - y0, torch.zeros_like(y1))
    
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
        self.histogram = torch.ones(opt.vocab_size)

    def sampling_weights(self, input_idx, pos_mask, gt_labels):
        """
        Computes weights for multinomial sampling such that the most uncommon words 
        in the page are used.

        Parameters
        ----------
        input_idx : torch.tensor of shape (N, B1)
                    Indices of target boxes with which input boxes have maximal IoU
        pos_mask  : torch.tensor of shape (N, B1)
                    Mask for positive RPN/input boxes
        gt_labels : torch.tensor of shape (N, P)
                    Ground truth labels of the target boxes

        Returns
        -------
        weights   : torch.Tensor of shape 
                    Weights for multinomial sampling
        """
        # Get the occurrences of the current labels from the histogram
        occ = torch.gather(self.histogram, 0, gt_labels.squeeze())
        # Invert the occurrences, words that haven't occurred get a weight of 1
        occ = 1 / occ
        # Get indices of all the positive boxes
        ii = input_idx[pos_mask.nonzero(as_tuple = True)].view(-1)
        # Map the weights onto each positive sample
        weights = occ[ii]
        return weights

    def update_histogram(self, pos_target_idx, gt_labels):
        """
        Update the histogram with which samples with corresponding label were chosen 
        for a particular batch

        Parameters
        ----------
        pos_target_idx : torch.tensor of shape (P', )
                         Gives indices into target_boxes for the positive boxes. Each 
                         element is in the range (0, B2).
        gt_labels      : torch.tensor of shape (P, )
                         Ground truth labels of the target boxes
        """
        l, counts = torch.unique(gt_labels[pos_target_idx], return_counts = True)
        self.histogram[l] = self.histogram[l] + counts

    def sample(self, input_boxes, target_boxes, gt_labels, bounds = None):
        """
        Samples P' positive and M' negative boxes based on IoUs between the 
        generated and target boxes. P' + M' = opt.sampler_batch_size
        
        The ith positive box is given by input_boxes[0, pos_input_idx[i]] and it 
        was matched to target_boxes[0, pos_target_idx[i]]. The ith negative box is 
        given by input_boxes[0, neg_input_idx[i]].
        
        Parameters
        ----------
        input_boxes    : torch.tensor of shape (N, B1, 4) 
                         Coordinates of generated box coordinates in (xc, yc, w, h) format
        target_boxes   : torch.tensor of shape (N, B2, 4) 
                         Coordinates of target box coordinates in (xc, yc, w, h) format
        gt_labels      : torch.tensor of shape (N, P)
                         Ground truth labels of the target boxes
        
        Returns
        -------
        pos_input_idx  : torch.LongTensor of shape (P',) 
                         Gives indices into input_boxes for the positive boxes. Each 
                         element is in the range [0, B1).
        pos_target_idx : torch.LongTensor of shape (P',) 
                         Gives indices into target_boxes for the positive boxes. Each 
                         element is in the range (0, B2).
        neg_input_idx  : torch.LongTensor of shape (M',) 
                         Each element is in the range [0, B1) and gives indices into 
                         input_boxes for the negative boxes.
        """
        # Compute IoU between input_boxes and target_boxes
        ious = box_iou(input_boxes, target_boxes)             # (N,B1,B2)
        # Find max IoU for each input box
        input_max_iou, input_idx = torch.max(ious, dim = 2)   # (N,B1)
        # Find max IoU for each target box
        target_max_iou, target_idx = torch.max(ious, dim = 1) # (N,B2)
        # Pick positive and negative boxes based on IoU thresholds
        pos_mask = torch.gt(input_max_iou, self.high_thresh) # (N,B1)
        neg_mask = torch.lt(input_max_iou, self.low_thresh)  # (N,B1)
        # Disable boxes which are outside image boundaries (if provided)
        if bounds:
            # Convert from (xc,yc,w,h) to (x1,y1,x2,y2) format
            boxes_x1y1x2y2 = torchvision.ops.box_convert(input_boxes, 'cxcywh', 'xyxy')
            # Find boxes which are outside x-bounds
            x_mask = (boxes_x1y1x2y2[..., 0] < bounds['x_min']) + (boxes_x1y1x2y2[..., 2] > bounds['x_max'])
            # Find boxes which are outside y-bounds
            y_mask = (boxes_x1y1x2y2[..., 1] < bounds['y_min']) + (boxes_x1y1x2y2[..., 3] > bounds['y_max'])
            # Mask positive and negative boxes which are outside either x or y-bounds
            pos_mask[x_mask + y_mask] = False
            neg_mask[x_mask + y_mask] = False

        # Count as positive each input box that has maximal IoU with each target box,
        # even if it is outside the bounds or does not meet the thresholds.
        # This is important since things will crash if we don't have at least one
        # positive box.        
        pos_mask.scatter_(1, target_idx, True)
        neg_mask.scatter_(1, target_idx, False)

        # If there are no negatives (very rare)
        if neg_mask.sum() == 0:
            # This can happen if all input boxes are either:
            # (1) An input box with maximal IoU with a target box
            # (2) Out of bounds, therefore clipped
            # (3) Max IoU to all target boxes is in the range [low_thresh, high_thresh]
            # Ideally this should do something like sort the non-positive in-bounds boxes
            # by their max IoU to target boxes and set the negative set to be those with
            # minimal IoU to target boxes; however this is complicated so instead we'll
            # just sample from non-positive boxes to get negatives.
            neg_mask = 1 - pos_mask # -- set neg_mask to inverse of pos_mask
        # Get indices of all positive and negative boxes
        pos_mask_nonzero = pos_mask.squeeze(0).nonzero(as_tuple = False).squeeze(1)
        neg_mask_nonzero = neg_mask.squeeze(0).nonzero(as_tuple = False).squeeze(1)
        # Find no. of positive and negative boxes
        total_pos = pos_mask_nonzero.shape[0]
        total_neg = neg_mask_nonzero.shape[0]
        # Ensure positive samples are atleast half of sampler batch size
        num_pos = min(self.batch_size // 2, total_pos)
        num_neg = self.batch_size - num_pos

        if self.biased_sampling:
            # Find sampling weights based on no. of occurrences
            pos_p = self.sampling_weights(input_idx, pos_mask, gt_labels)
        else:
            # Uniform sampling weights
            pos_p = torch.ones_like(pos_mask_nonzero, dtype = torch.float32)
        # Sample positives without replacement
        pos_sample_idx = torch.multinomial(pos_p, num_pos, False)
        # Uniform sampling weights
        neg_p = torch.ones_like(neg_mask_nonzero, dtype = torch.float32)
        # Sample with replacement if there are not enough negatives to fill out the minibatch
        neg_replace = (total_neg < num_neg)
        neg_sample_idx = torch.multinomial(neg_p, num_neg, neg_replace)
        # Get corresponding positive input indices
        pos_input_idx = pos_mask_nonzero[pos_sample_idx]
        # Get corresponding target box indices
        pos_target_idx = input_idx[:, pos_input_idx].view(num_pos)
        # Get corresponding negative input indices
        neg_input_idx = neg_mask_nonzero[neg_sample_idx]

        if self.biased_sampling:
            # Update histogram if using biased sampling
            self.update_histogram(pos_target_idx, gt_labels.squeeze(0))

        return pos_input_idx, pos_target_idx, neg_input_idx

    def forward(self, input_data, target_data, gt_labels, bounds):
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
        gt_labels     : torch.tensor of shape (N, P)
                        Ground truth labels of the target boxes

        Returns
        -------
        positive_list   : list
                          The first list contains data about positive input boxes. 
                          The first element is of shape (N, P, 4) and contains coordinates of 
                          positive boxes; the other elements correspond to the additional 
                          input data about the input boxes; in particular the ith element 
                          has shape (N, P, Di).
        target_list     : list
                          The second list contains data about target boxes corresponding to 
                          positive input boxes. The first element is of shape (N, P, 4) and 
                          contains coordinates of target boxes corresponding to sampled 
                          positive input boxes; the other elements correspond to the 
                          additional input data about the target boxes; in particular the
                          jth element has shape (N, P, Dj).
        negative_list   : list
                          The third list contains data about negative input boxes. The first 
                          element is of shape (N, M, 4) and contains coordinates of negative 
                          input boxes; the other elements correspond to the additional input 
                          data about the input boxes; in particular the ith element has 
                          shape (N, M, Di).
        label_injection : torch.tensor of shape (P', )
                          Each element is {-1, 1}. If 1, mismatching pair has been injected
        """
        input_boxes = input_data[0]
        target_boxes = target_data[0]
        # Run the sampler to get the indices of the positive and negative boxes
        pos_input_idx, pos_target_idx, neg_input_idx = self.sample(input_boxes, target_boxes, gt_labels, bounds)
        # Get new target indices with mismatching pairs injected
        new_pos_target_idx, label_injection = self.inject_mismatching_pairs(pos_target_idx)

        positive_list, negative_list = [], []
        # Now use the indicies to actually copy data from inputs to outputs
        for i in range(len(input_data)):
            positive_list.append(input_data[i][0, pos_input_idx])
            negative_list.append(input_data[i][0, neg_input_idx])
        
        target_list = []
        for i in range(len(target_data)):
            if i == 0:
                target_list.append(target_data[i][0, pos_target_idx])
            elif i == 1:
                target_list.append(target_data[i][0, new_pos_target_idx])

        return positive_list, target_list, negative_list, label_injection
    
    @torch.no_grad()
    def inject_mismatching_pairs(self, pos_target_idx):
        """
        Injects mismatching pairs for the cosine embedding loss.

        Parameters
        ----------
        pos_target_idx          : torch.tensor of shape (P', )
                                  Each element is in the range (0, B2) and gives 
                                  indices into target_boxes for the positive boxes.

        Returns
        -------
        modified_pos_target_idx : torch.tensor of shape (P', )
                                  Target indices modified with mismatching pairs
        label_injection         : torch.tensor of shape (P', )
                                  Each element is {-1, 1}. If 1, mismatching pair 
                                  has been injected
                                  
        """
        # No. of positive boxes
        n = pos_target_idx.shape[0]
        # The fraction of how many positive pairs are kept 3 = 66% positive
        frac = 5
        # Compute a random permutation of the indices and keep frac elements only
        z = torch.randperm(n, device = pos_target_idx.device) < (n / frac)
        # Find injected labels
        label_injection = torch.ones_like(pos_target_idx)
        label_injection[z] = -1
        p = z.clone().double()
        # Randomly select other word embeddings from the same page.
        modified_pos_target_idx = pos_target_idx.clone()
        modified_pos_target_idx[z] = torch.multinomial(p, z.sum(), True).to(pos_target_idx)

        return modified_pos_target_idx, label_injection

        
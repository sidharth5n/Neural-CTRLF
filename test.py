"""
This is the testing script
"""

import os
import argparse
import torch

from models.WordSpottingModel import WordSpottingModel
from dataloader import DataLoader
from misc import utils
from opts import str2bool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'washington',
                        help = 'The HDF5 file to load data from')
    parser.add_argument('--dataset_path', type = str, default = 'data',
                        help = 'Directory where pre-processed datasets are saved')
    parser.add_argument('--out_dir', type = str, default = 'descriptors', 
                        help = 'Directory for saving model outputs')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoints', 
                        help = 'Path to the directory where checkpoints are saved')
    parser.add_argument('--id', type = str, required = True, 
                        help = 'An id identifying this run/job; useful for cross-validation')
    parser.add_argument('--device', type = str, default = 'cpu', choices = ['cuda', 'cpu'],
                        help = 'Whether to use cuda or cpu')
    parser.add_argument('--split', type = str, default = 'test', choices = ['test', 'val'], 
                        help = 'Which split to evaluate; either val or test.')
    parser.add_argument('--augment', type = str2bool, default = False,  
                        help = "Whether to use augmented data")
    parser.add_argument('--fold', type = str2bool, default = False, 
                        help = 'Whether to use 4-fold cross validation')
    parser.add_argument('--rpn_nms_thresh', type = float, default = 0.7,
                        help = '')
    parser.add_argument('--final_nms_thresh', type = float, default = -1, 
                        help = '')
    parser.add_argument('--num_proposals', type = int, default = -1, 
                        help = 'How many proposals to use with the RPN, default = the same as the DTP')
    parser.add_argument('--batch_size', type = int, default = 1024,
                        help = 'No. of boxes to be processed at a time for RPN and DTP')

    args = parser.parse_args()

    return args

def test(opt):

    device = torch.device('cuda' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)
    
    checkpoint_path = os.path.join(opt.checkpoint_path, 
                                   opt.id + ('_augmented' if opt.augment else '') + f'_fold_{opt.fold}' if opt.fold else '')
    infos = utils.load(os.path.join(checkpoint_path, 'infos.pkl'))

    for k in vars(infos['opt']).keys():
        if k not in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model                    

    model = WordSpottingModel(opt).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location = device))
    model.eval()

    data_loader = DataLoader(opt, split = opt.split)

    all_boxes = []
    all_logprobs = []
    all_embeddings = []
    all_gt_embeddings = []
    all_gt_scores = []
    all_gt_boxes = []
    all_gt_targets = []
    all_rp_embeddings = []
    all_rp_scores = []
    all_region_proposals = []
    all_infos = []

    append = lambda x, y: y.append(x.cpu().numpy())

    for i, data in enumerate(data_loader, start = 1):
        img, gt_boxes, _, gt_targets, region_proposals, infos = [x.to(device) for x in data]
        if opt.num_proposals <= 0:
            num_proposals = region_proposals.shape[1]
        roi_scores, roi_boxes, roi_embeddings, gt_scores, gt_boxes, gt_embeddings, rp_scores, region_proposals, rp_embeddings = model(img, gt_boxes.float(), region_proposals.float(), num_proposals = num_proposals)
        append(roi_boxes, all_boxes)
        append(roi_scores, all_logprobs)
        append(roi_embeddings, all_embeddings)
        append(gt_embeddings, all_gt_embeddings)
        append(gt_boxes, all_gt_boxes)
        append(gt_scores, all_gt_scores)
        append(gt_targets, all_gt_targets)
        append(region_proposals, all_region_proposals)
        append(rp_scores, all_rp_scores)
        append(rp_embeddings, all_rp_embeddings)
        append(infos, all_infos)
        
        print(f'Processed image {i}/{len(data_loader)} of {opt.split} split, detected {roi_boxes.shape[0]} boxes')

    data = {'roi_boxes'        : all_boxes,
            'roi_scores'       : all_logprobs,
            'roi_embeddings'   : all_embeddings,
            'gt_boxes'         : all_gt_boxes,
            'gt_scores'        : all_gt_scores,
            'gt_embeddings'    : all_gt_embeddings,
            'gt_targets'       : all_gt_targets,
            'region_proposals' : all_region_proposals,
            'rp_scores'        : all_rp_scores,
            'rp_embeddings'    : all_rp_embeddings,
            'infos'            : all_infos
            }
    
    out_path = os.path.join(opt.out_dir, f'{opt.dataset}_{opt.embedding}_' + (f'fold_{opt.fold}' if opt.fold else 'no_fold') + '_descriptor.pkl')
    utils.save(data, out_path)


if __name__ == '__main__':
    args = parse_args()
    if args.fold:
        for fold in range(4):
            args.fold = fold
            test(args)
    else:
        test(args)
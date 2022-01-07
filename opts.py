import argparse

def str2bool(v):
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def parse_args():

    parser = argparse.ArgumentParser()
    
    # Model hyper parameters
    parser.add_argument('--rpn_hidden_dim', type = int, default = 128, 
                        help = 'Hidden size for the extra convolution in the RPN')
    parser.add_argument('--rnn_size', type = int, default = 512, 
                        help = 'Number of units to use at each layer of the RNN')
    parser.add_argument('-input_encoding_size', type = int, default = 512, 
                        help = 'Dimension of the word vectors to use in the RNN')
    parser.add_argument('--input_dim', type = int, default = 128,
                        help = 'Number of feature maps as input to the localization layer')
    parser.add_argument('--train_remove_outbounds_boxes', type = str2bool, default = True, 
                        help = 'Whether to ignore out-of-bounds boxes for sampling at training time')
    parser.add_argument('--embedding', type = str, choices = ['dct', 'phoc'], default = 'dct', 
                        help = 'Which embedding to use, dct or phoc')
    parser.add_argument('--embedding_size', type = int, choices = [108, 540], default = 108,
                        help = 'Size of embedding - 108 for dct and 540 for phoc')
    parser.add_argument('--descriptor_size', type = int, default = 512,
                        help = '')
    parser.add_argument('--fc_size', type = int, default = 4096,
                        help = '')
    
    # Sampler hyper parameters
    parser.add_argument('--sampler_batch_size', type = int, default = 256, 
                        help = 'Batch size to use in the box sampler')
    parser.add_argument('--sampler_high_thresh', type = float, default = 0.75, 
                        help = 'Boxes with IoU greater than this with a GT box are considered positives')
    parser.add_argument('--sampler_low_thresh', type = float, default = 0.4, 
                        help = 'Boxes with IoU less than this with all GT boxes are considered negatives')
    parser.add_argument('--biased_sampling', type = str2bool, default = True, 
                        help = 'Whether or not to try to bias sampling to use uncommon words as often as possible.')

    # Loss function weights
    parser.add_argument('--mid_box_reg_weight', type = float, default = 0.01,
                        help = 'Weight for box regression in the RPN')
    parser.add_argument('--mid_objectness_weight', type = float, default = 0.01, 
                        help = 'Weight for box classification in the RPN')
    parser.add_argument('--end_box_reg_weight', type = float, default = 0.1, 
                        help = 'Weight for box regression in the recognition network')
    parser.add_argument('--end_objectness_weight', type = float, default = 0.1, 
                        help = 'Weight for box classification in the recognition network')
    parser.add_argument('--embedding_weight', type = float, default = 3.0, 
                        help = 'Weight for embedding loss')
    parser.add_argument('--weight_decay', type = float, default = 1e-5, 
                        help = 'L2 weight decay penalty strength')
    parser.add_argument('--box_reg_decay', type = float, default = 5e-5, 
                        help = 'Strength of pull that boxes experience towards their anchor')
    parser.add_argument('--cosine_margin', type = float, default = 0.1, 
                        help = 'margin for the cosine loss')
    
    # Data input settings
    parser.add_argument('--dataset', type = str, default = '', 
                        help = 'HDF5 file containing the preprocessed dataset (from proprocess.py)')
    parser.add_argument('--val_dataset', type = str, default = 'washington', 
                        help = 'HDF5 file containing the preprocessed dataset (from proprocess.py)')
    parser.add_argument('--fold', type = int, default = 1, 
                        help = 'which fold to use')
    parser.add_argument('--dataset_path', type = str, default = 'data/dbs/', 
                        help = 'HDF5 file containing the preprocessed dataset (from proprocess.py)')
    parser.add_argument('--debug_max_train_images', type = int, default = -1,
                        help = 'Use this many training images (for debugging); -1 to use all images')

    # Optimization
    parser.add_argument('--learning_rate', type = float, default = 2e-3, 
                        help = 'learning rate to use')
    parser.add_argument('--reduce_lr_every', type = int, default = 10000, 
                        help = 'reduce learning rate every x iterations')
    parser.add_argument('--optim_beta1', type = float, default = 0.9, 
                        help = 'beta1 for adam')
    parser.add_argument('--optim_beta2', type = float, default = 0.999, 
                        help = 'beta2 for adam')
    parser.add_argument('--optim_epsilon', type = float, default = 1e-8, 
                        help = 'epsilon for smoothing')
    parser.add_argument('--drop_prob', type = float, default = 0.5, 
                        help = 'Dropout strength throughout the model.')
    parser.add_argument('--max_iters', type = int, default = 25000, 
                        help = 'Number of iterations to run; -1 to run forever')
    parser.add_argument('--checkpoint_start_from', type = str, default = '', 
                        help = 'Load model from a checkpoint instead of random initialization.')
    parser.add_argument('--finetune_cnn_after', type = int, default = 1000, 
                        help = 'Start finetuning CNN after this many iterations (-1 = never finetune)')
    parser.add_argument('--val_images_use', type = int, default = -1, 
                        help = 'Number of validation images to use for evaluation; -1 to use all')

    # Model checkpointing
    parser.add_argument('--save_checkpoint_every', type = int, default = 1000, 
                        help = 'How often to save model checkpoints')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoints/',
                        help = 'path to where checkpoints are saved')
    parser.add_argument('--checkpoint_start_from', type = str, default = '',
                        help = 'Name of the checkpoint file to use')
        
    # Test-time model options (for evaluation)
    parser.add_argument('--test_rpn_nms_thresh', type = float, default = 0.7, 
                        help = 'Test-time NMS threshold to use in the RPN')
    parser.add_argument('--test_final_nms_thresh', type = float, default = -1, 
                        help = 'Test-time NMS threshold to use for final outputs')
    parser.add_argument('--test_num_proposals', type = int, default = 1000, 
                        help = 'Number of region proposal to use at test-time')

    # Visualization
    parser.add_argument('--print_every', type = int, default = 200, 
                        help = 'How often to print the latest images training loss.')
    parser.add_argument('--progress_dump_every', type = int, default = 100, 
                        help = 'After how many iterations do we write a progress report to vis/out ?. 0 = disable.')
    parser.add_argument('--losses_log_every', type = int, default = 10, 
                        help = 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

    # Misc
    parser.add_argument('--id', type = str, default = 'presnet', 
                        help = 'an id identifying this run/job; useful for cross-validation')
    parser.add_argument('--seed', type = int, default = 123, 
                        help = 'random number generator seed to use')
    parser.add_argument('--gpu', type = int, default = 0, 
                        help = 'which gpu to use. -1 = use CPU')
    # parser.add_argument('--timing', type = str2bool, default = false, 
    #                   help = 'whether to time parts of the net')
    parser.add_argument('--clip_final_boxes', type = str2bool, default = True, 
                        help = 'Whether to clip final boxes to image boundar')
    

    args = parser.parse_args()

    args.embedding_size = 108 if args.embedding == 'dct' else 540

    return args
import os
import time
import torch
import torch.nn as nn

from opts import parse_args
from losses import BoxRegressionCriterion
from misc import utils
from model import WordSpottingModel
from dataloader import DataLoader
from misc import utils

def train(opt):
    checkpoint_path = os.path.join(opt.checkpoint_path, opt.id)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    device = torch.device('cuda' if opt.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = WordSpottingModel(opt).to(device)
    model.train()

    # Set up loss functions
    objectness_crit = nn.BCEWithLogitsLoss()
    box_reg_crit = BoxRegressionCriterion()
    emb_crit = nn.CosineEmbeddingCriterion(margin = opt.cosine_margin)
    rpn_objectness_crit = nn.BCELoss()
    rpn_box_reg_crit = nn.SmoothL1Loss()  # for RPN box regression

    # Set up optimizer
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = opt.learning_rate,
                                 betas = (opt.optim_beta1, opt.optim_beta2))

    # Initialize infos and histories
    infos, histories = {}, {}

    # Load checkpoint if available
    if opt.resume_training:
        status = True
        for f in ['infos.pkl', 'histories.pkl', 'model.pt', 'optimizer.pt']:
            if not os.path.exists(os.path.join(checkpoint_path, f)):
                status = False
                break
        if status:
            infos = utils.load(os.path.join(checkpoint_path, 'infos.pkl'))
            histories = utils.load(os.path.join(checkpoint_path, 'histories.pkl'))
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model.pt'), map_location=device))
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'optimizer.pt'), map_location=device))

    train_loader = DataLoader(opt, infos)
    val_loader = DataLoader(opt, infos)

    iteration = infos.get('iter', 0)
    start_epoch = infos.get('start_epoch', 0)
    best_loss = infos.get('best_loss', None)
    start = time.time()

    for epoch in range(start_epoch, opt.max_epochs):
        for data in train_loader:
            img, embeddings, boxes, labels, region_proposals = [x.to(device) for x in data]
            objectness_scores, pos_roi_boxes, final_box_trans, emb_output, label_injection, pos_scores, neg_scores, pos_trans, pos_trans_targets, gt_boxes, gt_embeddings = model(img, embeddings, boxes, labels, region_proposals)
            # Set labels for objectness confidence
            with torch.no_grad():
                objectness_labels = objectness_scores.zeros_like()
                objectness_labels[:pos_roi_boxes.shape[0]] = 1
            # Compute objectness loss
            objectness_loss = objectness_crit(objectness_scores, objectness_labels)
            # Compute bbox regression loss
            box_reg_loss = box_reg_crit(pos_roi_boxes, final_box_trans, gt_boxes)
            # Compute embedding loss
            emb_loss = emb_crit(emb_output, gt_embeddings, label_injection)
            # Set labels for positive and negative boxes of RPN
            with torch.no_grad():
                obj_crit_label = torch.cat([pos_scores.ones_like(), neg_scores.zeros_like()], dim = 0)
            # Compute loss for positive and negative boxes of RPN
            obj_loss = rpn_objectness_crit(torch.cat([pos_scores, neg_scores], dim = 0), obj_crit_label)
            # Compute bbox regression loss of RPN
            rpn_reg_loss = rpn_box_reg_crit(pos_trans, pos_trans_targets)
            # Compute total loss
            total_loss = opt.end_objectness_weight * objectness_loss + opt.mid_box_reg_weight * rpn_reg_loss + \
                         opt.mid_objectness_weight * obj_loss + opt.end_box_reg_weight * box_reg_loss + \
                         opt.embedding_weight * emb_loss
            # Reset optimizer gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Update parameters
            optimizer.step()

            end = time.time()

            if (iteration % opt.display_loss_every) == 0:
                print(f'Iter {iteration} (epoch {epoch}/{opt.max_epochs}) : Loss = {total_loss:.2f}, time/batch = {end-start:.3f}')

            iteration += 1

            if (iteration % opt.log_loss_every) == 0:
                histories[iteration] = total_loss

            if (iteration % opt.perform_validation_every) == 0:
                model.eval()
                # perform something
                # val_loss =
                model.train()
                print(f'Validation loss : {val_loss}')

                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model_best.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer_best.pt'))

                    infos = {'iter'      : iteration,
                             'epoch'     : epoch,
                             'best_loss' : best_loss,
                             'loader'    : train_loader.state_dict(),
                             'opt'       : opt
                             }

                    utils.save(infos, os.path.join(checkpoint_path, 'infos_best.pkl'))
                    utils.save(histories, os.path.join(checkpoint_path, 'histories_best.pkl'))

                    print(f'Best checkpoint saved to {checkpoint_path}')

            if (iteration % opt.save_checkpoint_every) == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model.pt'))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer.pt'))

                infos = {'iter'      : iteration,
                         'epoch'     : epoch,
                         'best_loss' : best_loss,
                         'loader'    : train_loader.state_dict(),
                         'opt'       : opt
                         }

                utils.save(infos, os.path.join(checkpoint_path, 'infos.pkl'))
                utils.save(histories, os.path.join(checkpoint_path, 'histories.pkl'))

                print(f'Checkpoint saved to {checkpoint_path}')

            start = time.time()


if __name__ == '__main__':
    args = parse_args()
    train(args)

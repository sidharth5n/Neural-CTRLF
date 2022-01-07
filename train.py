# -- Set up Criterions
        self.crits = {}
        self.crits.objectness_crit = nn.LogisticCriterion()
        self.crits.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)
        self.crits.emb_crit = nn.CosineEmbeddingCriterion(opt.cosine_margin)

# -- Construct criterions
        self.nets.obj_crit_pos = nn.OurCrossEntropyCriterion() -- for objectness
        self.nets.obj_crit_neg = nn.OurCrossEntropyCriterion() -- for objectness
        self.nets.box_reg_crit = nn.SmoothL1Criterion() -- for RPN box regression

# -- Compute RPN box regression loss
        self:timeit('box_reg_loss:forward', function()
          local crit = self.nets.box_reg_crit
          local weight = self.opt.mid_box_reg_weight
          local loss = weight * crit:forward(self.pos_trans, self.pos_trans_targets)
          self.stats.losses.box_reg_loss = loss
        end)
        
        # -- Fish out the box regression loss
        local reg_mods = self.nets.rpn:findModules('nn.RegularizeLayer')
        assert(#reg_mods == 1)
        self.stats.losses.box_decay_loss = reg_mods[1].loss


# -- Compute objectness loss
        self:timeit('objectness_loss:forward', function()
          if self.pos_scores:type() ~= 'torch.CudaTensor' then
            # -- ClassNLLCriterion expects LongTensor labels for CPU score types,
            # -- but CudaTensor labels for GPU score types. self.pos_labels and
            # -- self.neg_labels will be casted by any call to self:type(), so
            # -- we need to cast them back to LongTensor for CPU tensor types.
            self.pos_labels = self.pos_labels:long()
            self.neg_labels = self.neg_labels:long()
          end
          self.pos_labels:resize(num_pos):fill(1)
          self.neg_labels:resize(num_neg):fill(2)
          local obj_loss_pos = self.nets.obj_crit_pos:forward(self.pos_scores, self.pos_labels)
          local obj_loss_neg = self.nets.obj_crit_neg:forward(self.neg_scores, self.neg_labels)
          local obj_weight = self.opt.mid_objectness_weight
          self.stats.losses.obj_loss_pos = obj_weight * obj_loss_pos
          self.stats.losses.obj_loss_neg = obj_weight * obj_loss_neg
        end)
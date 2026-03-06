import torch
import torch.nn as nn
import torch.nn.functional as F

from models import pre_processor, backbone_3d, head, roi_head, img_cls
from models.text_encoder.clip_encoder import TextEncoder

class RL3DF_gate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        # Text Encoder for Contrastive Learning
        self.text_encoder = TextEncoder(freeze=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # ln(14.28)
        
        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'img_cls', 'backbone_3d', 'head', 'roi_head', 
        ]
        self.list_modules = []
        self.build_rl_detector()

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_img_cls(self):
        if self.cfg_model.get('IMG_CLS', None) is None:
            return None
        
        module = img_cls.__all__[self.cfg_model.IMG_CLS.NAME]()
        return module 

    def build_pre_processor(self):
        if self.cfg_model.get('PRE_PROCESSOR', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
        return module 
    
    def build_pre_processor2(self):
        if self.cfg_model.get('PRE_PROCESSOR2', None) is None:
            return None
        
        module = pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)
        return module 

    def build_backbone_3d(self):
        cfg_backbone = self.cfg_model.get('BACKBONE', None)
        return backbone_3d.__all__[cfg_backbone.NAME](self.cfg)

    def build_head(self):
        if (self.cfg.MODEL.get('HEAD', None)) is None:
            return None
        module = head.__all__[self.cfg_model.HEAD.NAME](self.cfg)
        return module

    def build_roi_head(self):
        if (self.cfg.MODEL.get('ROI_HEAD', None)) is None:
            return None
        head_module = roi_head.__all__[self.cfg_model.ROI_HEAD.NAME](self.cfg)
        return head_module

    def forward(self, x):
        # 1. Forward Pass Modules
        for module in self.list_modules:
            x = module(x)
            
        # 2. Contrastive Learning (Training Only)
        if self.training and 'condition_prompts' in x:
            # Get Condition Token (CT) from img_cls backbone (assuming it's used as CT generator)
            # x['img_feat'] is likely the output of img_cls, we need a pooled vector
            # If img_cls returns a dict or specific tensor, adjust accordingly.
            # Assuming img_cls adds 'img_embedding' or we pool 'img_feat'
            
            if 'img_embedding' in x:
                condition_token = x['img_embedding'] # (B, D)
            else:
                # Fallback: Pool from img_feat if available, or skip
                # This depends on img_cls implementation. Let's assume we modify img_cls later to output this.
                pass 
            
            # For now, let's implement the loss calculation assuming we have condition_token
            if 'img_embedding' in x:
                text_features = self.text_encoder(x['condition_prompts']) # (B, D)
                
                # Normalize
                condition_token = F.normalize(condition_token, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Cosine similarity as logits
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * condition_token @ text_features.t()
                logits_per_text = logits_per_image.t()
                
                # Contrastive Loss (InfoNCE)
                batch_size = condition_token.shape[0]
                labels = torch.arange(batch_size, device=condition_token.device)
                
                loss_i2t = F.cross_entropy(logits_per_image, labels)
                loss_t2i = F.cross_entropy(logits_per_text, labels)
                contrastive_loss = (loss_i2t + loss_t2i) / 2
                
                # Add to total loss (x is a dict containing 'loss' from head)
                # Weighted sum: You can tune the weight lambda
                lambda_contrastive = 0.1 
                x['contrastive_loss'] = lambda_contrastive * contrastive_loss
                
                # Log the loss
                if 'logging' not in x: x['logging'] = {}
                x['logging']['loss_contrastive'] = contrastive_loss.item()
        
        # 3. Branch Selection (Router) - Placeholder
        # In a full implementation, we would use the condition_token to select branches here
        # or inside the backbone_3d module if it supports dynamic routing.
        # Since backbone_3d is already executed in the loop above, we assume
        # the routing logic would be integrated into backbone_3d or executed before it.
        # For this step, we focus on the contrastive learning part as requested.

        return x
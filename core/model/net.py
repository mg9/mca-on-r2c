# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch
from core.model.faster_rcnn import SimpleDetector 
from transformers import BertTokenizer, BertForSequenceClassification, BertModel


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, vocab_size=30000):
        super(Net, self).__init__()

        output_dir = './core/bert'

        self.detector = SimpleDetector(pretrained=True, average_pool=True, final_dim=2048)
        #self.bert =  BertForSequenceClassification.from_pretrained(output_dir)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
       
       
        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.lstm = nn.LSTM(
            input_size= 768,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )


        self.backbone = MCA_ED(__C)
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)

        self.output_feat_linear = nn.Linear(
            768,
            __C.FLAT_OUT_SIZE
        )

        self.backbone2 = MCA_ED(__C)

        self.output_proj_linear = nn.Linear(
            __C.HIDDEN_SIZE,
            vocab_size
        )

        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, 
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qas: torch.Tensor,
                reasons: torch.Tensor,
                attention_qa_masks: torch.Tensor,
                attention_reason_masks: torch.Tensor
                ):
        

        ## Move everything to CUDA
        images = images.to(torch.device("cuda"))
        boxes = boxes.to(torch.device("cuda"))
        box_mask = box_mask.to(torch.device("cuda"))
        segms = segms.to(torch.device("cuda"))
        attention_qa_masks = attention_qa_masks.to(torch.device("cuda"))
        attention_reason_masks = attention_reason_masks.to(torch.device("cuda"))
        qas = qas.to(torch.device("cuda"))
        reasons = reasons.to(torch.device("cuda"))


        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]
        
        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        img_feat = obj_reps['obj_reps']
    
        tokens_tensor = qas.view(qas.shape[0], qas.shape[2])
        attention_qa_masks = attention_qa_masks.view(attention_qa_masks.shape[0], attention_qa_masks.shape[2])
        
        reasons_tensor = reasons.view(reasons.shape[0], reasons.shape[2])
        attention_reason_masks = attention_reason_masks.view(attention_reason_masks.shape[0], attention_reason_masks.shape[2])

        # Predict hidden states features for each layer
        with torch.no_grad():
            outputs = self.bert(tokens_tensor, token_type_ids= None, attention_mask = attention_qa_masks)
            encoded_layers = outputs[0]
        lang_feat = encoded_layers

        # Make mask
        lang_feat_mask = self.make_mask(lang_feat)
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)
        
        
        ######## FIRST BACKBONE FOR QA & IMAGE
        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )
       

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)

        with torch.no_grad():
            outputs = self.bert(reasons_tensor, token_type_ids = None, attention_mask = attention_reason_masks)
            encoded_layers = outputs[0]
        output_feat = encoded_layers

      
        ######## SECOND BACKBONE FOR REASONING
        proj_feat = proj_feat.view(proj_feat.shape[0], 1 , proj_feat.shape[1])
        output_feat = self.output_feat_linear(output_feat)
        
        # Make mask
        proj_feat_mask = self.make_mask(proj_feat)
        output_feat_mask = self.make_mask(output_feat)

        # Backbone Framework
        output_feat, proj_feat = self.backbone2(
            output_feat,
            proj_feat,
            output_feat_mask,
            proj_feat_mask
        )

        output = self.output_proj_linear(output_feat)
        result = self.softmax(output)
    
        return result, reasons_tensor

    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature), dim=-1) == 0).unsqueeze(1).unsqueeze(2)



# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


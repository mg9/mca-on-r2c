# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.data_utils import tokenize
from core.data.data_utils import proc_img_feat, proc_ques

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
from os import path 


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        self.i2w = {}
        # Loading question word list
        self.stat_ques_list = json.load(open(__C.QAR_PATH['train'], 'r')) 

        # Loading qa list
        self.ques_list = []

        for i in  self.stat_ques_list:
            imgpath = self.__C.IMG_FEAT_PATH['train']  + str(i['img_fn']) + '.npz'
            #print("img path searching: ", imgpath)
            if path.exists(imgpath):
                "File exists. adding..."
                self.ques_list +=  [i]

        self.data_size = self.ques_list.__len__()
        print('== Dataset size:', self.data_size)

        # Tokenize
        self.token_to_ix, self.pretrained_emb, self.i2w = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_to_ix_ans, self.pretrained_emb_ans, self.i2w = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
     
        print("i2w len: ", len(self.i2w))
        print('Finished. Token vocab size: ', self.token_size)
        print('')


    def getpairmanual(self, idx):
        
        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_ix_iter = np.zeros(1)

        # Load the run data from list
        ques = self.ques_list[idx]

        # Process image feature from (.npz) file
        imgpath = str(ques['img_fn']) + '.npz'
        imgpath = self.__C.IMG_FEAT_PATH['train'] + imgpath

        img_feat = np.load(imgpath, allow_pickle=True)
        img_feat = img_feat['arr_0'].tolist()['obj_reps']
        rs_x = img_feat.shape[1]
        rs_y = img_feat.shape[2]
        img_feat = img_feat.view(rs_x, rs_y) 
        img_feat = img_feat.detach().numpy()

        img_feat_iter = proc_img_feat(img_feat, self.__C.IMG_FEAT_PAD_SIZE)

        # Process question
        ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN, 'question')

        # Process answer
        ans_ix_iter = proc_ques(ques, self.token_to_ix_ans, self.__C.MAX_TOKEN, 'answer')


        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_ix_iter)


    def __len__(self):
        return self.data_size
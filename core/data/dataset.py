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
from copy import deepcopy


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C
    
        coco = json.load(open(__C.COCO_PATH, 'r')) 
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}

        self.items = json.load(open(__C.QAR_PATH['train'], 'r')) 


        """
        # Loading qa list
        self.ques_list = []

        for i in  self.items:
            imgpath = self.__C.IMG_FEAT_PATH['train']  + str(i['img_fn']) + '.npz'
            if path.exists(imgpath):
                "File exists. adding..."
                self.ques_list +=  [i]

        self.data_size = self.ques_list.__len__()
        print('== Dataset size:', self.data_size)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.items)
        self.token_to_ix_ans, self.pretrained_emb_ans= tokenize(self.items)
        self.token_size = self.token_to_ix.__len__()
     
        print("i2w len: ", len(self.i2w))
        print('Finished. Token vocab size: ', self.token_size)
        print('')
        """



    def __getitem__(self, index):

        item = deepcopy(self.items[index])
        print("item: ", item)
        ###################################################################
       
        dets2use = np.ones(len(item['objects']), dtype=bool)
        # we will use these detections
        dets2use = np.where(dets2use)[0]

        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(os.path.join(__C.VCR_IMAGES_DIR, item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=True)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        ###################################################################
        # Load boxes.
        with open(os.path.join(__C.VCR_IMAGES_DIR, item['metadata_fn']), 'r') as f:
            metadata = json.load(f)

        # [nobj, 14, 14]
        segms = np.stack([make_mask(mask_size=14, box=metadata['boxes'][i], polygons_list=metadata['segms'][i])
                          for i in dets2use])

        # Chop off the final dimension, that's the confidence
        boxes = np.array(metadata['boxes'])[dets2use, :-1]
        # Possibly rescale them if necessary
        boxes *= img_scale
        boxes[:, :2] += np.array(padding[:2])[None]
        boxes[:, 2:] += np.array(padding[:2])[None]
        obj_labels = [self.coco_obj_to_ind[item['objects'][i]] for i in dets2use.tolist()]
        
        #if self.add_image_as_a_box:
        boxes = np.row_stack((window, boxes))
        segms = np.concatenate((np.ones((1, 14, 14), dtype=np.float32), segms), 0)
        obj_labels = [self.coco_obj_to_ind['__background__']] + obj_labels

        instance_dict = {}
        instance_dict['segms'] = ArrayField(segms, padding_value=0)
        instance_dict['objects'] = ListField([LabelField(x, skip_indexing=True) for x in obj_labels])

        if not np.all((boxes[:, 0] >= 0.) & (boxes[:, 0] < boxes[:, 2])):
            import ipdb
            ipdb.set_trace()
        assert np.all((boxes[:, 1] >= 0.) & (boxes[:, 1] < boxes[:, 3]))
        assert np.all((boxes[:, 2] <= w))
        assert np.all((boxes[:, 3] <= h))
        instance_dict['boxes'] = ArrayField(boxes, padding_value=-1)

        instance = Instance(instance_dict)
        instance.index_fields(self.vocab)
        print("image returned:", image.shape)
        print("image name: ",item['img_fn'])
        return image, instance

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
        return len(self.items)
# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from keras.preprocessing.sequence import pad_sequences

import numpy as np
import glob, json, torch, time, os
import torch.utils.data as Data
from copy import deepcopy
from vcr_util.box_utils import load_image, resize_image, to_tensor_and_normalize
from vcr_util.mask_utils import make_mask
from allennlp.data.fields import TextField, ListField, LabelField, SequenceLabelField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch
from transformers import BertTokenizer

class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C
    
        coco = json.load(open(__C.COCO_PATH, 'r')) 
        self.coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
        self.coco_obj_to_ind = {o: i for i, o in enumerate(self.coco_objects)}
        self.items = json.load(open(__C.QAR_PATH['train'], 'r')) 
        


    def __getitem__(self, index):
        max_token_length = 32
        instance_dict = {}

        item = deepcopy(self.items[index])
        question_str = item['question']
        reason_str = item['answer']

        output_dir = './core/bert'
        #tokenizer = BertTokenizer.from_pretrained(output_dir)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        encoded_qa_token_ids = tokenizer.encode(
                        question_str,                      
                        add_special_tokens = True 
                   )

        encoded_reason_token_ids = tokenizer.encode(
                        reason_str,                      
                        add_special_tokens = True 
                   )

        encoded_qa_token_ids = pad_sequences([encoded_qa_token_ids], maxlen=max_token_length, dtype="long", 
                          value=0, truncating="post", padding="post")

        encoded_reason_token_ids = pad_sequences([encoded_reason_token_ids], maxlen=max_token_length, dtype="long", 
                          value=0, truncating="post", padding="post")

        # Create attention masks
        attention_qa_masks = []
        attention_reason_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in encoded_qa_token_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_qa_masks.append(seq_mask) 
        
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in encoded_reason_token_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_reason_masks.append(seq_mask) 
        

        qa_token_ids = torch.LongTensor(encoded_qa_token_ids) 
        reason_token_ids = torch.LongTensor(encoded_reason_token_ids) 
        attention_qa_masks = torch.LongTensor(attention_qa_masks) 
        attention_reason_masks = torch.LongTensor(attention_reason_masks) 

        ###################################################################
       
        dets2use = np.ones(len(item['objects']), dtype=bool)
        # we will use these detections
        dets2use = np.where(dets2use)[0]

        ###################################################################
        # Load image now and rescale it. Might have to subtract the mean and whatnot here too.
        image = load_image(os.path.join(self.__C.VCR_IMAGES_PATH, item['img_fn']))
        image, window, img_scale, padding = resize_image(image, random_pad=True)
        image = to_tensor_and_normalize(image)
        c, h, w = image.shape

        ###################################################################
        # Load boxes.
        with open(os.path.join(self.__C.VCR_IMAGES_PATH, item['metadata_fn']), 'r') as f:
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
        return image, instance, qa_token_ids, reason_token_ids, attention_qa_masks, attention_reason_masks



    def __len__(self):
        return len(self.items)


def collate_fn(data, to_gpu=False):
    """Creates mini-batch tensors
    """
    images, instances, qas, reasons, attention_qa_masks, attention_reason_masks = zip(*data)
    qas = torch.stack(qas, 0)
    reasons = torch.stack(reasons, 0)
    attention_qa_masks = torch.stack(attention_qa_masks, 0)
    attention_reason_masks = torch.stack(attention_reason_masks, 0)

    images = torch.stack(images, 0)
    batch = Batch(instances)
    td = batch.as_tensor_dict()

    td['box_mask'] = torch.all(td['boxes'] >= 0, -1).long()
    td['images'] = images
    td['qas'] = qas
    td['reasons'] = reasons
    td['attention_qa_masks'] = attention_qa_masks
    td['attention_reason_masks'] = attention_reason_masks

    return td


class TheLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def from_dataset(cls, data, batch_size=3, num_workers=6, num_gpus=1, **kwargs):
        loader = cls(
            dataset=data,
            batch_size=batch_size, #* num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: collate_fn(x, to_gpu=True),
            pin_memory=False,
            **kwargs,
        )
        return loader
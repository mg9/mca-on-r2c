from core.data.dataset import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')
        self.dataset = DataSet(__C)


    def train(self, dataset):
        dataloader = Data.DataLoader(
            dataset,
            batch_size=self.__C.BATCH_SIZE,
            shuffle=False,
            num_workers=self.__C.NUM_WORKERS,
            pin_memory=self.__C.PIN_MEM,
            drop_last=True
        )
        for step, (
                image,
                instance
        ) in enumerate(dataloader):
            print("image is: ", image)
            print("instance is: ", instance)


    """def train(self, dataset):

        train, val, test = VCR.splits(only_use_relevant_dets=True)

        loader_params = {'batch_size': 1, 'num_gpus':1, 'num_workers':num_workers}
        train_loader = VCRLoader.from_dataset(train, **loader_params)


        # Obtain needed information
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb
        pretrained_emb_ans = dataset.pretrained_emb_ans
        data_size = dataset.data_size

        fixed_ans_size = 16
       
        # Define the MCAN model
        net = Net(
            self.__C,
            pretrained_emb,
            pretrained_emb_ans,
            token_size,
            fixed_ans_size
        )
        net.cuda()
        net.train()

        loss_fn = nn.NLLLoss()

        #Create checkpoint
        if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
            shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)
        os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

        optim = get_optim(self.__C, net, data_size)
        start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())

        grad_norm = np.zeros(len(named_params))

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            print("Training epoch...",  epoch)

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            time_start = time.time()
            print("time_start:" , time_start)

            # Iteration
            for x in range(len(dataset)):
                optim.zero_grad()
                img_feat_iter, ques_ix_iter, ans_ix_iter =  dataset.getpairmanual(x)

                img_feat_iter = img_feat_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                ans_ix_iter = ans_ix_iter.cuda()

                #print("ques_ix_iter: ", ques_ix_iter)
                #print("ans_ix_iter: ", ans_ix_iter)

                pad = torch.tensor([1]).cuda()
                ans_ix_iter = torch.cat((ans_ix_iter, pad), 0)
                ans_ix_iter = torch.cat((pad, ans_ix_iter), 0)
        

                pred = net(
                    img_feat_iter,
                    ques_ix_iter,
                    ans_ix_iter[:-2]
                )
                
                pred_np = pred.cpu().data.numpy()
                pred_argmax = np.argmax(pred_np, axis=1)


                outreas = []
                for p in pred_argmax:
                    outreas.append(dataset.i2w[p])
                #print("reason: ", outreas)

                
                loss = loss_fn(pred, ans_ix_iter[1:-1])
                loss /= self.__C.GRAD_ACCU_STEPS
                loss.backward()
                loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                mode_str = self.__C.SPLIT['train']
                print("\r[version %s][epoch %2d][%s] loss: %.4f, lr: %.2e" % (
                    self.__C.VERSION,
                    epoch + 1,
                    mode_str,
                    loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                    optim._rate
                ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS

                optim.step()
        

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base
            }
            torch.save(
                state,
                self.__C.CKPTS_PATH +
                'ckpt_' + self.__C.VERSION +
                '/epoch' + str(epoch_finish) +
                '.pkl'
            )

            print("Checkpoint saved.")
    
            loss_sum = 0
            grad_norm = np.zeros(len(named_params))

    """
    def run(self, run_mode):
        if run_mode == 'train':
            self.train(self.dataset)

        elif run_mode == 'val':
            self.eval(self.dataset, valid=True)

        elif run_mode == 'test':
            self.eval(self.dataset)

        elif run_mode == 'pred':
            self.loadModel(self.dataset)

        else:
            exit(-1)


    def loadModel(self, dataset):
        
        path = self.__C.CKPTS_PATH + 'ckpt_35137398/epoch2.pkl' 

        print("loaded path: ", path)
        # Obtain needed information
        token_size = dataset.token_size
        pretrained_emb = dataset.pretrained_emb
        pretrained_emb_ans = dataset.pretrained_emb_ans
        data_size = dataset.data_size

        fixed_ans_size = 16
       

        # Define the MCAN model
        net = Net(
            self.__C,
            pretrained_emb,
            pretrained_emb_ans,
            token_size,
            fixed_ans_size
        )
        net.cuda()
        # Load the network parameters
        print('Loading ckpt {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')
        net.load_state_dict(ckpt['state_dict'])

        print('Finished!')

        # Iteration
        for x in range(len(dataset)):
            img_feat_iter, ques_ix_iter, ans_ix_iter =  dataset.getpairmanual(x)

            img_feat_iter = img_feat_iter.cuda()
            ques_ix_iter = ques_ix_iter.cuda()
            ans_ix_iter = ans_ix_iter.cuda()

            pad = torch.tensor([1]).cuda()
            ans_ix_iter = torch.cat((ans_ix_iter, pad), 0)
            ans_ix_iter = torch.cat((pad, ans_ix_iter), 0)
    

            pred = net(
                img_feat_iter,
                ques_ix_iter,
                ans_ix_iter[:-2]
            )
            
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)

            print("dataset i2w: " , dataset.i2w)
            qa = []
            for q in ques_ix_iter:
                q =  q.item()
                qa.append(dataset.i2w[q])
            print("qa: ", qa)

            outreas = []
            for p in pred_argmax:
                outreas.append(dataset.i2w[p])
            print("reason: ", outreas)

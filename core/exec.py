from core.data.dataset import DataSet, TheLoader
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from vcr_util.pytorch_misc import time_batch
from torch.autograd import Variable

class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')
        self.dataset = DataSet(__C)

    def train(self, dataset):
       
        net = Net(
            self.__C,
        )

        net.cuda()
        net.train()

         #Create checkpoint
        if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
            shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)
        os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)
        loader_params = {'batch_size': 16, 'num_gpus':1}
        dataloader = TheLoader.from_dataset(dataset, **loader_params)
        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))
        
        loss_fn = torch.nn.NLLLoss().cuda()
        
        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')
            path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, len(dataloader), ckpt['lr_base'])
            optim._step = int(len(dataloader) / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH
        else:
            optim = get_optim(self.__C, net, len(dataloader))
            start_epoch = 0

       

        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            print("Training epoch...",  epoch)
            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)
            
            time_start = time.time()
            print("time_start:" , time_start)
            pred_argmax = []
            
            for b, (time_per_batch, batch) in enumerate(time_batch(dataloader)):
                optim.zero_grad()
                x, goldsentence = net(**batch)
                goldsentence = goldsentence[:, 1:]
                x = x[:,:31,:]
                pred_argmax = np.argmax(x.cpu().data.numpy(), axis=2)


                loss = loss_fn(x.permute(0,2,1), goldsentence)
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
        
            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


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

           
            print("Gold sentence: " , str(goldsentence.cpu().data))
            print("A sample prediction: ", pred_argmax )
            print("Checkpoint saved. " )


    
   
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

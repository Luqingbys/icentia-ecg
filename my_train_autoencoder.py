import numpy as np
import torch
import torch.optim as optim
import math
import glob
# import torch.nn as nn
# from model import Autoencoder
from myAE import Autoencoder
from mydata_io import *
from utils import get_logger
import sys
import random


report_every: int = 10
frame_length: int = 2**11 + 1


def data_stream(filenames, shuffle=True, batch_size=16):
    ''' 
    用于读取文件，返回可以拿来训练的数据用
    filenames: list，文件名列表，默认为'/icentia-ecg/datasets/'下的文件名列表
    shuffle: 是否打乱
    batch_size: 批量，默认为16
    '''
    stream = stream_file_list(
        filenames,
        buffer_count=20,
        batch_size=batch_size,
        chunk_size=1,
        shuffle=shuffle
    )
    stream = threaded(stream, queue_size=5)
    return stream


if __name__ == "__main__":
    logger = get_logger(filename='out/my_autoencoder/train.log')
    logger.info('====== Start training myautoencoder... ======')
    # directory = sys.argv[1]
    directory = 'datasets/'
    filenames = [ directory + "/%05d_batched.pkl.gz" % i
                  for i in range(21) ]
    # filenames = [ directory + "\\%05d_batched.pkl.gz" % i
    #               for i in range(21) ]
    # print(filenames)
    train_count = int(len(filenames) * 0.9)
    # 划分训练集和验证集
    train_filenames = filenames[:train_count]
    valid_filenames = filenames[train_count:]


    model = Autoencoder(0, 1)
    # valid_data = torch.from_numpy(signal_data_valid).cuda()[:, None, :]
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    # model = torch.load('model.pt')
    model = model.cuda()

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-3) # , weight_decay=1e-6)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=10, verbose=True, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08
    )

    epochs = 10

    best_loss = np.inf
    i = 0
    input = None

    # 开始训练
    for epoch in range(epochs):
        running_loss = 0.0
        time_step_count = 0
        for data in data_stream(train_filenames): # data: (16, 1, 1048578)
            # get the inputs
            input = torch.from_numpy(data.astype(np.float32)).cuda() # input: (16, 1, 1048578)
            # print('input: ', input.shape, input[:, :, -5:])
            # zero the parameter gradients
            # forward + backward + optimize，model(input)直接返回输出和原始输入之间的损失
            loss = model(input)
            # print(f'{epoch}, ', loss)

            if i % 4 == 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics
            total_samples = input.numel()
            running_loss += loss.detach().item() * total_samples
            time_step_count += total_samples

            i += 1
            if i % report_every == 0:    # print every 500 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch, i, running_loss / time_step_count))
                logger.info('Epoch:[{}/{}]\t current loss={:.5f}\t mini_batch average loss={:.3f}'.format(epoch , epochs, loss, running_loss/time_step_count))
                running_loss = 0.0
                time_step_count = 0

            if i % (report_every * 10) == 0:
            # 验证集
                model.eval()
                with torch.no_grad():
                    total_loss = 0.
                    count = 0
                    for data in data_stream(valid_filenames[:20], shuffle=False,
                                            batch_size=32):
                        # get the inputs
                        input = torch.from_numpy(data.astype(np.float32)).cuda()
                        loss = model(input)
                        # print(loss)
                        total_loss += loss.data.item()
                        count += 1
                    valid_loss = total_loss / count
                    if valid_loss < best_loss:
                        # print("Best valid loss:", valid_loss)
                        logger.info(f'Best valid loss:{valid_loss}')
                        with open('model.pt', 'wb') as f:
                            torch.save(model, f)
                        best_loss = valid_loss
                    else:
                        # print("Valid loss:", valid_loss)
                        logger.info(f'Valid loss: {valid_loss}')
                random.shuffle(valid_filenames)
                scheduler.step(valid_loss)
                model.train()
    # 保存模型文件
    with open('newAe.pt', 'wb') as f:
        torch.save(model, f)


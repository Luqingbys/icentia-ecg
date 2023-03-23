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
import os
import argparse

parser = argparse.ArgumentParser('train my autoencoder...')
parser.add_argument('--model_path', type=str, default='')
args = parser.parse_args()


report_every: int = 10
checkpoint_every: int = 10
frame_length: int = 2**11 + 1
batch_size: int = 2


def data_stream(filenames, shuffle=True, batch_size=batch_size):
    ''' 
    用于读取文件，返回可以拿来训练的数据用
    filenames: list，文件名列表，默认为'/icentia-ecg/datasets/'下的文件名列表
    shuffle: 是否打乱
    batch_size: 批量，默认为16
    '''
    print(batch_size)
    stream = stream_file_list(
        filenames,
        buffer_count=20,
        batch_size=batch_size,
        chunk_size=1,
        shuffle=shuffle
    )
    stream = threaded(stream, queue_size=5)
    return stream

# /mnt/ssd/ecg_big_data/icentia11k/


if __name__ == "__main__":
    logger = get_logger(filename='out/my_autoencoder/train.log')
    logger.info('====== Start training myautoencoder... ======')
    # directory = sys.argv[1]
    directory = 'datasets/'
    filenames = [directory + "/%05d_batched.pkl.gz" % i
                 for i in range(21) if os.path.exists(directory + "/%05d_batched.pkl.gz" % i)]
    print('filename length: ', len(filenames))

    # filenames = [ directory + "\\%05d_batched.pkl.gz" % i
    #               for i in range(21) ]
    # print(filenames)
    train_count = int(len(filenames) * 0.9)
    # 划分训练集和验证集
    train_filenames = filenames[:train_count]
    valid_filenames = filenames[train_count:]

    model = Autoencoder(0, 1)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # valid_data = torch.from_numpy(signal_data_valid).cuda()[:, None, :]
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    # model = torch.load('model.pt')
    model = model.to(device)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=1e-3)  # , weight_decay=1e-6)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=10, verbose=True, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08
    )

    epochs = 3
    start_epoch = 0

    model_path = 'out/my_autoencoder/' + args.model_path
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dic'])
        start_epoch = checkpoint['epoch']
        scheduler.last_epoch = start_epoch


    best_loss = np.inf
    i = 0
    input = None

    # 开始训练
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        time_step_count = 0
        step = 0
        for data in data_stream(train_filenames, shuffle=True, batch_size=batch_size):  # data: (16, 1, 1048578)
            step += 1
            # get the inputs
            # input: (16, 1, 1048578)
            input = torch.from_numpy(data.astype(np.float32)).to(device)
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
            if i % report_every == 0:    # print every 500 mini-batches，每处理batch_size*report_every份样本打印一次训练日志
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch, i, running_loss / time_step_count))
                logger.info('Epoch:[{}/{}], [{}/?]\t current loss={:.5f}\t mini_batch average loss={:.3f}'.format(
                    epoch, epochs, step*batch_size, loss, running_loss/time_step_count))
                running_loss = 0.0
                time_step_count = 0

            if i % checkpoint_every == 0:
                logger.info('checkpoint saving...')
                # if (epoch+1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dic": optimizer.state_dict(),
                              "loss": loss,
                              "epoch": epoch}
                path_checkpoint = "out/my_autoencoder/checkpint_{}_epoch_{}_.pkl".format(epoch, i)
                torch.save(checkpoint, path_checkpoint)

            if i % (report_every * 10) == 0:  # 每训练10*batch_size*report_every份样本，就在验证集上跑一次
                # 验证集
                model.eval()
                with torch.no_grad():
                    total_loss = 0.
                    count = 0
                    for data in data_stream(valid_filenames[:20], shuffle=False,
                                            batch_size=32):
                        # get the inputs
                        input = torch.from_numpy(
                            data.astype(np.float32)).to(device)
                        loss = model(input)
                        # print(loss)
                        total_loss += loss.data.item()
                        count += 1
                    valid_loss = total_loss / count
                    if valid_loss < best_loss:
                        # print("Best valid loss:", valid_loss)
                        logger.info(f'Best valid loss:{valid_loss}')
                        with open('mewAE.pt', 'wb') as f:
                            torch.save(model, f)
                        best_loss = valid_loss
                    else:
                        # print("Valid loss:", valid_loss)
                        logger.info(f'Valid loss: {valid_loss}')
                random.shuffle(valid_filenames)
                scheduler.step(valid_loss)
                model.train()

        # 单轮训练完之后，保存一次断点
        logger.info('checkpoint saving...')
        # if (epoch+1) % checkpoint_interval == 0:
        checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dic": optimizer.state_dict(),
                        "loss": loss,
                        "epoch": epoch}
        path_checkpoint = "out/my_autoencoder/checkpint_{}_epoch_{}_.pkl".format(epoch, i)
        torch.save(checkpoint, path_checkpoint)
    # 保存模型文件
    with open('newAE.pt', 'wb') as f:
        torch.save(model, f)

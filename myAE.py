import torch
from torch import nn
import torch.nn.functional as F


def receptive_field(op_params):
    ''' 
    感受野
    '''
    _, _, erfield, estride = op_params[0]
    for i in range(1, len(op_params)):
        _, _, kernel, stride = op_params[i]
        one_side = erfield // 2
        erfield = (kernel - 1) * estride + 1 + 2 * one_side
        estride = estride * stride
        if erfield % 2 == 0:
            print("EVEN", erfield)
        print(erfield, estride)
    return erfield, estride


class ResidualEncoder(torch.nn.Module):
    ''' 
    带有残差结构的编码器：本质上就是一个带有残差结构的卷积结构
    in_channels: 1
    out_channels: 100
    kernel_size: 2049
    stride: 2048
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=torch.nn.ELU(), dropout=0.1, last=False):
        super(ResidualEncoder, self).__init__()
        self.last = last

        self.conv_op = torch.nn.Conv1d(
            in_channels=in_channels, # 1
            out_channels=2 * out_channels, # 200
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=1, groups=1, bias=True
        )

        self.nin_op = torch.nn.Conv1d(
            in_channels=2 * out_channels, # 200
            out_channels=out_channels, # 100
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, groups=1, bias=True
        )
        self.res_op = torch.nn.Conv1d(
            in_channels=2 * out_channels, # 200
            out_channels=out_channels, # 100
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, groups=1, bias=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(2 * out_channels)


    def forward(self, x):
        z_ = self.bn(self.conv_op(x))
        # print('after conv1: ', z_.shape)
        z = self.dropout(self.activation(z_))
        y_ = self.nin_op(z)
        if not self.last:
            y = self.dropout(self.activation(y_))
            return y + self.res_op(z_)
        else:
            return y_


class ResidualDecoder(torch.nn.Module):
    ''' 
    带有残差结构的解码器：本质上就是带有残差结构的卷积结构
    (1, 100, 2049, 2048)
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=torch.nn.ELU(), dropout=0.5, last=False):
        ''' 
        in_channels: 100
        out_channels: 1
        kernel_size: 2049
        stride: 2048
        '''
        super(ResidualDecoder, self).__init__()
        self.last = last
        self.conv_op = torch.nn.ConvTranspose1d(
            in_channels=in_channels, # 100
            out_channels=out_channels * 2, # 1*2
            kernel_size=kernel_size, # 2049
            stride=stride, # 2048
            dilation=1, groups=1, bias=True
        )
        self.nonlin = torch.nn.Conv1d(
            in_channels=out_channels * 2, # 1*2
            out_channels=out_channels, # 1
            kernel_size=1,
            stride=1,
            dilation=1, groups=1, bias=True
        )
        self.res_op = torch.nn.Conv1d(
            in_channels=out_channels * 2, # 
            out_channels=out_channels, # 
            kernel_size=1,
            stride=1,
            dilation=1, groups=1, bias=True
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        self.bn = nn.BatchNorm1d(2 * out_channels)


    def forward(self, x):
        
        z_ = self.bn(self.conv_op(x))
        # print('after convTranspose1: ', z_.shape)
        z = self.dropout(self.activation(z_))
        y_ = self.nonlin(z)
        # print(y_.size(), z.size())
        if not self.last:
            y = self.dropout(self.activation(y_))
            return y + self.res_op(z_)
        else:
            return y_


class ConvAutoencoder(torch.nn.Module):
    ''' 
    卷积自编码器，包含编码器、解码器两部分
    '''
    def __init__(self, stack_spec, debug=True):
        ''' 
        stack_spec: list (1, 100, 2049, 2048)
        '''
        super(ConvAutoencoder, self).__init__()
        activation = torch.nn.ELU()
        encode_ops = []
        dropout = torch.nn.Dropout(0.5)

        # 添加多层编码器ResidualEncoder
        for i, (in_c, out_c, kernel, stride) in enumerate(stack_spec):
            # (1, 100, 2049,  2048)
            last = i == (len(stack_spec)-1) # 最后一个残差编码器块时，last==True
            encode_ops.append(ResidualEncoder(in_c, out_c, kernel, stride,
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass

        #    encode_ops.append(dropout)
        # encode_ops = encode_ops[:-1]

        self.encode = torch.nn.Sequential(*encode_ops)
        erfield, estride = receptive_field(stack_spec)
        if debug:
            print("Effective receptive field:", erfield, estride)
            self.test_conv = torch.nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=erfield,
                stride=estride,
                padding=0,
                dilation=1, groups=1, bias=True
            )

        # 添加多层解码器ResidualDecoder
        decode_ops = []
        for i, (out_c, in_c, kernel, stride) in enumerate(stack_spec[::-1]):
            # (1, 100, 2049, 2048)
            last = (i == len(stack_spec)-1)
            decode_ops.append(ResidualDecoder(in_c, out_c, kernel, stride,
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass


            #    decode_ops.append(dropout)
            # decode_ops = decode_ops[:-1]
        self.decode = torch.nn.Sequential(*decode_ops)
        self.activation = activation
        self.dropout = dropout
        self.debug = debug


    def forward(self, x):
        encoding = self.encode(x)
        output = self.decode(encoding)
        return output


class Autoencoder(torch.nn.Module):
    def __init__(self, mean=0, std=1, bottleneck_size=32, patient_num=11000):
        super(Autoencoder, self).__init__()
        self.mean = mean
        self.std = std
        activation = torch.nn.ELU()
        self.dropout = dropout = torch.nn.Dropout(0.5)
        self.patient_num = patient_num

        frame_dim = 100
        segment_dim = 256
        patient_dim = 256
        # Output should be [batch, *, 4089]
        # (  1,  16, 2049, 256),
#        self.autoencode_1 = ConvAutoencoder([
#                # in, out, kernel, stride
#                (  1, 512, 1025, 512),
#                (512, frame_dim,   3,   4),
#            ],
#        )
        self.autoencode_1 = ConvAutoencoder([
                # in, out, kernel, stride
                (1, frame_dim, 2049,  2048), # frame_dim: 100
            ],
        )
 
        self.encode_2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                frame_dim, 128, 5, 2,
                padding=0, dilation=1, groups=1, bias=True
            ),
            activation,
            torch.nn.MaxPool1d(3),
            dropout,
            torch.nn.Conv1d(
                128, 128, 5, 2,
                padding=0, dilation=1, groups=1, bias=True
            ),
            activation,
            torch.nn.MaxPool1d(3),
            dropout,
            torch.nn.Conv1d(
                128, segment_dim, 3, 1,
                padding=0, dilation=1, groups=1, bias=True
            ),
            activation,
        )

        self.encode_3_1 = torch.nn.Sequential(
            nn.Linear(segment_dim, patient_dim, 1),
            activation,
            dropout,
            nn.Linear(patient_dim, patient_dim, 1),
        )

        self.encode_3_2 = torch.nn.Sequential(
            nn.Linear(patient_dim * 2, patient_dim),
            activation,
            dropout
        )

        self.frame_transform = nn.Linear(frame_dim, frame_dim * 2)
        self.segment_transform = nn.Linear(segment_dim, frame_dim * 2)
        self.patient_transform = nn.Linear(patient_dim, frame_dim * 2)

        self.decode_transform = nn.Sequential(
            activation,
            nn.Linear(frame_dim * 2, frame_dim),
        )
        self.frame_bn = nn.BatchNorm1d(frame_dim)

        # self.linear_trans = nn.Linear(frame_dim+self.patient_num//frame_dim, frame_dim)
        # TODO: 改为卷积，将第三维大小622变为512，参与后续的解码
        self.merge_conv = nn.Conv1d(frame_dim, frame_dim, kernel_size=self.patient_num//frame_dim+1)


    def encode_3(self, x, input_shape):
        x = x.view(input_shape[0], input_shape[1], x.size(-1))
        h_1 = self.encode_3_1(x)
        h_2 = torch.cat([torch.max(h_1, dim=1)[0],
                         torch.min(h_1, dim=1)[0]], dim=-1)
        emb = self.encode_3_2(h_2)
        return emb


    def encode(self, input_flat):
        encoding_1 = self.frame_bn(self.autoencode_1.encode(input_flat))
        return encoding_1


    def decode(self, encoding):
        '''encoding: (16, 100, 512)'''
        output = self.autoencode_1.decode(encoding)
        return output


    def forward(self, input: torch.Tensor):
        ''' 
        自编码器，解码器部分添加病人的唯一标识，用独热编码表示
        input: (16, 1, 1048577)
        '''
        identity = torch.eye(self.patient_num)
        # print('====== AutoEncoder ======')
        # input: (16, 1, 1048577+1)
        input = input[:, :, :-1]
        patient_idx = input[:, :, -1].flatten().long() # (16,)

        input = (input - self.mean) / self.std
        input_flat = input.view(-1, 1, input.size(-1))

        encode = self.encode(input_flat) # encode: (16, 100, 512)

        # TODO: 拼接病人的唯一标识
        bz, c, l = encode.shape # 16, 100, 512
        extra = identity[patient_idx].reshape(bz, c, -1).cuda() # (16, 11000) => (16, 100, 110), 每一行均为独热，注意要将其转移到cuda，不然在gpu上训练时会报错

        # print('encode: ', encode.shape)
        # encode = encode.flatten(1) # (16, 51200)
        encode_h = torch.cat([encode, extra], dim=2) # encode_h: (16, 100, 622)
        encode_h = self.merge_conv(encode_h) # (16, 100, 622) => (16, 100, 512)
        # encode_h = self.linear_trans(encode_h) # encode_h: (16, 100, 512)
        # encode_h = encode_h.reshape(bz, c, l) # encode_h: (16, 100, 512)

        output = self.decode(encode_h) # output: (16, 1, 1048577)
        # print('decode: ', output.shape)

        output = output.view(input.size())
        # print('output: ', output.shape)
        
        # input为原始输入，output为自编码器的输出
        loss = torch.sqrt(torch.mean((output - input)**2))
        # loss = torch.mean(abs(output - input))
        return loss


# auto = Autoencoder()
# print(auto)
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class TinyObservationEncoder(nn.Module):
    def __init__(self, depth, stride, shape, activation):
        super().__init__()
        assert shape[1:] == (32, 32)

        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride), # (d, (32-4)/2+1, 15) 
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride), # (2d, (15-4)/2+1, 6) # 32
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride), # (4d, (6-4)/2+1, 2) # 64, 2, 2
            activation(),
        )
        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, obs): # (B, 1, 16, 16)
        embed = self.convolutions(obs) # (B, d, 2, 2)
        return embed

class ObservationEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, shape=(3, 64, 64), activation=nn.ReLU, tiny_factor=2):
        super().__init__()
        assert shape[1:] == (64, 64) and tiny_factor == 2 and depth==32
        self.tiny_shape = (shape[0], shape[1]//tiny_factor, shape[2]//tiny_factor)

        self.convolutions_0 = TinyObservationEncoder(depth=depth//tiny_factor, stride=stride, shape=self.tiny_shape, activation=activation)
        self.convolutions_1 = TinyObservationEncoder(depth=depth//tiny_factor, stride=stride, shape=self.tiny_shape, activation=activation)
        self.convolutions_2 = TinyObservationEncoder(depth=depth//tiny_factor, stride=stride, shape=self.tiny_shape, activation=activation)
        self.convolutions_3 = TinyObservationEncoder(depth=depth//tiny_factor, stride=stride, shape=self.tiny_shape, activation=activation)

        self.shape = shape
        self.stride = stride
        self.depth = depth

        print("Using Tiny Encoder-Decoder!")

    def forward(self, obs): # (1,1,64,64) (50, 50, 1, 64, 64)
        batch_shape = obs.shape[:-3] # (1) (50, 50)
        img_shape = obs.shape[-3:] # (1, 64, 64)
        obs = obs.reshape(-1, *img_shape) # (2500, 1, 64, 64)

        embed_0 = self.convolutions_0(obs[:, :, :32, :32])
        embed_1 = self.convolutions_1(obs[:, :, 32:, :32])
        embed_2 = self.convolutions_2(obs[:, :, :32, 32:])
        embed_3 = self.convolutions_3(obs[:, :, 32:, 32:]) # (B, 2d, 2, 2)

        embed = torch.cat([embed_0, embed_1, embed_2, embed_3], axis=1) # (B, 4*2d, 2, 2)
        embed = torch.reshape(embed, (*batch_shape, -1)) # (50, 50, 8d*4)
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.tiny_shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        embed_size = 4 * 2 * self.depth * np.prod(conv3_shape).item()
        return embed_size


class TinyObservationDecoder(nn.Module):
    def __init__(self, depth, stride, activation, embed_size, shape):
        super().__init__()
        assert shape[1:] == (32, 32)

        c, h, w = shape
        conv1_kernel_size = 6
        conv2_kernel_size = 6
        conv3_kernel_size = 5
        padding = 0
        conv1_shape = conv_out_shape((h, w), padding, conv1_kernel_size, stride) # (14, 14)
        conv1_pad = output_padding_shape((h, w), conv1_shape, padding, conv1_kernel_size, stride) # (0, 0)
        conv2_shape = conv_out_shape(conv1_shape, padding, conv2_kernel_size, stride) # (5, 5)
        conv2_pad = output_padding_shape(conv1_shape, conv2_shape, padding, conv2_kernel_size, stride) # (0, 0)
        conv3_shape = conv_out_shape(conv2_shape, padding, conv3_kernel_size, stride) # (1, 1)
        conv3_pad = output_padding_shape(conv2_shape, conv3_shape, padding, conv3_kernel_size, stride) # (0, 0)
        self.conv3_shape = conv3_shape
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16 * depth, 2 * depth, conv3_kernel_size, stride, output_padding=conv3_pad), # 128-->64  64,2,2-->32
            activation(),
            nn.ConvTranspose2d(2 * depth, 1 * depth, conv2_kernel_size, stride, output_padding=conv2_pad), # 64-->32  32-->16
            activation(),
            nn.ConvTranspose2d(1 * depth, shape[0], conv1_kernel_size, stride, output_padding=conv1_pad), # 32-->1  16-->1
        )

    def forward(self, x):
        """
        :param x: (squeezed_size, 1024//4, 1, 1) ; squeezed_size=50*50
        :return: x = (squeezed_size, 1, 64//2, 64//2)
        """
        x = self.decoder(x) # (2500, 1, 32, 32)
        return x

class ObservationDecoder(nn.Module):
    def __init__(self, depth=32, stride=2, activation=nn.ReLU, embed_size=1024, shape=(3, 64, 64), tiny_factor=2):
        super().__init__()
        assert shape[1:] == (64, 64) and tiny_factor == 2 and depth == 32
        self.tiny_shape = (shape[0], shape[1]//tiny_factor, shape[2]//tiny_factor)
        self.depth = depth
        self.shape = shape

        self.decoder_1 = TinyObservationDecoder(depth=depth//2, stride=stride, activation=activation, embed_size=embed_size//4, shape=self.tiny_shape)
        self.decoder_2 = TinyObservationDecoder(depth=depth//2, stride=stride, activation=activation, embed_size=embed_size//4, shape=self.tiny_shape)
        self.decoder_3 = TinyObservationDecoder(depth=depth//2, stride=stride, activation=activation, embed_size=embed_size//4, shape=self.tiny_shape)
        self.decoder_4 = TinyObservationDecoder(depth=depth//2, stride=stride, activation=activation, embed_size=embed_size//4, shape=self.tiny_shape)
        conv3_shape = self.decoder_1.conv3_shape

        self.conv_shape = (32 * depth, *conv3_shape) # (1024, 1, 1)
        self.linear = nn.Linear(embed_size, 32 * depth * np.prod(conv3_shape).item())

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1] # (50, 50)
        embed_size = x.shape[-1] # 230
        squeezed_size = np.prod(batch_shape).item() # 2500
        x = x.reshape(squeezed_size, embed_size) # (squeezed_size, 230)
        x = self.linear(x) # (squeezed_size, 1024)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape)) # (squeezed_size, 1024, 1, 1)

        x1 = self.decoder_1(x[:, :256])    # (squeezed_size, 1, :32, :32) 0.169s -->
        x2 = self.decoder_2(x[:, 256:512]) # (squeezed_size, 1, 32:, :32) in gpu: 0.18s --> 0.00065 (padding=1-->0)
        x3 = self.decoder_3(x[:, 512:768]) # (squeezed_size, 1, :32, 32:) 0.18s
        x4 = self.decoder_4(x[:, 768:])    # (squeezed_size, 1, 32:, 32:) 0.18s
        x1 = torch.cat([x1, x2], axis=2) # (squeezed_size, 1, 64, :32)
        x3 = torch.cat([x3, x4], axis=2) # (squeezed_size, 1, 64, 32:)
        x = torch.cat([x1, x3], axis=3) # (squeezed_size, 1, 64, 64)

        mean = torch.reshape(x, (*batch_shape, *self.shape)) # (50, 50, 1, 64, 64)
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.shape))
        return obs_dist


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))

import torch


def pseudo_wasserstein(first, second):
    '''
    Computes a Wasserstein-like distance between two distributions.
    '''
    # 随机选取batch内的一张图片， 验证做差的结果
    # print(first.shape)  # torch.Size([bn, 3, 256])
    # print(second.shape)  # torch.Size([bn, 3, 256])
    # __ = []
    # for bn in range(len(first)):
    #     if bn == 28:
    #         for che in range(3):
    #             b1_f = first[bn][che]
    #             b1_s = second[bn][che]
    #             _ = []
    #             for i in range(len(b1_f)):
    #                 _.append((b1_f[i] - b1_s[i]).numpy())
    #             __.append(_)
    #         break
    # a = first - second  # # print(a.shape)  # torch.Size([bn, 3, 256])
    # print(__[0])
    # print(a[0])
    # print(__[1])
    # print(a[1])
    # print(__[2])
    # print(a[2])
    # quit()
    a = first - second  # # print(a.shape)  # torch.Size([bn, 3, 256])
    b = torch.square(a)  # print(b.shape)  # torch.Size([bn, 3, 256])
    c = torch.sum(b, dim=2)  # print(c.shape)  # torch.Size([bn, 3])
    return c

import os
import math
import torch
import shutil


def formatters(batch_size, size, epochs=1):
    '''
    Returns three functions which return strings for pretty printing epoch, batch, and summary information respectively.
    '''

    epoch_digits, size_digits = int(math.log10(epochs)) + 1, int(math.log10(size)) + 1
    return lambda epoch: f'Epoch {str(epoch).rjust(epoch_digits, " ")} {"-" * (12 + 2 * size_digits - epoch_digits)}', \
           lambda loss, batch: f'Loss: {loss:.7f} ({str(batch * batch_size).zfill(size_digits)}/{str(size).zfill(size_digits)})', \
           lambda loss_sum: f'Average Loss: {loss_sum / size:.7f}'


# def save(model, file_path):
#     '''
#     Saves a Wasserstein_SqueezeNet to a file.
#     '''
#
#     torch.save({'args': model.init_args, 'kwargs': model.init_kwargs, 'model': model.state_dict()}, file_path)
#
#
# def load(file_path):
#     '''
#     Loads a Wasserstein_SqueezeNet from a file.
#     '''
#
#     contents = torch.load(file_path)
#     model = Wasserstein_SqueezeNet(*(contents['args']), **(contents['kwargs']))
#     model.load_state_dict(contents['model'])
#     return model


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, args, is_best, filename='checkpoint.pth'):
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    filename = os.path.join(args.save, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save, 'model_best.pth'))

import torch
from val import test
from pytorchtools import EarlyStopping
from loss import pseudo_wasserstein
from utils import formatters, adjust_learning_rate, save_checkpoint


def train(args, train_loader, validation_loader, model, optimizer, device, best_loss):
    '''
    Trains a given model with given data.  The parameters which produce the lowest validation loss are returned.
    '''
    epoch_formatter, batch_formatter, summary_formatter = formatters(train_loader.batch_size, len(train_loader.dataset),
                                                                     args.epochs)

    best_val_loss = test(validation_loader, model, device, summary_prefix='Validation')

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for args.start_epoch in range(args.start_epoch, args.epochs):
        print(epoch_formatter(args.start_epoch + 1))
        adjust_learning_rate(optimizer, args.start_epoch, args)
        model.train()
        cumulative_loss = 0.0

        for batch, (sources, references) in enumerate(train_loader):
            #  print(sources.shape)  torch.Size([bn, 224, 224, 3])
            #  print(references.shape)  torch.Size([bn, 3, 256])
            sources = torch.squeeze(sources)
            sources = torch.transpose(sources, dim0=1, dim1=3)

            references = torch.squeeze(references)
            sources, references = sources.to(device), references.to(device)

            predictions = model(sources)
            _loss = pseudo_wasserstein(predictions, references)  # torch.Size([bn, 3])
            losses = torch.mean(_loss, dim=1)  # torch.Size([bn])
            batch_loss = torch.mean(losses)

            optimizer.zero_grad()
            # print("=============更新之前===========")
            # for name, parms in model.named_parameters():
            #     print('-->name:', name)
            #     # print('-->para:', parms)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")
            batch_loss.backward()
            optimizer.step()
            # print("=============更新之后===========")
            # for name, parms in model.named_parameters():
            #     print('-->name:', name)
            #     # print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            # print(optimizer)
            # input("=====迭代结束=====")
            # quit()
            print(batch_formatter(batch_loss.item(), batch))
            cumulative_loss += torch.sum(losses).item()

        print(f'(Training) {summary_formatter(cumulative_loss)}\n')
        val_loss = test(validation_loader, model, device, summary_prefix='Validation')

        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early Stopping ! ! !")
        #     break

        is_best = val_loss < best_val_loss
        best_loss = min(val_loss, best_val_loss)
        save_checkpoint({
            'epoch': args.start_epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, args, is_best)

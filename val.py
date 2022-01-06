import torch
from utils import formatters
from loss import pseudo_wasserstein


def test(data_loader, model, device, summary_prefix='Testing', out_dir=None):
    '''
    Tests a given model with given data.  Returns the average loss across all given data.
    '''

    model.eval()
    _, batch_formatter, summary_formatter = formatters(data_loader.batch_size, len(data_loader.dataset))

    cumulative_loss = 0.0

    with torch.no_grad():
        for batch, (sources, references) in enumerate(data_loader):
            sources = torch.transpose(sources, dim0=1, dim1=3)
            sources, references = sources.to(device), references.to(device)
            predictions = model(sources)
            losses = torch.mean(pseudo_wasserstein(predictions, references), dim=1)  # torch.Size([40])
            batch_loss = torch.mean(losses)  # tensor(0.3428)

            print(batch_formatter(batch_loss.item(), batch))
            cumulative_loss += torch.sum(losses).item()

    #         if out_dir is not None:
    #             for idx in range(len(predictions)):
    #                 idx_out_dir = os.path.join(out_dir, str(batch * data_loader.batch_size + idx + 1))
    #
    #                 for ref_idx, reference in enumerate(predictions[idx]):
    #                     hist_save(reference, os.path.join(idx_out_dir, f'{ref_idx + 1}.npy'))
    #
    print(f'({summary_prefix}) {summary_formatter(cumulative_loss)}')
    return cumulative_loss / len(data_loader.dataset[0][0])

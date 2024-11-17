import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import REDNet10, REDNet20, REDNet30
from dataset import BDDDataset
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='REDNet10', help='REDNet10, REDNet20, REDNet30')
    parser.add_argument('--raw_images_dir', type=str, required=True)
    parser.add_argument('--comp_images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'REDNet10':
        model = REDNet10()
    elif opt.arch == 'REDNet20':
        model = REDNet20()
    elif opt.arch == 'REDNet30':
        model = REDNet30()

    # If there is an existing model, load it and train it
    if os.path.exists(os.path.join(opt.outputs_dir, f'{opt.arch}_best.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.outputs_dir, f'{opt.arch}_best.pth')))

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataset = BDDDataset(opt.raw_images_dir, opt.comp_images_dir, opt.patch_size)
    val_dataset = BDDDataset(opt.raw_images_dir, opt.comp_images_dir, opt.patch_size, train_flag=False)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    
    val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    
    best_val_loss = float('inf')

    # Initialize Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join('./', 'logs'))

    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        
        # Log training loss to Tensorboard
        writer.add_scalar('Loss/train', epoch_losses.avg, epoch)

        # Validation phase
        model.eval()
        val_losses = AverageMeter()
        
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                preds = model(inputs)
                loss = criterion(preds, labels)
                
                val_losses.update(loss.item(), len(inputs))

        print(f'Validation loss: {val_losses.avg:.6f}')

        # Log validation loss to Tensorboard
        writer.add_scalar('Loss/val', val_losses.avg, epoch)

        # Save model if validation loss improves
        if val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            torch.save(model.state_dict(), 
                      os.path.join(opt.outputs_dir, f'{opt.arch}_best.pth'))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))

writer.close()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler as lrsched
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from PFITRE_Net import PFITRE_net
from LossFunctions import VGGloss_Wratio
from torch.cuda.amp import GradScaler
from utils import ReconArtiDataset3D, loss_plot

def train_nn(
    config, ckpt_dir, train_gt_data_dir, train_arti_data_dir,
    val_gt_data_dir=None, val_arti_data_dir=None,
    pre_ckpt_dir=None, ckpt_pre_num=0
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = config["model"].to(device)
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    w_pix, w_vgg, w_identity = 1e7, 3e6, 0.9375

    if config.get("load_pretrain", False):
        checkpoint = torch.load(os.path.join(pre_ckpt_dir, f"ckpt_{ckpt_pre_num}.pth"))
        model.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['net_opt_state_dict'])

    os.makedirs(ckpt_dir, exist_ok=True)
    train_loss_log = open(os.path.join(ckpt_dir, "train_loss.dat"), 'w')
    valid_loss_log = open(os.path.join(ckpt_dir, "test_loss.dat"), 'w')

    trainset = ReconArtiDataset3D(
        image_dir=train_gt_data_dir,
        arti_dir=train_arti_data_dir,
        transform_img=True,
        max_data=config["dataset_size"]
    )

    if config.get("train_val_splitted", False):
        validset = ReconArtiDataset3D(
            image_dir=val_gt_data_dir,
            arti_dir=val_arti_data_dir,
            transform_img=True,
            max_data=config["dataset_size"]
        )
        trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
        valloader = DataLoader(validset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    else:
        split_idx = int(len(trainset) * 0.9)
        train_subset, val_subset = random_split(trainset, [split_idx, len(trainset) - split_idx])
        trainloader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
        valloader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)

    vgg = VGGloss_Wratio().to(device)

    for epoch in tqdm(range(config["num_epochs"] + 1)):
        model.train()
        running_loss = pix_loss = vgg_loss = identity_loss = 0.0

        for i, (gt_img, art_img) in enumerate(trainloader):
            art_img = art_img.to(device, memory_format=torch.channels_last)
            gt_img = gt_img.to(device, memory_format=torch.channels_last)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(art_img)
                tensor_size = outputs.numel()
                l1_loss = torch.div(nn.functional.l1_loss(outputs, gt_img), tensor_size)
                vggloss = vgg(outputs, gt_img)
                sec_output = model(outputs)
                id_loss = nn.functional.l1_loss(outputs, sec_output)

                loss = w_pix * l1_loss + w_vgg * vggloss + w_identity * id_loss

            scaler.scale(loss).backward()
            if (i + 1) % config["iters_to_accumulate"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            pix_loss += l1_loss.item()
            vgg_loss += vggloss.item()
            identity_loss += id_loss.item()

        train_stats = {
            'epoch': epoch,
            'total_loss_mean': running_loss / len(trainloader),
            'pix_loss_mean': pix_loss / len(trainloader),
            'vgg_loss_mean': vgg_loss / len(trainloader),
            'identity_loss_mean': identity_loss / len(trainloader),
        }
        train_loss_log.write(
            "{epoch:>7d} {total_loss_mean:10.7f} {pix_loss_mean:10.7f} "
            "{vgg_loss_mean:10.7f} {identity_loss_mean:10.7f}\n".format(**train_stats)
        )

        # Validation
        model.eval()
        val_running_loss = val_pix_loss = val_vgg_loss = val_identity_loss = 0.0
        with torch.no_grad():
            for val_gt_img, val_art_img in valloader:
                val_art_img = val_art_img.to(device, memory_format=torch.channels_last)
                val_gt_img = val_gt_img.to(device, memory_format=torch.channels_last)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    val_outputs = model(val_art_img)
                    tensor_size = val_outputs.numel()
                    l1_loss = torch.div(nn.functional.l1_loss(val_outputs, val_gt_img), tensor_size)
                    vggloss = vgg(val_outputs, val_gt_img)
                    sec_output = model(val_outputs)
                    id_loss = nn.functional.l1_loss(val_outputs, sec_output)
                    val_loss = w_pix * l1_loss + w_vgg * vggloss + w_identity * id_loss

                val_running_loss += val_loss.item()
                val_pix_loss += l1_loss.item()
                val_vgg_loss += vggloss.item()
                val_identity_loss += id_loss.item()

        val_stats = {
            'epoch': epoch,
            'total_loss_mean': val_running_loss / len(valloader),
            'pix_loss_mean': val_pix_loss / len(valloader),
            'vgg_loss_mean': val_vgg_loss / len(valloader),
            'identity_loss_mean': val_identity_loss / len(valloader),
        }
        valid_loss_log.write(
            "{epoch:>7d} {total_loss_mean:10.7f} {pix_loss_mean:10.7f} "
            "{vgg_loss_mean:10.7f} {identity_loss_mean:10.7f}\n".format(**val_stats)
        )

        # Save checkpoint
        if os.path.isfile("EXIT") or epoch % 5 == 0 or os.path.isfile("SAVE"):
            torch.save({
                'epoch': epoch,
                'net_state_dict': model.state_dict(),
                'net_opt_state_dict': optimizer.state_dict()
            }, os.path.join(ckpt_dir, f"ckpt_{epoch}.pth"))
            if os.path.isfile("EXIT"):
                train_loss_log.close()
                valid_loss_log.close()
                return
            if os.path.isfile("SAVE"):
                os.remove("SAVE")

    train_loss_log.close()
    valid_loss_log.close()
    torch.cuda.empty_cache()

def main(gpus_per_trial=1):
    ckpt_dir = "/path/to/checkpoints/"
    train_gt_data_dir = '/path/to/train_gt/'
    train_arti_data_dir = '/path/to/train_arti/'
    valid_gt_data_dir = None
    valid_arti_data_dir = None
    pre_ckpt_dir = "/path/to/pretrained/"
    ckpt_pre_num = 20

    config = {
        "lr": 0.0002,
        "batch_size": 64,
        "iters_to_accumulate": 1,
        "num_epochs": 100,
        "model": PFITRE_net(),
        "Decay": False,
        "dataset_size": None,
        "train_val_splitted": False,
        "load_pretrain": False,
    }

    train_nn(
        config,
        ckpt_dir=ckpt_dir,
        train_gt_data_dir=train_gt_data_dir,
        train_arti_data_dir=train_arti_data_dir,
        val_gt_data_dir=valid_gt_data_dir,
        val_arti_data_dir=valid_arti_data_dir,
        pre_ckpt_dir=pre_ckpt_dir,
        ckpt_pre_num=ckpt_pre_num
    )

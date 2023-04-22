import argparse
import builtins
import glob
import os
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from lion_pytorch import Lion
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TextDataset, EEDataset
from model import EntityEmbedding, TransformerRegressor

builtins.print = partial(print, flush=True)


def train(model, optimizer, train_loader, val_loader, args):
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch: {epoch}")
        args.epoch = epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, val_loader, args)
        val_loss, val_mape = val(model, val_loader, args)
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss} | Val MAPE: {val_mape} | Best Val MAPE: {args.best_val_mape}"
        )
        args.writer.add_scalar("Val/loss", val_loss, args.iter)
        args.writer.add_scalar("Val/MAPE", val_mape, args.iter)
        if val_mape < args.best_val_mape:
            args.best_val_mape = val_mape
            is_best = True
        else:
            is_best = False
        save_checkpoint(
            {
                "epoch": epoch,
                "iter": args.iter,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_mape": args.best_val_mape,
            },
            is_best,
            args,
        )


def save_checkpoint(state, is_best, args):
    torch.save(state, os.path.join(args.save_dir, f"epoch_{state['epoch']}.pth.tar"))
    last_epoch_path = os.path.join(args.save_dir, f"epoch_{state['epoch'] - 1}.pth.tar")

    try:
        os.remove(last_epoch_path)
    except OSError:
        pass

    if is_best:
        past_best = glob.glob(os.path.join(args.save_dir, "model_best_*.pth.tar"))
        past_best = sorted(past_best, key=lambda x: int("".join(filter(str.isdigit, x))))
        if len(past_best) >= 5:
            try:
                os.remove(past_best[0])
            except:
                pass
        torch.save(state, os.path.join(args.save_dir, f"model_best_epoch_{state['epoch']}.pth.tar"))


def train_one_epoch(model, optimizer, train_loader, val_loader, args):
    model.train()
    loss_fn = torch.nn.MSELoss()
    losses = []
    batch_start_time = time.time()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        B = len(y)
        x = x.to(args.device)
        y = y.to(args.device)
        output = model(x)
        # loss = loss_fn(output.squeeze(), y.to(args.device))
        loss = torch.mean(torch.abs(output - y) / (torch.abs(y) + 1e-8))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if i % args.log_every == 0:
            # print(f"[{i}] Loss: {loss.item()}")
            args.writer.add_scalar("Train/loss", loss.item(), args.iter)

        torch.cuda.empty_cache()
        # print(f"[{i}] Batch time: {time.time() - batch_start_time}")
        batch_start_time = time.time()
        args.iter += B

    return np.mean(losses)


def val(model, val_loader, args):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    losses = []
    mape = 0
    with torch.no_grad():
        total = 0
        for i, (x, y) in enumerate(tqdm(val_loader)):
            total += len(y)
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x).squeeze()
            loss = loss_fn(output, y)
            losses.append(loss.item())
            # output = torch.exp(output * val_loader.dataset.std + val_loader.dataset.mean)
            # y = torch.exp(y * val_loader.dataset.std + val_loader.dataset.mean)
            y = y.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            mape += np.sum(np.abs(y - output) / (np.abs(y) + 1e-8))
        print(output[:10], y[:10], total, mape / total)
    return np.mean(losses), mape / total


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    df = pd.read_csv("dataset/split_train.csv")
    vc = dict(df["PRODUCT_TYPE_ID"].value_counts())
    id_to_ind = {}
    default_ind = 0
    for k, v in vc.items():
        if v > 10:
            id_to_ind[k] = default_ind
            default_ind += 1
        else:
            id_to_ind[k] = default_ind
    train_set = EEDataset(
        path="dataset/split_train.csv", id_to_ind=id_to_ind, default_ind=default_ind, drop_big=True
    )
    val_set = EEDataset(path="dataset/split_val.csv", id_to_ind=id_to_ind, default_ind=default_ind)

    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, num_workers=args.num_workers, shuffle=False
    )
    print(f"Train loader size: {len(train_loader)}")
    print(f"Val loader size: {len(val_loader)}")

    model = EntityEmbedding(embedding_dim=32, num_embeddings=len(id_to_ind)).to(args.device)
    params = []
    for n, p in model.named_parameters():
        if "transformer" in n:
            params.append({"params": p, "lr": args.lr / 10})
        else:
            params.append({"params": p, "lr": args.lr})

    optimizer = Lion(params, lr=args.lr)

    args.save_dir = f"checkpoints/{args.run_name}"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"logs/{args.run_name}", exist_ok=True)
    args.writer = SummaryWriter(f"logs/{args.run_name}")

    if args.resume:
        print("Resuming from checkpoint")
        state = torch.load(args.resume)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        args.best_val_mape = state["best_val_mape"]
        args.start_epoch = state["epoch"] + 1
        args.iter = state["iter"]
    else:
        args.best_val_mape = np.inf
        args.start_epoch = 0
        args.iter = 0

    print("Checking gradients")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")
    train(model, optimizer, train_loader, val_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="ee_v0")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    # Model
    args = parser.parse_args()
    main(args)

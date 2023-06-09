import argparse
import builtins
import glob
import os
import time
from functools import partial
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import TextDataset, TextEEDataset
from model import TransformerEntityRegressor, TransformerEntityRegressorOld

builtins.print = partial(print, flush=True)


def train(model, optimizer, train_loader, val_loader, args):
    print("In train function")
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch: {epoch}")
        args.epoch = epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, val_loader, args)


def save_checkpoint(state, is_best, args):
    torch.save(state, os.path.join(args.save_dir, f"iter_{state['iter']}.pth.tar"))
    last_epoch_path = os.path.join(
        args.save_dir, f"iter_{state['iter'] - args.save_every*args.batch_size}.pth.tar"
    )

    # try:
    #     os.remove(last_epoch_path)
    # except OSError:
    #     pass

    # if is_best:
    #     past_best = glob.glob(os.path.join(args.save_dir, "model_best_*.pth.tar"))
    #     past_best = sorted(past_best, key=lambda x: int("".join(filter(str.isdigit, x))))
    #     if len(past_best) >= 5:
    #         try:
    #             os.remove(past_best[0])
    #         except:
    #             pass
    #     torch.save(state, os.path.join(args.save_dir, f"model_best_iter_{state['iter']}.pth.tar"))


def train_one_epoch(model, optimizer, train_loader, val_loader, args):
    model.train()
    loss_fn = torch.nn.MSELoss()
    losses = []
    batch_start_time = time.time()
    for i, (string, x, y) in enumerate(train_loader):
        B = len(y)
        output = model(string=string, type_id=x, device=args.device)
        y = y.to(args.device)
        loss = loss_fn(output.squeeze(), y.to(args.device))
        # loss = torch.mean(torch.abs(output - y) / (torch.abs(y) + 1e-8))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if i % args.val_every == 0:
            print("Validating...")
            val_time = time.time()
            train_loss = np.mean(losses)
            val_loss, val_mape = val(model, val_loader, args)
            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss} | Val MAPE: {val_mape} | Best Val MAPE: {args.best_val_mape}"
            )
            args.writer.add_scalar("Val/loss", val_loss, args.iter)
            args.writer.add_scalar("Val/MAPE", val_mape, args.iter)
            is_best = val_mape < args.best_val_mape
            if is_best:
                args.best_val_mape = val_mape
            print(f"Val time: {time.time() - val_time}")

        if i % args.save_every == 0:
            state = {
                "epoch": args.epoch,
                "iter": args.iter,
                "state_dict": model.state_dict(),
                "best_val_mape": args.best_val_mape,
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state, is_best, args)
            print("Saved model")

        if i % args.log_every == 0:
            print(f"[{i}] Loss: {loss.item()}")
            args.writer.add_scalar("Train/loss", loss.item(), args.iter)

        torch.cuda.empty_cache()
        print(f"[{i}] Batch time: {time.time() - batch_start_time}")
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
        for i, (string, x, y) in enumerate(tqdm(val_loader)):
            total += len(y)
            y = y.to(args.device)
            output = model(string, x, device=args.device).squeeze()
            loss = loss_fn(output, y)
            losses.append(loss.item())
            if args.transform:
                output = torch.exp(output * val_loader.dataset.std + val_loader.dataset.mean)
                y = torch.exp(y * val_loader.dataset.std + val_loader.dataset.mean)
            mape += torch.sum(torch.abs(output - y) / (torch.abs(y) + 1e-8))
            if i >= 100:
                break
    return np.mean(losses), mape.item() / total


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
    train_set = TextEEDataset(
        path="dataset/train.csv",
        id_to_ind=id_to_ind,
        default_ind=default_ind,
        transform=args.transform,
    )
    val_set = TextEEDataset(
        path="dataset/split_val.csv",
        id_to_ind=id_to_ind,
        default_ind=default_ind,
        transform=args.transform,
    )

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

    print(f"Using model: {args.model}")
    model = TransformerEntityRegressorOld(
        transformer=args.model, embedding_dim=32, num_embeddings=len(id_to_ind)
    ).to(args.device)
    params = []
    for n, p in model.named_parameters():
        if "transformer" in n:
            params.append({"params": p, "lr": args.lr / 10})
        else:
            params.append({"params": p, "lr": args.lr})

    optimizer = torch.optim.Adam(params, lr=args.lr, amsgrad=True)

    args.save_dir = f"checkpoints/{args.run_name}"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"logs/{args.run_name}", exist_ok=True)
    args.writer = SummaryWriter(f"logs/{args.run_name}")

    if args.resume:
        print("Resuming from checkpoint")
        state = torch.load(args.resume)
        model.load_state_dict(state["state_dict"])
        # optimizer.load_state_dict(state["optimizer"])
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
    print("Starting training")
    train(model, optimizer, train_loader, val_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="v0")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--transform", type=bool, default=False)

    # Model
    parser.add_argument("--model", type=str, default="bert-base-uncased")

    args = parser.parse_args()
    main(args)

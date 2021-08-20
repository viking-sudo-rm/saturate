"""Script for training transformers with potential architectural modifications."""

import tqdm
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import argparse
import logging
from rich.logging import RichHandler
from rich import print
import pickle
import os

from src.capture_attention import CaptureAttention
from src.schedulers import get_policy
from src.loss import sequence_cross_entropy_with_logits
from src.language_model import transformers, LanguageModel
from src.tokenizer import Tokenizer
from src.utils import get_text_tokens_mask
from src.metrics import get_norm, get_saturation
from src.kl_reg import KlSatReg
from src.reg_schedules import reg_schedules


DATA = os.getenv("DATA")
assert os.path.isdir(str(DATA)), f"Could not find data folder: {DATA}"
PATH = DATA
MODELS = os.getenv("MODELS")
assert os.path.isdir(str(MODELS)), f"Could not find models folder: {MODELS}"


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


optims = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dev_batch_size", type=int, default=50)
    # Wikitext-2: Largest sentence is 699 on train, 429 on test.
    # Penn: Largest sentence is 82 on train, 74 on test.
    parser.add_argument("--seq_len", type=int, default=700)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--trans", type=str, default="vaswani", choices=["vaswani"] + list(transformers.keys())
    )
    parser.add_argument("--fig_dir", type=str, default="figs/finetune-trans")
    parser.add_argument("--data_dir", type=str, default=f"{MODELS}/finetune-trans")
    parser.add_argument("--no_bias", action="store_true")
    parser.add_argument("--data", choices=["wikitext-2", "penn"], default="wikitext-2")
    parser.add_argument("--optim", choices=optims.keys(), default="adamw")
    parser.add_argument("--sched", choices=["constant_lr", "linear_lr", "sqrt_lr"], default="constant_lr")
    parser.add_argument("--stop_iteration", type=int, default=1000)  # End of constant LR warmup
    parser.add_argument("--batch_metrics", type=int, default=None)
    parser.add_argument("--add_eos", action="store_true", help="Add <eos> to train sentences.")
    parser.add_argument("--reg_schedule", choices=reg_schedules.keys(), default=None)
    return parser.parse_args()


@torch.no_grad()
def get_metrics(args, model, dev_tokens, dev_mask, reg=None, device="cuda:0"):
    all_attn_saturation = []
    all_agreement = []
    all_saturation = []
    all_loss = []
    # In this loop, we iterate over the full dev set, including the small bit at the end.
    for b in range(0, len(dev_tokens), args.dev_batch_size):
        dev_batch_tokens = dev_tokens[b : b + args.dev_batch_size].to(device)
        dev_batch_mask = dev_mask[b : b + args.dev_batch_size].to(device)
        with model.capture_attention() as attns:
            dev_encoding, dev_logits = model(dev_batch_tokens[:, :-1])
        dev_loss = sequence_cross_entropy_with_logits(
            dev_logits, dev_batch_tokens[:, 1:], dev_batch_mask[:, :-1], average=None
        )
        attn_saturation = reg(torch.cat([a for _, a in attns], dim=0)).item() if reg else 0.
        dev_preds = dev_logits.argmax(dim=-1)
        agreement = (dev_preds == dev_batch_tokens[:, 1:]).float() * dev_batch_mask[
            :, :-1
        ]
        saturation = get_saturation(
            dev_encoding * dev_batch_mask[:, :-1].unsqueeze(dim=-1),
            model,
            lambda: model(dev_batch_tokens[:, :-1])[0]
            * dev_batch_mask[:, :-1].unsqueeze(dim=-1),
        )
        all_loss.append(dev_loss.cpu())
        all_attn_saturation.append(attn_saturation)
        all_agreement.append(agreement.cpu())
        all_saturation.append(saturation.cpu())

    all_loss = torch.cat(all_loss, dim=0)
    all_attn_saturation = torch.cat(all_attn_saturation, dim=0)
    all_agreement = torch.cat(all_agreement, dim=0)
    all_saturation = torch.cat(all_saturation, dim=0)
    all_perps = torch.pow(2, all_loss)
    numel = dev_mask[:, :-1].sum()
    return {
        "acc1": (all_agreement.sum() / numel).item(),
        "norm": get_norm(model).item(),
        "loss": all_loss.mean().item(),
        "pplx": all_perps.mean().item(),
        "sat": (all_saturation.sum() / numel).item(),
        # Technically a mean of means, but that's probably okay.
        "attn_sat": all_attn_saturation.mean().item(),
    }


def train_model(
    args,
    model,
    train_tokens,
    train_mask,
    dev_tokens,
    dev_mask,
    optimizer,
    epochs=10,
    record_init=False,
    device="cuda:0",
    scheduler: str = None,
    max_iterations = None,
):
    reg = KlSatReg()
    reg_sched = reg_schedules[args.reg_schedule]
    batch_timeseries = defaultdict(list)
    timeseries = defaultdict(list)
    if record_init:
        metrics = get_metrics(args, model, dev_tokens, dev_mask, reg=reg, device=device)
        for name, value in metrics.items():
            timeseries[name].append(value)
        print(metrics)

    best_loss = float("inf")
    lr_adjuster = get_policy(scheduler)(optimizer, args, max_iterations=max_iterations)
    iteration = 0
    max_iterations = len(train_tokens) // args.batch_size * epochs
    for e in range(epochs):
        model.train()
        log.info(f"Starting epoch {e}...")
        perm = torch.randperm(len(train_tokens))
        train_tokens = train_tokens[perm, :]
        train_mask = train_mask[perm, :]

        for b in tqdm.trange(0, len(train_tokens) - args.batch_size, args.batch_size):
            cur_lr = lr_adjuster(e, iteration)
            if args.batch_metrics is not None and iteration % args.batch_metrics == 0:
                norm = get_norm(model).item()
                batch_timeseries["step"].append(iteration)
                batch_timeseries["norm"].append(norm)
                batch_timeseries["lr"].append(cur_lr)

            tqdm.tqdm.write(f"i={iteration}, lr={cur_lr}", end="\r")
            batch_tokens = train_tokens[b : b + args.batch_size].to(device)
            batch_mask = train_mask[b : b + args.batch_size].to(device)
            optimizer.zero_grad()
            with model.capture_attention() as attns:
                _, logits = model(batch_tokens[:, :-1])
            loss = sequence_cross_entropy_with_logits(
                logits, batch_tokens[:, 1:], batch_mask[:, :-1]
            )
            if args.reg_weight > 0:
                reg_weight = reg_sched(iteration, max_iterations)
                loss += reg_weight * reg(torch.cat([a for _, a in attns], dim=0))
            loss.backward()
            optimizer.step()
            iteration += 1

        model.eval()
        metrics = get_metrics(args, model, dev_tokens, dev_mask, reg=reg, device=device)
        for name, value in metrics.items():
            timeseries[name].append(value)
        print(metrics)

        # Save the model checkpoint if this is the best performance yet.
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            data_dir = os.path.join(args.data_dir, args.data)
            ckpt_path = os.path.join(data_dir, args.trans + ".pt")
            torch.save(model.state_dict(), ckpt_path)

    return timeseries, batch_timeseries


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    raw_train, train_tokens, train_mask = get_text_tokens_mask(f"{PATH}/{args.data}/train.txt", args.seq_len, tokenizer, name="train", log=log)
    _, dev_tokens, dev_mask = get_text_tokens_mask(f"{PATH}/{args.data}/valid.txt", args.seq_len, tokenizer, name="dev", log=log)
    # Maximum number of training steps, used for linearly decaying learning rate schedule.
    max_iterations = len(raw_train) // args.batch_size * args.epochs

    log.info("Constructing model...")
    model = LanguageModel(
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_vocab=tokenizer.d_vocab,
        seq_len=args.seq_len,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        encoder_type=args.trans,
        bias=not args.no_bias,
    )
    model = CaptureAttention(model)
    log.info("Model constructed :)")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    opt = optims[args.optim]

    # Train the model and collect metrics.
    log.info("Starting training :)")
    timeseries, batch_timeseries = train_model(
        args,
        model,
        train_tokens,
        train_mask,
        dev_tokens,
        dev_mask,
        opt(model.parameters(), lr=args.lr),
        epochs=args.epochs,
        record_init=True,
        scheduler=args.sched,
        max_iterations=max_iterations,
        device=device,
    )
    
    # Save all the raw data from this model run.
    dirname = f"{args.trans}-{args.optim}-{args.sched}"
    data_dir = os.path.join(args.data_dir, args.data, dirname)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, "timeseries.dat"), "wb") as fh:
        pickle.dump(timeseries, fh)
    with open(os.path.join(data_dir, "batch_timeseries.dat"), "wb") as fh:
        pickle.dump(batch_timeseries, fh)
    torch.save(model.state_dict(), os.path.join(data_dir, "model.pt"))

    # Generate figures for each metric over this training run.
    fig_dir = os.path.join(args.fig_dir, args.data, dirname)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    for metric, values in timeseries.items():
        plt.figure()
        plt.plot(values)
        plt.title(f"dev {metric} over training")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.savefig(f"{fig_dir}/{metric}.pdf")


if __name__ == "__main__":
    main(parse_args())

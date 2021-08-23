"""Script for training transformers with potential architectural modifications."""

from typing import Tuple
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

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import GPT2Config

from src.metrics import get_norm
from src.kl_reg import KlSatReg
from src.reg_schedules import reg_schedules
from src.utils import iterate_lines


DATA = os.getenv("DATA")
assert os.path.isdir(str(DATA)), f"Could not find data folder: {DATA}"
MODELS = os.getenv("MODELS")
assert os.path.isdir(str(MODELS)), f"Could not find models folder: {MODELS}"


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("rich")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Name of huggingface model."
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dev_batch_size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fig_dir", type=str, default="figs/finetune-trans")
    parser.add_argument("--data_dir", type=str, default=f"{MODELS}/finetune-trans")
    parser.add_argument("--data", choices=["wikitext-2", "penn"], default="wikitext-2")
    parser.add_argument("--batch_metrics", type=int, default=None)
    parser.add_argument("--reg_schedule", choices=reg_schedules.keys(), default=None)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--no_pretrain", action="store_true")
    return parser.parse_args()


def get_dirs(args) -> Tuple[str, str]:
    """Get the proper directories for storing data and figures."""
    dirname = args.model
    data_dir = os.path.join(args.data_dir, args.data, dirname)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    fig_dir = os.path.join(args.fig_dir, args.data, dirname)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    return data_dir, fig_dir


@torch.no_grad()
def get_metrics(args, model, dev, reg=None, device="cpu"):
    dev_tokens = dev["input_ids"]
    dev_mask = dev["attention_mask"]
    lm_losses, attn_losses = [], []
    # In this loop, we iterate over the full dev set, including the small bit at the end.
    # FIXME: Missing end of loop.
    for b in range(0, len(dev_tokens) - args.dev_batch_size, args.dev_batch_size):
        dev_batch_tokens = dev_tokens[b : b + args.dev_batch_size].to(device)
        dev_batch_mask = dev_mask[b : b + args.dev_batch_size].to(device)
        # Labels are shifted inside GPT2.
        lm_outputs = model(dev_batch_tokens, labels=dev_batch_tokens, attention_mask=dev_batch_mask)
        lm_loss, _, _, attns = lm_outputs.values()
        attn_loss = torch.mean(
            torch.stack([args.reg * reg(attn, dev_batch_mask) for attn in attns])
        )
        lm_losses.append(lm_loss.cpu())
        attn_losses.append(attn_loss.cpu())
    return {
        # "norm": get_norm(model.encoder).item(),  # Ignore embedding parameters.
        "lm_loss": torch.stack(lm_losses).mean().item(),
        "attn_loss": torch.stack(attn_losses).mean().item(),
    }


def train_model(
    args,
    model,
    train,
    dev,
    optimizer,
    scheduler,
    epochs=10,
    record_init=False,
    device="cpu",
    max_iterations=None,
):
    reg = KlSatReg()
    reg_sched = reg_schedules[args.reg_schedule]
    timeseries = defaultdict(list)
    batch_timeseries = defaultdict(list)
    if record_init:
        log.info("Computing initial metrics...")
        metrics = get_metrics(args, model, dev, reg=reg, device=device)
        for name, value in metrics.items():
            timeseries[name].append(value)
        log.info(metrics)

    best_loss = float("inf")
    iteration = 0
    n_train = len(train["input_ids"])
    max_iterations = n_train // args.batch_size * epochs
    for e in range(epochs):
        model.train()
        log.info(f"Starting epoch {e}...")
        perm = torch.randperm(n_train)
        train_tokens = train["input_ids"][perm, :]
        train_mask = train["attention_mask"][perm, :]

        for b in tqdm.trange(0, len(train_tokens) - args.batch_size, args.batch_size):
            if args.batch_metrics is not None and iteration % args.batch_metrics == 0:
                norm = get_norm(model).item()
                batch_timeseries["step"].append(iteration)
                batch_timeseries["norm"].append(norm)

            batch_tokens = train_tokens[b : b + args.batch_size].to(device)
            batch_mask = train_mask[b : b + args.batch_size].to(device)
            optimizer.zero_grad()
            lm_outputs = model(batch_tokens, labels=batch_tokens, attention_mask=batch_mask)
            loss, _, _, attns = lm_outputs.values()
            loss = loss.mean()  # FIXME: Why does the loss return two values here?
            reg_weight = reg_sched(iteration, max_iterations)
            if reg_weight != 0:
                # Mean of means is fine here as long as internal number stays constant.
                loss += reg_weight * torch.mean(
                    torch.stack([args.reg * reg(attn, batch_mask) for attn in attns])
                )
            loss.backward()
            optimizer.step()
            scheduler.step()
            iteration += 1

        model.eval()
        metrics = get_metrics(args, model, dev, reg=reg, device=device)
        for name, value in metrics.items():
            timeseries[name].append(value)
        print(metrics)

        # TODO: How do we do model selection here?
        if metrics["lm_loss"] < best_loss:
            best_loss = metrics["lm_loss"]
            data_dir, _ = get_dirs(args)
            torch.save(model.state_dict(), os.path.join(data_dir, "model.pt"))

    return timeseries, batch_timeseries


def main(args):
    assert args.model == "gpt2"
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    if not args.no_pretrain:
        model = GPT2LMHeadModel.from_pretrained(args.model, output_attentions=True)
    else:
        model = GPT2LMHeadModel(GPT2Config(output_attentions=True, return_dict=True))
    tokenizer.pad_token = tokenizer.eos_token
    train = tokenizer(
        list(iterate_lines(f"{DATA}/{args.data}/train.txt")),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    dev = tokenizer(
        list(iterate_lines(f"{DATA}/{args.data}/valid.txt")),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    n_train = len(train["input_ids"])
    max_iterations = n_train // args.batch_size * args.epochs

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=-1)

    # Train the model and collect metrics.
    log.info("Starting training...")
    timeseries, batch_timeseries = train_model(
        args,
        model,
        train,
        dev,
        optimizer,
        scheduler,
        epochs=args.epochs,
        record_init=True,
        max_iterations=max_iterations,
        device=device,
    )

    # Save all the raw data from this model run.
    data_dir, fig_dir = get_dirs(args)
    with open(os.path.join(data_dir, "timeseries.dat"), "wb") as fh:
        pickle.dump(timeseries, fh)
    with open(os.path.join(data_dir, "batch_timeseries.dat"), "wb") as fh:
        pickle.dump(batch_timeseries, fh)

    # Generate figures for each metric over this training run.
    for metric, values in timeseries.items():
        plt.figure()
        plt.plot(values)
        plt.title(f"dev {metric} over training")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.savefig(os.path.join(fig_dir, f"{metric}.pdf"))


if __name__ == "__main__":
    main(parse_args())

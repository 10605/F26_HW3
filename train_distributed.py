"""CPU data-parallel training on the HW2 binary classification dataset.

Complete the four distributed TODOs. Run the accompanying notebook to check
your implementation and produce the validation-loss and throughput plots.
"""

import argparse
import csv
import time
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


BATCH_SIZE = 512  # Per worker; the effective batch grows with worker count.
SEED = 10605
LEARNING_RATES = {1: 1e-3, 2: 1.4e-3, 4: 2e-3}


class MLP(nn.Module):
    """The same 80-feature architecture used in HW2."""

    def __init__(self, number_of_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(number_of_features, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def make_datasets(path: Path) -> tuple[TensorDataset, TensorDataset]:
    frame = pl.read_csv(path)

    def split(name: str) -> TensorDataset:
        rows = frame.filter(pl.col("split") == name)
        features = rows.drop("row_id", "split", "label").to_numpy().astype("float32")
        labels = rows["label"].to_numpy().astype("float32")
        return TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))

    return split("train"), split("validation")


@torch.inference_mode()
def validation_loss(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_rows = 0
    for features, labels in loader:
        total_loss += loss_fn(model(features), labels).item()
        total_rows += len(labels)
    return total_loss / total_rows


def initialize_distributed() -> tuple[int, int]:
    """TODO 1: Initialize a Gloo process group; return (rank, world_size).

    torchrun supplies the rendezvous environment variables.
    """
    raise NotImplementedError("TODO 1: initialize the process group")

def broadcast_parameters(model: nn.Module) -> None:
    """TODO 2: Copy each parameter from rank 0 to every other rank."""
    raise NotImplementedError("TODO 2: broadcast the initial parameters")


def average_gradients(model: nn.Module, world_size: int) -> None:
    """TODO 3: Pack all gradients; all-reduce once; write back their mean.

    Skip communication for one worker. See the handout for pseudocode.
    """
    raise NotImplementedError("TODO 3: average packed gradients")

def average_model(model: nn.Module, world_size: int) -> None:
    """TODO 4: Average all model weights. See the handout for pseudocode
    """
    raise NotImplementedError("TODO 4: average model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("features.csv"))
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for this training run")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--average-model", type = int, default = None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    rank, world_size = initialize_distributed()
    try:
        if world_size not in LEARNING_RATES:
            raise ValueError("Run with 1, 2, or 4 workers")
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        train_set, val_set = make_datasets(args.data)
        sampler = DistributedSampler(train_set, num_replicas = world_size, rank = rank)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE) if rank == 0 else None

        model = MLP(train_set.tensors[0].shape[1])
        broadcast_parameters(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATES[world_size], weight_decay=1e-4
        )
        loss_fn = nn.BCEWithLogitsLoss()

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        history = []
        training_seconds = 0.0
        examples_seen = 0

        acc = 0
        for epoch in range(1, args.epochs + 1):
            sampler.set_epoch(epoch)  # New, coordinated shuffle each epoch.
            model.train()
            dist.barrier()  # Rank 0's validation must finish before timing starts.
            start = time.perf_counter()
            local_rows = 0
            for features, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(features), labels)
                loss.backward()
                if args.average_model is None:
                    average_gradients(model, world_size)
                acc += 1
                optimizer.step()
                if args.average_model is not None and acc % args.average_model == 0:
                    average_model(model, world_size)
                local_rows += len(labels)

            # All workers report the time of the slowest worker.
            seconds = torch.tensor(time.perf_counter() - start, dtype=torch.float64)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            training_seconds += seconds.item()
            # DistributedSampler gives each worker the same number of rows.
            examples_seen += local_rows * world_size

            if rank == 0:
                loss = validation_loss(model, val_loader)
                history.append({"workers": world_size, "epoch": epoch,
                                "train_time_s": training_seconds,
                                "examples_seen": examples_seen, "val_loss": loss})
                print(f"workers={world_size} epoch={epoch:02d} "
                      f"train_s={training_seconds:.3f} val_loss={loss:.5f}", flush=True)

        if rank == 0:
            with (args.output / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

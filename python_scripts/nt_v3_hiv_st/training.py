import numpy as np
import torch
import pandas as pd
import os
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from huggingface_hub import login
token_path = '/workspaces/mpath/oanoufa/data/ibenstoken.txt'
with open(token_path, 'r') as f:
    token = f.read().strip()
login(token=token)

workspace_path = "/workspaces/mpath/oanoufa"
constants_path = f"{workspace_path}/python_scripts/utils/constants.py"
sys.path.append(os.path.dirname(constants_path))
import constants
import model_config

pure_st_to_id_dict = constants.ST_TO_ID_DICT
num_subtypes       = len(pure_st_to_id_dict)
config             = model_config.config
max_length         = config["sequence_length"]
pad_multiple_of    = config["pad_multiple_of"]

from hiv_dataset_class import HIVSequenceDataset, open_memmaps, visualize_sample
from hiv_nt_training_class import HFModelForHIVSubtyping, train_step, validation_step
from hiv_nt_metrics_class import HIVSubtypingMetrics


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)

    device = torch.device(config["device"])
    print(f"Using device: {device}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")

    model = HFModelForHIVSubtyping(model_name=config["model_name"], num_subtypes=num_subtypes)
    model = model.to(device)
    model.train()

    print(f"Model loaded: {config['model_name']}")
    print(f"Number of subtypes: {num_subtypes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Data loading — open memmaps (no RAM allocation for the full dataset)
    # ------------------------------------------------------------------
    seq_mm, lbl_mm, mask_mm = open_memmaps(config["sequences_path"], config["labels_path"], config["loss_masks_path"])
    metadata = pd.read_csv(config["metadata_path"], sep='\t')

    assert seq_mm.shape[0] == len(metadata), "Mismatch between sequences and metadata row counts"
    print(f"Sequences memmap : {seq_mm.shape}  dtype={seq_mm.dtype}")
    print(f"Labels memmap    : {lbl_mm.shape}  dtype={lbl_mm.dtype}")
    print(f"Loss masks memmap: {mask_mm.shape}  dtype={mask_mm.dtype}")

    # ------------------------------------------------------------------
    # Datasets — each split views its subset of rows via stored indices
    # ------------------------------------------------------------------
    train_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=num_subtypes,
        max_length=max_length, pad_multiple_of=pad_multiple_of, split="train",
    )
    val_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=num_subtypes,
        max_length=max_length, pad_multiple_of=pad_multiple_of, split="val",
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
    )

    print(f"\nTrain samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW([
        {"params": model.backbone.parameters(),
         "lr": config["learning_rate"] * config["backbone_learning_rate_multiplier"]},
        {"params": model.subtype_head.parameters(), "lr": config["learning_rate"]},
    ], weight_decay=config["weight_decay"])

    num_warmup_steps = int(config["warmup_proportion"] * config["num_steps_training"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config["num_steps_training"],
    )

    # ------------------------------------------------------------------
    # Optional: load from checkpoint to resume training
    # ------------------------------------------------------------------
    if config["load_checkpoint"]:
        checkpoint = torch.load(
            os.path.join(config["data_cache_dir"], config["checkpoint_name"]),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"],
                            strict=False)

    print(f"\nTraining configuration:")
    print(f"  Batch size              : {config['batch_size']}")
    print(f"  Total training steps    : {config['num_steps_training']}")
    print(f"  Learning rate           : {config['learning_rate']}")
    print(f"  Backbone LR multiplier  : {config['backbone_learning_rate_multiplier']}")
    print(f"  Weight decay            : {config['weight_decay']}")
    print(f"  Warmup steps            : {num_warmup_steps}")
    print(f"  Log every               : {config['log_every_n_steps']} steps")
    print(f"  Validate every          : {config['validate_every_n_steps']} steps")

    id_to_st   = {v: k for k, v in pure_st_to_id_dict.items()}
    n_token_id = tokenizer.encode("N", add_special_tokens=False)[0]

    train_metrics = HIVSubtypingMetrics(num_subtypes, "train",
        output_path=os.path.join(config["data_cache_dir"], f"train_metrics_v{config['model_version']}.tsv"),
        id_to_st=id_to_st, n_token_id=n_token_id)

    val_metrics   = HIVSubtypingMetrics(num_subtypes, "val",
        output_path=os.path.join(config["data_cache_dir"], f"val_metrics_v{config['model_version']}.tsv"),
        id_to_st=id_to_st, n_token_id=n_token_id)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training for {config['num_steps_training']} steps\n")

    train_iter  = iter(train_loader)
    best_val_f1 = 0.0
    model.train()

    for step_idx in tqdm(range(config["num_steps_training"]),
                         desc="Training steps...",
                         mininterval=300):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        train_step(model, optimizer, scheduler, batch, train_metrics)

        if (step_idx + 1) % (config["log_every_n_steps"] * config["num_steps_training"]) == 0:
            train_metrics.print_metrics()
            train_metrics.save_metrics(step=step_idx + 1)
            train_metrics.reset()

        if (step_idx + 1) % (config["validate_every_n_steps"] * config["num_steps_training"]) == 0:
            print(f"\nRunning validation at step {step_idx + 1}...")
            model.eval()

            for i, val_batch in enumerate(val_loader):
                validation_step(model, val_batch, val_metrics)
                if i >= config["max_val_batches"]:
                    break

            # If we are at the last validation step, compute and save final metrics (including confusion matrices)
            if (step_idx + 1) >= config["num_steps_training"]:
                val_metrics.print_detailed()
            else:
                val_metrics.print_metrics()
            val_result = val_metrics.compute()
            val_metrics.save_metrics(step=step_idx + 1)
            val_metrics.reset()

            if val_result["f1/micro"] > best_val_f1:
                best_val_f1 = val_result["f1/micro"]
                torch.save({
                    "step": step_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_f1": best_val_f1,
                }, os.path.join(config["data_cache_dir"], config["checkpoint_name"]))

            print("\n" + "-" * 50 + "\nTraining metrics:")
            model.train()

    print(f"\nTraining completed after {config['num_steps_training']} steps.")
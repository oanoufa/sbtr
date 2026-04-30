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

import config

from huggingface_hub import login
TOKEN_PATH = config.TOKEN_PATH
with open(TOKEN_PATH, 'r') as f:
    token = f.read().strip()
login(token=token)

WORKSPACE_PATH = config.WORKSPACE_PATH

ST_TO_ID_DICT = config.ST_TO_ID_DICT
NUM_SUBTYPES       = len(ST_TO_ID_DICT)
MODEL_CONFIG       = config.MODEL_CONFIG
MAX_LENGTH         = MODEL_CONFIG["sequence_length"]
PAD_MULTIPLE_OF    = MODEL_CONFIG["pad_multiple_of"]

from hiv_dataset_class import HIVSequenceDataset, open_memmaps
from hiv_nt_training_class import HFModelForHIVSubtyping, train_step, validation_step
from hiv_nt_metrics_class import HIVSubtypingMetrics


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"], trust_remote_code=True)

    device = torch.device(MODEL_CONFIG["device"])
    print(f"Using device: {device}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")

    model = HFModelForHIVSubtyping(model_name=MODEL_CONFIG["model_name"], num_subtypes=NUM_SUBTYPES)
    model = model.to(device)
    model.train()

    print(f"Model loaded: {MODEL_CONFIG['model_name']}")
    print(f"Number of subtypes: {NUM_SUBTYPES}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Data loading — open memmaps (no RAM allocation for the full dataset)
    # ------------------------------------------------------------------
    seq_mm, lbl_mm, mask_mm = open_memmaps(MODEL_CONFIG["sequences_path"], MODEL_CONFIG["labels_path"], MODEL_CONFIG["loss_masks_path"])
    metadata = pd.read_csv(MODEL_CONFIG["metadata_path"], sep='\t')

    assert seq_mm.shape[0] == len(metadata), "Mismatch between sequences and metadata row counts"
    print(f"Sequences memmap : {seq_mm.shape}  dtype={seq_mm.dtype}")
    print(f"Labels memmap    : {lbl_mm.shape}  dtype={lbl_mm.dtype}")
    print(f"Loss masks memmap: {mask_mm.shape}  dtype={mask_mm.dtype}")

    # ------------------------------------------------------------------
    # Datasets — each split views its subset of rows via stored indices
    # ------------------------------------------------------------------
    train_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, split="train",
    )
    val_dataset = HIVSequenceDataset(
        seq_mm=seq_mm, lbl_mm=lbl_mm, mask_mm=mask_mm, metadata=metadata,
        tokenizer=tokenizer, n_subtypes=NUM_SUBTYPES,
        max_length=MAX_LENGTH, pad_multiple_of=PAD_MULTIPLE_OF, split="val",
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=MODEL_CONFIG["batch_size"],
        shuffle=True, num_workers=MODEL_CONFIG["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=MODEL_CONFIG["batch_size"],
        shuffle=False, num_workers=MODEL_CONFIG["num_workers"],
    )

    print(f"\nTrain samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW([
        {"params": model.backbone.parameters(),
         "lr": MODEL_CONFIG["learning_rate"] * MODEL_CONFIG["backbone_learning_rate_multiplier"]},
        {"params": model.subtype_head.parameters(), "lr": MODEL_CONFIG["learning_rate"]},
    ], weight_decay=MODEL_CONFIG["weight_decay"])

    num_warmup_steps = int(MODEL_CONFIG["warmup_proportion"] * MODEL_CONFIG["num_steps_training"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=MODEL_CONFIG["num_steps_training"],
    )

    # ------------------------------------------------------------------
    # Optional: load from checkpoint to resume training
    # ------------------------------------------------------------------
    if MODEL_CONFIG["load_checkpoint"]:
        checkpoint = torch.load(
            os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["checkpoint_name"]),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"],
                            strict=False)

    print(f"\nTraining configuration:")
    print(f"  Batch size              : {MODEL_CONFIG['batch_size']}")
    print(f"  Total training steps    : {MODEL_CONFIG['num_steps_training']}")
    print(f"  Learning rate           : {MODEL_CONFIG['learning_rate']}")
    print(f"  Backbone LR multiplier  : {MODEL_CONFIG['backbone_learning_rate_multiplier']}")
    print(f"  Weight decay            : {MODEL_CONFIG['weight_decay']}")
    print(f"  Warmup steps            : {num_warmup_steps}")
    print(f"  Log every               : {MODEL_CONFIG['log_every_n_steps']} steps")
    print(f"  Validate every          : {MODEL_CONFIG['validate_every_n_steps']} steps")

    id_to_st   = {v: k for k, v in ST_TO_ID_DICT.items()}
    n_token_id = tokenizer.encode("N", add_special_tokens=False)[0]

    train_metrics = HIVSubtypingMetrics(NUM_SUBTYPES, "train",
        output_path=os.path.join(MODEL_CONFIG["metrics_dir"], f"train_metrics_v{MODEL_CONFIG['model_version']}.tsv"),
        id_to_st=id_to_st, n_token_id=n_token_id)

    val_metrics   = HIVSubtypingMetrics(NUM_SUBTYPES, "val",
        output_path=os.path.join(MODEL_CONFIG["metrics_dir"], f"val_metrics_v{MODEL_CONFIG['model_version']}.tsv"),
        id_to_st=id_to_st, n_token_id=n_token_id)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\nStarting training for {MODEL_CONFIG['num_steps_training']} steps\n")

    train_iter  = iter(train_loader)
    best_val_f1 = 0.0
    model.train()

    for step_idx in tqdm(range(MODEL_CONFIG["num_steps_training"]),
                         desc="Training steps...",
                         mininterval=300):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        train_step(model, optimizer, scheduler, batch, train_metrics)

        if (step_idx + 1) % (MODEL_CONFIG["log_every_n_steps"] * MODEL_CONFIG["num_steps_training"]) == 0:
            train_metrics.print_metrics()
            train_metrics.save_metrics(step=step_idx + 1)
            train_metrics.reset()

        if (step_idx + 1) % (MODEL_CONFIG["validate_every_n_steps"] * MODEL_CONFIG["num_steps_training"]) == 0:
            print(f"\nRunning validation at step {step_idx + 1}...")
            model.eval()

            for i, val_batch in enumerate(val_loader):
                validation_step(model, val_batch, val_metrics)
                if i >= MODEL_CONFIG["max_val_batches"]:
                    break

            # If we are at the last validation step, compute and save final metrics (including confusion matrices)
            if (step_idx + 1) >= MODEL_CONFIG["num_steps_training"]:
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
                }, os.path.join(MODEL_CONFIG["checkpoint_dir"], MODEL_CONFIG["checkpoint_name"]))

            print("\n" + "-" * 50 + "\nTraining metrics:")
            model.train()

    print(f"\nTraining completed after {MODEL_CONFIG['num_steps_training']} steps.")
#!/usr/bin/env python3
"""
AIDO.Cell Multi-GPU Embedding Script

This script uses PyTorch Distributed Data Parallel (DDP) to generate embeddings
across multiple GPUs. It is optimized for RTX 2080 (FP16 support).

Run this script using torchrun:
torchrun --nproc_per_node=NUM_GPUS embed_multi_gpu.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import anndata as ad
import numpy as np
from tqdm import tqdm

from aido_cell.models import CellFoundationModel, CellFoundationConfig
from aido_cell.utils import align_adata, preprocess_counts

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "genbio-ai/AIDO.Cell-3M"
INPUT_FILE = "/kaggle/input/singlecellperturbationdata/adamson/adamson/perturb_processed.h5ad"
BATCH_SIZE = 2  # Per GPU. Total batch = 2 * NUM_GPUS
EMBEDDING_KEY = "X_aido_cell"
# RTX 2080 does not support BF16 hardware acceleration. Use FP16.
DTYPE = torch.float16 

def setup_distributed():
    """Initializes the distributed process group."""
    if 'RANK' in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        print("Distributed environment not detected. Running in single-GPU mode.")
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, local_rank, world_size

class CellDataset(Dataset):
    """Simple dataset to handle AnnData indexing in a distributed manner."""
    def __init__(self, adata_aligned):
        self.data = adata_aligned.X
        self.is_sparse = hasattr(self.data, 'toarray')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        if self.is_sparse:
            row = row.toarray().squeeze(0)
        return row

def main():
    rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)

    if is_main_process:
        print(f"Starting distributed inference on {world_size} GPUs...")

    # Load and align data (Main process does initial heavy lifting)
    if is_main_process:
        print("Loading and aligning AnnData...")
    
    adata = ad.read_h5ad(INPUT_FILE)
    adata_aligned, attention_mask = align_adata(adata)
    
    dataset = CellDataset(adata_aligned)
    # Use DistributedSampler to shard the data rows across GPUs
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # Load Model
    config = CellFoundationConfig.from_pretrained(MODEL_NAME)
    model = CellFoundationModel.from_pretrained(MODEL_NAME, config=config)
    
    # Move to GPU and cast to FP16 for RTX 2080
    model = model.to(local_rank).to(DTYPE)
    model.eval()

    # Wrap in DDP (Note: For inference, DDP mainly helps with data sharding)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Convert attention mask to tensor and move to GPU
    # Add depth tokens (2) to the original mask length
    attn_mask_base = torch.from_numpy(attention_mask).to(local_rank).to(torch.bool)
    depth_mask = torch.ones(2, device=local_rank, dtype=torch.bool)
    full_attn_mask = torch.cat([attn_mask_base, depth_mask])

    local_embeddings = []
    
    # Inference loop
    with torch.no_grad():
        # Force SDPA (Flash/Mem-Efficient) kernels for the 19k sequence
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            for batch_counts in tqdm(dataloader, disable=not is_main_process, desc="GPUS Processing"):
                batch_counts = batch_counts.to(local_rank)
                
                # Preprocess counts (normalize, etc.)
                batch_processed = preprocess_counts(batch_counts, device=local_rank)
                
                # Expand mask for batch
                current_batch_size = batch_processed.shape[0]
                batch_mask = full_attn_mask.unsqueeze(0).expand(current_batch_size, -1)

                # Forward
                outputs = model(
                    input_ids=batch_processed,
                    attention_mask=batch_mask,
                    output_hidden_states=True
                )

                # Weighted Average Pooling
                last_hidden_state = outputs.last_hidden_state
                mask_expanded = batch_mask.unsqueeze(-1).to(DTYPE)
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.sum(mask_expanded, dim=1)
                batch_embeddings = sum_embeddings / sum_mask

                local_embeddings.append(batch_embeddings.cpu().numpy())

    # Gather all results from all GPUs
    local_embeddings = np.vstack(local_embeddings)
    
    # Prepare to collect on Rank 0
    if world_size > 1:
        # Each GPU might have a slightly different number of samples if not perfectly divisible
        all_gathered = [None] * world_size
        dist.all_gather_object(all_gathered, local_embeddings)
        
        if is_main_process:
            final_embeddings = np.vstack(all_gathered)
            # Match the original AnnData order (DistributedSampler can reorder)
            # We sort back to original indices if necessary, but shuffle=False usually preserves
            print(f"Successfully gathered {final_embeddings.shape[0]} embeddings.")
    else:
        final_embeddings = local_embeddings

    # Save on main process
    if is_main_process:
        # Note: If world_size > 1 and n_samples wasn't perfectly divisible, 
        # DistributedSampler adds padding. We trim to original size.
        final_embeddings = final_embeddings[:len(adata)]
        
        adata.obsm[EMBEDDING_KEY] = final_embeddings
        output_path = f"{os.path.splitext(INPUT_FILE)[0]}_embeddings.h5ad"
        adata.write_h5ad(output_path)
        print(f"✓ Process Complete. Saved to {output_path}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
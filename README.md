## Architecture Overview

The model follows a hierarchical Transformer design:

- Input image is divided into patches
- Local self-attention is applied within non-overlapping windows
- Shifted windows enable cross-window interaction
- Patch Merging reduces spatial resolution while increasing channel depth

This design significantly reduces computational complexity compared to global attention.

## Key Challenge

One of the most critical parts of the implementation was the **shifted window attention mechanism**.

Without attention masking, shifted windows cause incorrect interactions between unrelated patches.

To solve this, an attention mask was implemented to restrict attention only within valid regions.

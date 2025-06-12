#  Low-Rank Adaptation (LoRA)

##  Motivation

Large pre-trained language models like GPT, BERT, etc., contain billions of parameters and require massive compute and memory for fine-tuning on downstream tasks. Traditional full fine-tuning involves updating all parameters, which is inefficient, memory-intensive, and impractical for resource-constrained setups.

**LoRA (Low-Rank Adaptation)** provides a simple yet effective solution — instead of updating full-rank weight matrices during adaptation, it introduces a low-rank trainable update. This allows the majority of model parameters to remain frozen, significantly reducing training cost while achieving comparable performance.

---

##  Core Idea

LoRA assumes that the required updates to a pre-trained model’s weights during fine-tuning lie in a low-dimensional subspace. Hence, instead of full-rank updates, LoRA adds a **low-rank decomposition** to the weight matrices.

###  Mathematical Formulation

Let `W₀ ∈ ℝ^(d×k)` be a pre-trained weight matrix. LoRA modifies it as follows:


Where:
- `B ∈ ℝ^(d×r)`
- `A ∈ ℝ^(r×k)`
- `r ≪ min(d, k)` (i.e., r is much smaller than d and k)

During training:
- `W₀` remains frozen (no gradients)
- Only `A` and `B` are trainable
- `ΔW = B × A` is initialized such that `B = 0` and `A` is random (ΔW starts as 0)
- The final forward pass becomes:
  

Optionally, this is scaled by a factor `α / r` to control update magnitude.

---

##  Application to Transformers

In Transformer architectures, LoRA is applied to the weight matrices inside self-attention modules:

- `Wq`, `Wk`, `Wv`, `Wo`

In most setups, LoRA is used only on the `Wq` and `Wv` matrices for simplicity and efficiency. The MLP blocks and biases remain frozen.

---

##  Benefits of LoRA

-  **Memory Efficient**: Reduces GPU VRAM usage drastically. (e.g., GPT-3 175B fine-tuning VRAM drops from 1.2TB to 350GB)
-  **Smaller Checkpoints**: Checkpoint size reduced by 10,000× (e.g., from 350GB to 35MB with r = 4)
-  **Faster Training**: 25% speedup in training due to fewer parameters being updated
-  **Flexible Deployment**: Easily switch between tasks by swapping LoRA weights (A, B), while the base model remains constant
-  **No Additional Inference Latency**: Can merge `W₀ + ΔW` after training for efficient inference

---

##  Limitations

-  Not ideal for batching multiple tasks with different LoRA modules in the same forward pass
-  If weights are merged (for zero-latency inference), task switching requires explicit unmerging and re-merging

---

## 📚 Conclusion

LoRA offers a highly scalable and efficient way to fine-tune large models with minimal compute and memory. It enables multi-task adaptation, rapid deployment, and reduced storage requirements — making it a practical choice for fine-tuning massive language models.


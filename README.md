# Form Follows Function: Recursive Stem Model

paper: https://arxiv.org/abs/2603.15641


This is the codebase for the paper: "Form Follows Function: Recursive Stem Model". RSM is a recursive reasoning approach that achieves amazing scores 97.5% sudoku extreme and 80% on Maze Hard within 1 hour of training by recursively growing a tiny 2.5-5M parameters neural network.

<p align="center">
  <img src="https://github.com/navidivan/rsm/blob/main/assets/sudoku_strict_18.gif" alt="Sudoku">
</p>

[Paper](https://arxiv.org/abs/2510.04871)


<p align="center">
  <img src="https://github.com/navidivan/rsm/blob/main/assets/RSM_fig.png" alt="RSM">
</p>

### Motivation

Recursive Stem Model (RSM) is a recursive reasoning model that achieves amazing scores of 97.5% on sudoku extreme and 80% on Maze Hard with a tiny 2.5-5 M parameters neural network. I show that by only using 1 step loss (Truncated BackPropagation) and growing the outer depth (H) and inner depth (L) independently, a recursive model can simulate a fixed point solver similar to neural ODE solvers (or a denoisig process if that's your thing), where at test time, the hierarchical depth can be increased to orders of magnitude beyond training depth. Training at a shallow depth and solving stability of gradients problems allows us to efficiently train the networks by cutting training time by more than 20x while achieving much higher accuracy (cutting error rate by 5x).

This work came to be after I learned about the recent innovative Tiny Recursive Model (TRM) and Hierarchical Recursive Model (HRM). This paper opens a new avenue for efficient training of recursive networks and enabling unlimited test time compute to be thrown at a problem. Follow up work will investigate (stay tuned!)

<p align="center">
  <img src="https://github.com/navidivan/rsm/blob/main/assets/SRM%20cells.png" alt="Sudoku">
</p>

I also offer some interesting discussion/rant about what this is and what it does. Read the paper! In the rant, I talk about developemental biology and annealing ;)




<p align="center">
  <img src="https://github.com/navidivan/rsm/blob/main/assets/maze_reasoning_idx_5.gif" alt="Maze">
</p>

### How RSM works

RSM uses TRM's backbone, but enables efficient training (20x improvement) and higher accuracy (5x improvement in error rate). Instead of unrolling the full computational graph—which is memory-expensive and prone to vanishing gradients—or by doing Truncated BackPropagation Through Time (Deep Supervision)—which not only slows down training but forces greedy, immature behaviour—RSM completely detaches the history of hidden states during training. This forces the network to learn a stable, iterative transition function (acting as a fixed-point operator) rather than memorizing a specific depth-dependent path. By calculating loss only at the final step and treating intermediate steps as detached "warm-up" iterations, RSM compels the model to refine its state incrementally toward a solution and only use intermidiate calculations (z_h) as a scratch pad to free up computation. This architecture yields two critical benefits:

* **Training Efficiency:** We can train with shallow recurrence (total depth starts at $\approx 2$ and goes up to $\sim 20$, instead of the fixed $16 \times 4$ depth of TRM).

* **Test-Time Scaling:** Because the model learns a generalized state update rule at variable depths, it can run for arbitrary durations at inference ($H_{test} \gg 10,000s$), allowing it to "think" longer on harder puzzles without retraining.

* **Knows When It's Wrong!:** Hellucination, or confidence sounding slop, is a seriuos problem in modern models. By simulating a fixed point process, the RSM only stops when it's reached a viable solution! This gives an uprecedendet verfiable process that can be used as an Oracle. The verification process is extremely simple: if RSM has not settled down into a fixed point, then it's wrong. if it has, it's right.

<p align="center">
  <img src="https://github.com/navidivan/rsm/blob/main/assets/smart_heatmap.png" alt="Maze">
</p>


### Requirements

Installation should take a few minutes. The scripts were tested on Colab with A100 gpu defualt setting.

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
!pip install -q hydra-core wandb pydantic ninja argdantic coolname huggingface_hub adam-atan2 einops
```

### Dataset Preparation

```bash
# Sudoku-Extreme
!python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-all 

# Maze-Hard
!python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Sudoku-Extreme (assuming 1 A100 GPU):

```bash
# Define run name. prob_detach_prev_H 0 will always train the last 2 steps
run_name="Sudoku_Curriculum_Run"

!python pretrain.py \
arch=rsm \
data_paths="[data/sudoku-extreme-1k-aug-all]" \
evaluators="[]" \
epochs=2 \
eval_interval=7484 \
lr=1e-3 puzzle_emb_lr=1e-4 weight_decay=.1 puzzle_emb_weight_decay=1 global_batch_size=512 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=1 \
arch.H_cycles=2 arch.L_cycles=2 \
+arch.prob_detach_prev_H=0.99 \
+project_name="Sudoku_Extreme" \
+curriculum_milestones="[20.1, 40.1, 55.1, 65.1, 75.1, 85.1]" \
+curriculum_heads_to_add="[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]" \
+curriculum_L_milestones="[30.1, 50.1, 70.1, 80.1, 84.1, 86.1, 88.1, 90.1, 92.1, 94.1]" \
+curriculum_L_to_add="[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]" \
+arch.freeze_emb_prc=100.0 \
+grad_clip_norm=1.0 \
+local_warmup_steps=100 \
+optimizer_reset_scale=1 \
+arch.alpha_blend=False \
+save_checkpoints_at_percent="[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 99.9]" \
+short_eval_interval=500 \
+short_eval_fraction=0.005 \
+run_name=${run_name} ema=True
```

Expected: Around 80% exact-accuracy, (97% with test time compute)

*Runtime:* < 1 hour

### Maze-Hard (assuming 1 A100 GPUs):

```bash
# Define run name. prob_detach_prev_H 0 will always train the last 2 steps
run_name="Maze_Curriculum_Run"

!python pretrain.py \
arch=rsm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=500 \
eval_interval=7484 \
arch.mlp_t=False arch.pos_encodings=rope \
lr=1e-3 puzzle_emb_lr=1e-4 weight_decay=1 puzzle_emb_weight_decay=1 global_batch_size=128 arch.hidden_size=512 \
arch.L_layers=2 \
arch.H_cycles=2 arch.L_cycles=2 \
+arch.prob_detach_prev_H=90 \
+project_name="Maze_Hard" \
+curriculum_milestones="[20.1, 40.1, 55.1, 65.1, 75.1, 85.1, 95.1]" \
+curriculum_heads_to_add="[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]" \
+curriculum_L_milestones="[30.1, 50.1, 70.1]" \
+curriculum_L_to_add="[1, 1, 1]" \
+arch.freeze_emb_prc=100.0 \
+grad_clip_norm=1.0 \
+local_warmup_steps=200 \
+optimizer_reset_scale=1 \
+arch.alpha_blend=False \
+transition_lr_warmup_steps=200 \
+save_checkpoints_at_percent="[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 99.9]" \
+short_eval_interval=200 \
+short_eval_fraction=1 \
+run_name=${run_name} ema=True
```

*Runtime:* ~ 40 mins , exact accuracy  ~80%



## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{hakimi2026formfollowsfunctionrecursive,
      title={Form Follows Function: Recursive Stem Model}, 
      author={Navid Hakimi},
      year={2026},
      eprint={2603.15641},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.15641}, 
}
```


```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on Tiny Recursive Model [code] (https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

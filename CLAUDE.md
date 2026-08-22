# AirfRANS GNN Surrogate

Portfolio project: a MeshGraphNets-style GNN surrogate trained on the AirfRANS dataset
(RANS CFD around 2D airfoils) that predicts pressure/velocity fields directly from mesh
geometry, then integrates surface quantities to get lift and drag. Framed for interviews —
README narrative matters as much as raw accuracy.

## Stack

- v1: Gmsh (meshing) + FEniCSx (reference solver context) + PyTorch Geometric (model)
- PyVista for 3D/mesh visualization, Lightning for the training loop, Gradio for the demo
- v2 (later): port the trained model to NVIDIA PhysicsNeMo — document what changed and
  what it bought, as its own README section

Out of scope for now: PINNs beyond one toy exercise, transient simulation, multi-GPU training.

## Layout

- `src/` — all real logic (data loading, graph construction, model, training, metrics).
  Agents diff `.py` files well and `.ipynb` files poorly, so nothing important should
  live only in a notebook.
- `notebooks/` — exploration and plotting only. Keep cells under ~10 lines; call into
  `src/` rather than redefining logic inline.
- `configs/` — hyperparameters and run configs.
- `data/` — downloaded datasets, gitignored.

## Conventions

- Always report error as **relative L2 per field** (pressure, velocity components
  separately), never a single averaged number across fields.
- Don't tune architecture or hyperparameters before the data loader is solid and the
  model has been shown to overfit deliberately on a handful of cases.
- Local machine has a 4GB GPU (GTX 1650) — fine for Claude Code, exploration, and small
  smoke tests; real training runs happen on Colab. Keep local runs small (few cases,
  small batch) by default.

## Workflow

- Claude Code (this agent) handles multi-file/agentic work: the mesh-to-graph converter,
  training loop, evaluation metrics (especially surface integration for lift/drag), and
  the Gradio demo.
- GitHub Copilot (if active) handles inline autocomplete/boilerplate in the editor.
  Don't run both against the same file at the same time.
- Judging whether a resulting drag/lift error is physically plausible is the user's call,
  not something to automate away. 

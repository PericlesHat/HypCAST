# HypCAST

Official implementation of HypCAST for type 2 diabetes complication prediction and subtyping.

## Requirements

Core dependencies:
- Python ≥ 3.8
- PyTorch
- PyTorch Geometric
- scikit-learn
- numpy
- tqdm


```bash
pip install torch torch-geometric numpy scikit-learn tqdm
````

---

## Data Format

### Raw data directory

The script expects the following directory structure:

```text
data/
├── raw_data/
│   └── <dataset_name>/
│       ├── node-embeddings-<dataset_name>
│       └── edge-labels-<dataset_name>.txt
        └── hyperedges-<dataset_name>.txt
src/

```

Organize your hypergraph adjacency list into `hyperedges-<dataset_name>.txt` and your hyperedge (patient) labels into `edge-labels-<dataset_name>.txt`. Use DeepWalk algorithm (or other initializations) to generate initial embedding for each nodes into `node-embeddings-<dataset_name>`.


## Training

### Basic command

```bash
python train_hypcast.py \
  --dname demo \
  --epochs 100 \
  --num_cluster 5 \
  --lr 1e-3
```

### Important arguments

| Argument        | Description                  |
| --------------- | ---------------------------- |
| `--dname`       | Dataset name                     |
| `--epochs`      | Number of training epochs        |
| `--warmup`      | Warm-up epochs before clustering |
| `--num_cluster` | Number of hyperedge clusters |
| `--lr`          | Learning rate                  |
| `--wd`          | Weight decay                 |
| `--alpha`       | Weight for clustering loss   |
| `--threshold`   | Classification threshold     |
| `--cuda`        | GPU id (`-1` for CPU)        |

---

## Outputs

After training, the following files are saved under:

```text
logs/<timestamp>/
```

### Metrics

* `<dataset>_valid_<method>.txt`
* `<dataset>_test_<method>.txt`

### Predictions

* `hyg_prob.csv` — test set prediction probabilities
* `hyg_test_gt.csv` — test set ground-truth labels

### Clustering Results

* `edge_feat.npy` — hyperedge embeddings
* `edge_Q.npy` — soft cluster assignments
* `edge_label.npy` — hard cluster labels

### Configuration

* `hyperparameters.txt` — full training configuration




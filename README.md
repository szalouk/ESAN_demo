# ESAN demo
This repo contains the code to replicate the experiments of the [Equivariant Subgraph Aggregation Networks](https://arxiv.org/pdf/2110.02910.pdf) paper.


## Usage

### Example command 

  ```bash
  python main.py --gnn_type "graphconv" --num_layer 3 --emb_dim 64 --batch_size 32 --dataset "MUTAG" --jk "last" --drop_ratio 0.0 --channels "64-64" --policy "ego_plus" --model "dss"
  ````

## Arguments

```
main.py [-h] [--device DEVICE] [--model MODEL] [--gnn_type GNN_TYPE] [--drop_ratio DROP_RATIO] [--num_layer NUM_LAYER] [--channels CHANNELS] [--emb_dim EMB_DIM] [--jk JK] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--decay_rate DECAY_RATE] [--decay_step DECAY_STEP] [--epochs EPOCHS] [--num_workers NUM_WORKERS] [--dataset DATASET] [--policy POLICY] [--num_hops NUM_HOPS]
               [--seed SEED] [--patience PATIENCE] [--test] [--filename FILENAME]

Options:
  -h, --help            show this help message and exit
  --device DEVICE       which gpu to use if any (default: 0)
  --model MODEL         Type of model {deepsets, dss}
  --gnn_type GNN_TYPE   Type of convolution {gin, originalgin, zincgin, graphconv}
  --drop_ratio DROP_RATIO
                        dropout ratio (default: 0.5)
  --num_layer NUM_LAYER
                        number of GNN message passing layers (default: 5)
  --channels CHANNELS   String with dimension of each DS layer, separated by "-"(considered only if args.model is deepsets)
  --emb_dim EMB_DIM     dimensionality of hidden units in GNNs (default: 300)
  --jk JK               JK strategy, either last or concat (default: last)
  --batch_size BATCH_SIZE
                        input batch size for training (default: 32)
  --learning_rate LEARNING_RATE
                        learning rate for training (default: 0.01)
  --decay_rate DECAY_RATE
                        decay rate for training (default: 0.5)
  --decay_step DECAY_STEP
                        decay step for training (default: 50)
  --epochs EPOCHS       number of epochs to train (default: 100)
  --num_workers NUM_WORKERS
                        number of workers (default: 0)
  --dataset DATASET     dataset name (default: MUTAG)
  --policy POLICY       Subgraph selection policy in {edge_deletion, node_deletion, ego, ego_plus, original} (default: edge_deletion)
  --num_hops NUM_HOPS   Depth of the ego net if policy is ego (default: 2)
  --seed SEED           random seed (default: 0)
  --patience PATIENCE   patience (default: 20)
  --test                quick test
  --filename FILENAME   filename to output result (default: )
````

## Hyperparameter tuning

To perform hyperparameter tuning, we make use of `wandb`:

1. Run
    ```bash
    wandb sweep configs/<config-name>
    ````
    to obtain a sweep id `<sweep-id>`

2. Run the hyperparameter tuning with
    ```bash
    wandb agent <sweep-id>
    ```
    You can run the above command multiple times on each machine you would like to contribute to the grid-search

## Credits

Code for model training and evaluation is heavily based on the official [ESAN author implementation](https://github.com/beabevi/ESAN).

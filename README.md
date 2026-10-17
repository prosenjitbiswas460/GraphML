# When exposure bias becomes structural bias

This experiment asks whether an exposure-biased user–item graph makes LightGCN learn the exposure neighborhood instead of the preference neighborhood.

Every run trains the same model on several graphs over the same users and items:

| Graph | Edges | What it answers |
|---|---|---|
| `raw` | Edges that survived the exposure policy, weight 1 | The biased recommender |
| `uniform_matched` | The same number of edges, drawn from the preference graph without the popularity skew | Whether the harm is the skew rather than the smaller sample. Built for the simulated datasets |
| `ips_graph` | The raw edges, with clipped inverse propensity as the adjacency weight and loss weight 1 | Whether the exposure skew is in the propagated neighborhood. This is the structural correction |
| `ips_loss` | The same raw edges, with adjacency weight 1 and clipped inverse propensity on the BPR pairs | The statistical correction, without changing the graph |
| `resampled` | A subset of the raw edges kept with probability proportional to `1 / p` | A deconfounded edge set, trained with uniform weights |
| `counterfactual` | The preference graph, or the random-exposure graph on KuaiRand | The graph the model would have trained on without the exposure filter |
| `mf_raw` | The raw edges with zero propagation layers | Optional (`--include-mf`). Asks whether message passing is needed for the leakage |

Held-out labels are preferences (or random-policy feedback on KuaiRand). The core filter is fit on the training split, so a test edge cannot keep a user or an item in the catalog. Early stopping uses validation NDCG, and the test split is scored once on that checkpoint.

Negatives are every item outside the training split. Validation and test positives are eligible negatives; protecting them would use held-out labels while training. Every model ranks the same candidates: an item is hidden only if it appears in some model's training graph. Inverse propensity is applied once, either in the adjacency (`ips_graph`) or in the loss (`ips_loss`).

## Datasets

Run these three.

1. **KuaiRec** (`--dataset kuairec`). The small matrix is almost fully observed. The code binarizes `watch_ratio > 2` and then hides training edges with a known popularity policy. This is the clean identification result: same preferences, two graphs.
2. **KuaiRand-Pure** (`--dataset kuairand`). Standard-policy feedback is the raw graph. Random insertions are the counterfactual graph and the unbiased test. Item propensities are impression counts from 8–21 April, before that random window. The production logging probability is not in the file.
3. **MovieLens-1M** (`--dataset movielens`). Ratings of 4 and above are treated as the preference graph, then masked by a known policy. This is a controlled simulation on a standard graph-recommendation benchmark. It is not a fully observed preference matrix.

Two extra settings use the same code:

- `--dataset kuairec_natural` trains on the big-matrix log and evaluates held-out labels from the small matrix. Propensities are platform frequencies, not a known policy.
- `--dataset synthetic` is a two-community preference graph with an independent exposure head. Use it to check that the environment runs.
- `--policy uniform` is the negative control for the simulated datasets. The mask no longer prefers popular items, so `raw` and the two IPS models should look alike.

## Metrics

Ranking is full-catalog Recall and NDCG at 10 and 20. Tail, mid, and head are thirds of the items that appear in a training graph, ordered by exposure score. Items absent from training are left out of those slices. `head_tail_gap` is head NDCG minus tail NDCG at the larger cutoff.

Neighborhoods use cosine 10-NN among items that share users. A slot stays empty when an item has fewer than 10 real co-occurrence neighbors.

- preference neighbors come from co-occurrence in the counterfactual training graph
- exposure neighbors come from co-occurrence in the raw graph
- `leakage` is the embedding neighborhood's overlap with exposure neighbors, minus its overlap with preference neighbors

A larger leakage means the learned item neighborhood recovered what was shown together. `item_cka` and `user_cka` compare each model with the counterfactual model's representations. `knn_popularity_corr` is the correlation between an item's exposure score and the exposure score of its embedding neighbors.

The JSON also records a manipulation check, `head_edge_share`. Under a popularity policy the raw graph should place more of its edges on the top 20% of items than the counterfactual graph does. If that gap is missing, the later metrics are not evidence about exposure.

Contrasts are named by their direction. `leakage_raw_minus_uniform_matched > 0` means the popularity skew pulled neighborhoods toward co-exposure at matched density. That is the decision contrast. `tail_ndcg_ips_graph_minus_raw > 0` means adjacency reweighting ranked the least-exposed test items higher. The counterfactual neighborhood overlap is an upper bound: those neighbors are built from the same graph that model trained on.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest tests.test_core

# pipeline check; usually under a minute on CPU
python run_experiment.py --dataset synthetic --epochs 15 --dim 32 --layers 2 --device cpu

# the three study datasets
python run_experiment.py --dataset kuairec --download --device cuda --seeds 0,1,2
python run_experiment.py --dataset kuairand --download --device cuda --seeds 0,1,2
python run_experiment.py --dataset movielens --download --device cuda --seeds 0,1,2
```

`--dry-run` builds the graphs and prints the manipulation check without training. `--download` fetches MovieLens-1M, KuaiRec, or KuaiRand-Pure into `--data-dir` when the files are not already there. Results go to `results/<dataset>_<policy>_a<alpha>_r<align>_L<layers>_seed<k>.json`. A multi-seed run also writes a summary with means and standard deviations.

## Phase diagram

This is the comparison that asks when exposure bias becomes structural.

```bash
python run_phase.py --dataset synthetic --epochs 40 --dim 64 --device cpu \
  --alphas 0,0.25,0.5,1 --aligns 0,0.5,1 --layer-grid 1,3 --seeds 0,1

python run_phase.py --dataset kuairec --download --device cuda \
  --alphas 0.5,1,2 --layer-grid 1,3 --seeds 0
```

`--exposure-align` is the fraction of the synthetic exposure head drawn from preference community 0. Zero is an exposure head independent of preference. One is ordinary popularity bias. `--alphas` is the exposure skew. The summary is `results/phase_summary.json`.

The phase command lowers the propensity floor to 0.01 unless you pass `--min-prop`. The single-run floor of 0.1 clips the synthetic tail for every alpha at or above 0.5, and those settings would then build the same mask. The run prints the tail propensity for each alpha.

Three fields carry the claim:

- `frontier` is the smallest alpha at which raw leakage is positive and larger than `uniform_matched`.
- `propagation` compares the untrained LightGCN polynomial at 1 and 3 layers. The support gap is raw minus `uniform_matched`. The weight gap is raw minus `ips_graph`. Extra layers should widen the support gap by more than the weight gap, because inverse propensity rescales edges that exist and cannot restore edges the policy removed.
- `selector` compares validation NDCG with leakage as predictors of NDCG on a second test. That test is a uniform slice of the held-out preferences. It is not used to build the graph or to pick the checkpoint. The correlation uses `raw`, `uniform_matched`, `ips_graph`, and `ips_loss`, which share an edge count. A useful leakage score has a negative Spearman with shifted NDCG.

On KuaiRec and MovieLens the same alpha sweep applies to the simulated mask. Alignment is synthetic only. On KuaiRand the random-exposure test is already the policy shift, and alpha only changes the propensity estimate.

Useful knobs: `--alpha` and `--min-prop` set the simulated exposure skew, `--ips-clip` caps the inverse-propensity weight, `--watch-threshold` and `--feedback` change what counts as a positive on KuaiRec and KuaiRand, and `--include-mf` adds the no-propagation control. If `val_history` is still improving at the last epoch, raise `--epochs`.

## Caveats

On MovieLens and on the controlled KuaiRec run, the counterfactual model sees every training preference, so a higher NDCG mixes less bias with more edges. Read it next to `uniform_matched`, which holds the edge count fixed.

On KuaiRand there is no uniform-matched graph, because the random-policy edges are not a superset of the standard-policy edges. The IPS weights use a one-dimensional item-frequency model estimated from the earlier impression window, not the platform's logging propensity.

`kuairec_natural` keeps a run only when the big matrix shares a large positive subgraph with the fully observed block. Otherwise the loader stops and points you back to `--dataset kuairec`.

A test item can be drawn as a negative. That pushes it down. It does not tell the model that the item was a positive.

## References

LightGCN: He et al., SIGIR 2020. KuaiRec and KuaiRand: Gao et al., CIKM 2022. MovieLens-1M: GroupLens. The exposure mask follows the popularity-propensity setup used in unbiased recommendation (Schnabel et al., ICML 2016). Inverse-propensity aggregation inside a LightGCN was studied by Navip (Kim et al., CIKM 2022). Chen et al. (arXiv 2305.14886) show that graph convolution amplifies popularity bias. This experiment adds the density-matched and fully observed controls, and it scores neighborhood leakage rather than ranking alone.

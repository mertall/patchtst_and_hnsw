# Very smol PatchTST and lightweight HNSW

### Overview
This project trains a lightweight PatchTST model in 10 hours for a univariate time-series—the **Australian electricity demand** series. We preprocess the series into fixed-length windows, split train/test **before** training to avoid leakage, train a Transformer-based patch model to predict future windows, then index the train embeddings in an HNSW index. We will then compare brute force KNN to HNSW using the test embeddings as queries on our HNSW index of train embeddings.

---

### Experiments

1. **Data Preparation**  
   - Loaded univariate series from the `australian_electricity_demand` dataset.  
   - Created sliding windows of length 96 for “past” context and 24 for “future” targets.  
   - Split chronologically: Randomly split test into train and test.

2. **Baseline Training**  
   - **Model**: PatchTST-small (8 heads, 2 layers, patch length 16)  
   - **Optimizer**: AdamW, learning rate = 2 × 10⁻⁴  
   - **Schedule**: 10 epochs, batch size = 128  

3. **HNSW Indexing & Search**  
   - **Embeddings**: Extracted 128-dim patch embeddings from the trained model for each window.  
   - **Index**: hnswlib L2 space, `M=16`, `ef_construction=200`.  
   - **Query**: set `ef=50`, retrieved top-k neighbors (`k=5`) for each test embedding.

---

### Results

- **Forecasting (PatchTST)**  
  - Test RMSE: 209.9130    
  Graphs showing prediction versus actual value:     
![alt text](graphs/output.png)
![alt text](graphs/output1.png)
![alt text](graphs/output2.png)

- **HNSW Vector Search**  
Highly recommend reading notebook summary of how HNSW work if you are not familiar.  
HNSW index with all of our training data, we search our test data converted to embeddings and compare KNN to HNSW performance.
  - **Recall@5** (percentage of true nearest neighbors in top-5): 100%
  - **Average query latency**: Brute-force total time: 0.3219s (0.003219s/query)     
                               HNSW total time: 0.0435s (0.000435s/query)     
                               Mean recall for all 5: 100.00%    
                               Speedup: approx. 7.4 × faster with HNSW    

These results show that the patch embeddings capture meaningful temporal patterns and that HNSW can retrieve similar contexts with high recall and low latency. We should note that HNSW requires us to store the full index in memory for search, this means as embeddings scale our memory size will increase signficantly. We could leverage sharding to work around this issue, with potential to optimize sharding strategy based on use case.


### Roadmap & Experiment Tracker

| ✓? | Idea                         | Hypothesis                                                                                           | Proposed Solution                                                                                                           |
|----|------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [ ] | Masked self‑supervision **In Progress** | Pre‑training with masked patches yields better representations and cuts label need.                 | Train `PatchTSTModel` (`do_mask_input=True`) for 1 epoch on ever datset in GiftEvalPretrain, peeking at each dataset on only past data and 20% of data masked, then fine‑tune with past and future from train data in GiftEval; log val MSE/MAE vs scratch.                    |
| [ ] | More training data  **In Progress**  | A broader corpus (weather, stocks, etc.) provides richer temporal priors and lowers variance.        | Merge additional GiftEval splits;                            |
| [ ] |Compute power & Heavier Model     **In Progress**    | Longer schedules & larger batches let PatchTST converge to lower minima. Wider emdbeddings, longer patch and stride length to store long-term temporal relationships.                            | Train 100 epochs on T4, use mixed precision, change paramaters mentioned earlier.                                                        |
| [ ] | **HNSW tuning**             | Index hyper‑params affect recall/latency trade‑off as corpus scales.                                 | Grid‑search `M {16,32}` × `ef_construction {100,200,400}` on >100 k embeddings.                                            |
| [ ] | **RAG over time‑series**    | Text‑prompt retrieval of Time Series | Finetune CLIP with constrastive learning, use filenames to generate text tokens and create associative tags, for now create a static mapping of file names to overarching domain(s)... then we pull in anything not in the domain and label as negative.
| [ ] | **Frequency-Aware Parameterization** | To improve model quality for different types of real world data, we should parameterize windows based on frequency | Consideration of the different frequencies across datasets (e.g., daily, weekly, monthly) and parametezing the window length and scaling factor based on frequency given.

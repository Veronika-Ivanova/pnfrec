# Benefiting from Negative yet Informative Feedback by Contrasting Opposing Sequential Patterns

This repository contains code for *Benefiting from Negative yet Informative Feedback by Contrasting Opposing Sequential Patterns* paper.

## Usage
Install requirements:
```sh
pip install -r requirements.txt
```

For configuration we use [Hydra](https://hydra.cc/). Parameters are specified in [config files](runs/configs/).

Example of run for ML1M dataset:

```sh
cd runs
python run_pnfrec.py 
```
Below are the tables with main results for different datasets. The best result for **$\Delta HR@10$** and **$\Delta NDCG@10$** in each row is highlighted in bold. Each metric is averaged over five runs and presented as Mean $\pm$ SD.

**$\Delta HR@10 = HR_p@10 - HR_n@10$** 

**$\Delta NDCG@10 = NDCG_p@10 - NDCG_n@10$**

Our models:
* **$PNFRec$**, trained using positive and negative CE;
* **$PNFRec_{pn}$**, trained using positive CE and contrastive
loss;
* **$PNFRec_{pc}$**, trained using positive and negative CE and contrastive loss.

Baseline models:
* **$SASRec_p$**, trained on positive interactions only;
* **$SASRec$**, trained on the user’s entire interaction history;
* **$SASRec_c$**, trained on the loss objective with the contrastive term introduced in [Enhancing Sequential Music Recommendation with Negative Feedback-informed Contrastive Learning](https://arxiv.org/abs/2409.07367).

**MovieLens-1m**

| **Model**          | **$PNFRec (our)$**   | **$PNFRec_{pn} (our)$**|**$PNFRec_{pc} (our)$**| **$SASRec_p$**   |**$SASRec_c$**    |**$SASRec$**      |
|--------------------|--------------------|--------------------|-------------------|----------------|----------------|----------------|
| **$HR_p@10$**      | 0.1716 $\pm$ 0.0083| 0.1677 $\pm$ 0.0131| 0.1612 $\pm$ 0.0022| 0.1587 $\pm$ 0.0112| 0.1638 $\pm$ 0.0083| 0.1683 $\pm$ 0.0089|
| **$NDCG_p@10$**    | 0.1041 $\pm$ 0.0026|	0.1024 $\pm$ 0.0069| 0.0921 $\pm$ 0.0047|	0.0935 $\pm$ 0.0086| 0.0892 $\pm$ 0.0053| 0.0927 $\pm$ 0.0034|      
| **$\Delta HR@10$** | 0.0724 $\pm$ 0.0152|	0.0644 $\pm$ 0.0148| **0.0760 $\pm$ 0.0109**|	0.0726 $\pm$ 0.0170| -0.0213 $\pm$ 0.0044|	-0.0160 $\pm$ 0.0061|
|**$\Delta NDCG@10$**| **0.0484 $\pm$ 0.0055**|	0.0467 $\pm$ 0.0068| 0.0435 $\pm$ 0.0047|	0.0449 $\pm$ 0.0052| -0.0174 $\pm$ 0.0036|	-0.0116 $\pm$ 0.0064|
| Training time, s      | 208|188|	157|	123|	423|	160|
| Best epoch     |41|	44|	40|	42|	52|	41|

**MovieLens-20m**

| **Model**          | **$PNFRec (our)$**   | **$PNFRec_{pn} (our)$**|**$PNFRec_{pc} (our)$**| **$SASRec_p$**   |**$SASRec_c$**    |**$SASRec$**      |
|--------------------|--------------------|--------------------|-------------------|----------------|----------------|----------------|
| **$HR_p@10$**      | 0.0933 $\pm$ 0.0045| 0.0930 $\pm$ 0.0028| 0.0920 $\pm$ 0.0033| 0.0873 $\pm$ 0.0053| 0.0893 $\pm$ 0.0014| 0.0902 $\pm$ 0.0026|
| **$NDCG_p@10$**    | 0.0541 $\pm$ 0.0025|	0.0544 $\pm$ 0.0015| 0.0538 $\pm$ 0.0009|	0.0511 $\pm$ 0.0032| 0.0541 $\pm$ 0.0010| 0.0545 $\pm$ 0.0011|      
| **$\Delta HR@10$** | **0.0288 $\pm$ 0.0085**|	0.0265 $\pm$ 0.0071| 0.0278 $\pm$ 0.0033|	0.0276 $\pm$ 0.0047| 0.0016 $\pm$ 0.0022|	0.0043 $\pm$ 0.0050|
|**$\Delta NDCG@10$**| 0.0168 $\pm$ 0.0046|	0.0162 $\pm$ 0.0032| **0.0176 $\pm$ 0.0034**|	0.0174 $\pm$ 0.0042| 0.0007 $\pm$ 0.0023|	0.0028 $\pm$ 0.0029|
| Training time, s      | 1023|	898|	564|	549|	2755|	1959|
| Best epoch     |17|	15|	17|	17|	23|	19|

**Toys & Games**

| **Model**          | **$PNFRec (our)$**   | **$PNFRec_{pn} (our)$**|**$PNFRec_{pc} (our)$**| **$SASRec_p$**   |**$SASRec_c$**    |**$SASRec$**      |
|--------------------|--------------------|--------------------|-------------------|----------------|----------------|----------------|
| **$HR_p@10$**      | 0.1109 $\pm$ 0.0083| 0.1119 $\pm$ 0.0072| 0.1008 $\pm$ 0.0067| 0.0998 $\pm$ 0.0039| 0.1279 $\pm$ 0.0041| 0.1275 $\pm$ 0.0043|
| **$NDCG_p@10$**    | 0.0731 $\pm$ 0.0036|	0.0743 $\pm$ 0.0027| 0.0656 $\pm$ 0.0034|	0.0661 $\pm$ 0.0035| 0.0795 $\pm$ 0.0019| 0.0804 $\pm$ 0.0014|      
| **$\Delta HR@10$** | 0.0419 $\pm$ 0.0175|	**0.0429 $\pm$ 0.0156**| 0.0364 $\pm$ 0.0100|	0.0354 $\pm$ 0.0037| -0.0284 $\pm$ 0.0096|	-0.0425 $\pm$ 0.0116|
|**$\Delta NDCG@10$**| 0.0377 $\pm$ 0.0069|	**0.0385 $\pm$ 0.0060**| 0.0295 $\pm$ 0.0025|	0.0290 $\pm$ 0.0024| -0.0191 $\pm$ 0.0079|	-0.0256 $\pm$ 0.0078|
| Training time, s     | 254|	222|	160|	158|	313|	260|
| Best epoch     |26|	26|	26|	26|	35|	30|

**Kion**

| **Model**          | **$PNFRec (our)$**   | **$PNFRec_{pn} (our)$**|**$PNFRec_{pc} (our)$**| **$SASRec_p$**   |**$SASRec_c$**    |**$SASRec$**      |
|--------------------|--------------------|--------------------|-------------------|----------------|----------------|----------------|
| **$HR_p@10$**      | 0.2157 $\pm$ 0.0005| 0.2161 $\pm$ 0.0012| 0.2129 $\pm$ 0.0011| 0.2135 $\pm$ 0.0020| 0.2182 $\pm$ 0.0011| 0.2180 $\pm$ 0.0014|
| **$NDCG_p@10$**    | 0.1310 $\pm$ 0.0005|	0.1313 $\pm$ 0.0006| 0.1293 $\pm$ 0.0006|	0.1300 $\pm$ 0.0011| 0.1308 $\pm$ 0.0009| 0.1316 $\pm$ 0.0009|      
| **$\Delta HR@10$** | **0.1036 $\pm$ 0.0016**|	0.1026 $\pm$ 0.0023| 0.1026 $\pm$ 0.0024|	0.1024 $\pm$ 0.0019| 0.0727 $\pm$ 0.0018|	0.0719 $\pm$ 0.0047|
|**$\Delta NDCG@10$**| 0.0700 $\pm$ 0.0010|	**0.0701 $\pm$ 0.0013**| 0.0684 $\pm$ 0.0011|	0.0689 $\pm$ 0.0009| 0.0506 $\pm$ 0.0014|	0.0521 $\pm$ 0.0012|
| Training time, s    | 1666|	1467|	714|	1004|	1730|	1124|
| Best epoch     |36 |	30|	21|	19|	32|	28|

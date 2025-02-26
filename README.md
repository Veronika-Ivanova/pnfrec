# Benefiting from Negative yet Informative Feedback in Sequential Recommendations with Contrastive Learning

This repository contains code for *Benefiting from Negative yet Informative Feedback in Sequential Recommendations with Contrastive Learning* paper.

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

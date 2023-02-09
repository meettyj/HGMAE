# HGMAE

This is the official repo for the AAAI'23 paper "[Heterogeneous Graph Masked Autoencoders](https://arxiv.org/abs/2208.09957)".


## How to Run the code

```
python main.py --dataset dblp --task classification --use_cfg 
```

Supported datasets: "dblp", "freebase", "acm", "aminer".

Supported tasks: "classification", "clustering". 


## Citing HGMAE

If you find HGMAE useful, please cite our paper.
```
@inproceedings{HGMAE,
  title={Heterogeneous Graph Masked Autoencoders},
  author={Tian, Yijun and Dong, Kaiwen and Zhang, Chunhui and Zhang, Chuxu and Chawla, Nitesh V},
  booktitle = {AAAI},
  year={2023}
}
```

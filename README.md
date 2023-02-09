# HGMAE
Code and data for Heterogeneous Graph Masked Autoencoders.


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
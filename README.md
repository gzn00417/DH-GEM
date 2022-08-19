# DH-GEM

Source code for the paper, Talent Demand-Supply Joint Prediction with Dynamic Heterogeneous Graph Enhanced Meta-Learning, in SIGKDD 2022.

# Requirement

```
torch==1.10
numpy==1.21.2
pandas==1.3.4
pytorch-lightning==1.5.3
dgl==0.6.1
```

# Usage

You can easily run our code by

```python
python code/main.py -R your-data-path -D dataset-name
```

More hyper-parameters setting please refer to `code/args.py`.

# Dataset

In this work, we conduct experiments on 3 datasets, i.e. IT, FIN, CONS. They have the same format and in this repository we provide an example of job postings (demand) and work experiences (supply) data. You can collect your own datasets and run our code.

### Job Postings (Demand)

|Company|Time|Position|Location|
|:-:|:-:|:-:|:-:|
|Amazon|201903|Information|Boston, MA, US|
|...|...|...|...|

### Work Experiences (Supply)

|People|Company|StartDate|EndDate|Position|
|:-:|:-:|:-:|:-:|:-:|
|456342|IBM|201603|201603|Information|
|...|...|...|...|

# Citation

If you find our work interesting, you can cite the paper as

```
@inproceedings{guo2022talent,
  title={Talent Demand-Supply Joint Prediction with Dynamic Heterogeneous Graph Enhanced Meta-Learning},
  author={Guo, Zhuoning and Liu, Hao and Zhang, Le and Zhang, Qi and Zhu, Hengshu and Xiong, Hui},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2957--2967},
  year={2022}
}
```

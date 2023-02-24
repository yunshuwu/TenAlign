# TenAlign: Joint Tensor Alignment and Coupled Factorization

Official repository for the paper at this [TenAlign](https://www.cs.ucr.edu/~epapalex/papers/22-ICDM-TenAlign.pdf) link. 


Multimodal datasets represented as tensors oftentimes share some of their modes. However, even though there may exist a one-to-one (or perhaps partial) correspondence between the coupled modes, such correspondence/alignment may not be given, especially when integrating datasets from disparate sources. This is a very important problem, broadly termed as entity alignment or matching, and subsets of the problem such as graph matching have been extremely popular in the recent years. In order to solve this problem, current work computes the alignment based on existing embeddings of the data. This can be problematic if our end goal is the joint analysis of the two datasets into the same latent factor space: the embeddings computed separately per dataset may yield a suboptimal alignment, and if such an alignment is used to subsequently compute the joint latent factors, the computation will similarly be plagued by compounding errors incurred by the imperfect alignment. In this work, we are the first to define and solve the problem of joint tensor alignment and factorization into a shared latent space. By posing this as a unified problem and solving for both tasks simultaneously, we observe that the both alignment and factorization tasks benefit each other resulting in superior performance compared to two-stage approaches. We extensively evaluate our proposed method TENALIGN and conduct a thorough sensitivity and ablation analysis. We demonstrate that TENALIGN significantly outperforms baseline approaches where embedding and matching happen separately.

TODO: update the abstract (shorter), also introduce the problem and two formulas here

## Requirements

The implementation makes use of the following toolboxes/packages, which need to be downloaded and installed separately:
* MATLAB CMTF Toolbox v1.1, 2014
* D. M. Dunlavy, T. G. Kolda, and E. Acar, Poblano v1.0: A Matlab Toolbox for Gradient-Based Optimization, SAND2010-1422, March 2010. Avaiable online at https://github.com/sandialabs/poblano_toolbox
* Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox, Version 3.1. Available online at https://www.tensortoolbox.org, 2020
* G. Chierchia, E. Chouzenoux, P. L. Combettes, and J.-C. Pesquet. "The Proximity Operator Repository. User's guide". Availaible online at http://proximity-operator.net/download/guide.pdf 
* S. Becker, “L-BFGS-B C code with Matlab wrapper,” 2019. Available online at https://github.com/stephenbeckr/L-BFGS-B-C, see also: R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing , 16, 5, pp. 1190-1208

## Methods

### TENALIGN-L1
Formula $L1$ is defining a harder optimization problem, and the permutation matrix $\Pi$ is directly affecting tensor $Y$

### TENALIGN-L2
Formula $L2$ softly enforces the permutation matrix $\Pi$ as a regularization term.


## Example

Tensor $X$ and $Y$ are coupled through the first mode, where tensor $X$ of size 10×30×40 and tensor $Y$ of size 10×70×10, and they are created by ground-truth factor matrices with rank R = 4. 

Here the synthetic data generation is using implementation from paper:
E. A. Carla Schenker, Je ́re ́my Cohen, “An optimization framework for regularized linearly coupled matrix-tensor factorization,” in EUSIPCO 2020 - 28th European Signal Processing Conference, Jan 2021, Virtual, Netherlands. EUSIPCO, 2020, pp. 1–5.


### Evaluate the example by using algorithm TENALIGN-L1 with metric Raw Accuracy 
Metric Raw Accuracy: refer to equation (10) in the paper

Run the following commend to jointly decomposite tensor $X$ and $Y$ via algorithm TENALIGN-L1: 

First go into matlab environment:
```shell
matlab
```

Run the experiment:
```shell
experiment_f1_syn_raw
```

### Evaluate the example by using algorithm TENALIGN-L1 with metric Clustering Accuracy 
Metric Clustering Accuracy: refer to equation (12) $\Pi_{accuracy}$ in the paper

Run the following commend to jointly decomposite tensor $X$ and $Y$ via algorithm TENALIGN-L1: 

First go into matlab environment:
```shell
matlab
```
Run the experiment:

```shell
experiment_f1_syn_clusterAcc
```
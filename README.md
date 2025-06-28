# Adaptive Time Rescaling Stochastic Gradient MCMC for Bayesian Deep Learning


# Dependencies
* Python 2.7
* [PyTorch 1.2.0](http://pytorch.org/) 
* [torchvision 0.4.0](https://github.com/pytorch/vision/)


## CIFAR-10
To train models with SGLD on CIFAR-10, run:
```
cd experiments
python cifar/sgld.py --dir=<DIR> --data_path=<PATH> --temperature=<TEMPERATURE>
```
```
Parameters:

* ```DIR``` &mdash; path to training directory where samples will be stored
* ```PATH``` &mdash; path to the data directory
* ```ALPHA``` &mdash; One minus the momentum term. One is corresponding to SGLD and a number which is less than one is corresponding to SGHMC
* ```TEMPERATURE``` &mdash; temperature in the posterior

To test the ensemble of the collected samples on CIFAR-10, run `experiments/cifar_ensemble.py`


## CIFAR-100

Similarly, for CIFAR-100, run

```
cd experiments
python cifar100/sgld.py --dir=<DIR> \
                           --data_path=<PATH> \
                           --temperature=<TEMPERATURE>
```

To test the ensemble of the collected samples on CIFAR-100, run `cifar100/eval.py`



# References
* Code of Gaussian mixtures is adapted from https://github.com/tqchen/ML-SGHMC
* Models are adapted from https://github.com/kuangliu/pytorch-cifar
* Code borrowed from cSGMCMC

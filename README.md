This is an unofficial TensorFlow implementation of [Presence-Only Geographical Priors for Fine-Grained Image Classification](https://arxiv.org/abs/1906.05272)

### Results

| Prior                              | Classifier* | Dataset  | Accuracy |
|------------------------------------|-------------|----------|----------|
| No Prior [1]                       | InceptionV3 | iNat2018 | 60.20    |
| Geo Prior (no photographer) [1]    | InceptionV3 | iNat2018 | 72.84    |
| Geo Prior (no photographer) [(ours)](https://drive.google.com/file/d/1lQ3X1x3cAu-o0hvg1eribtY0oKfDjtFx/view?usp=sharing) | InceptionV3 | iNat2018 | 71.53    |

*Classifier predictions are from original paper [1] and they can be found in [2].

### Source

[1] Original paper: https://arxiv.org/abs/1906.05272

[2] Official PyTorch code: https://github.com/macaodha/geo_prior

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)
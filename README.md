This is an unofficial TensorFlow implementation of [Presence-Only Geographical Priors for Fine-Grained Image Classification](https://arxiv.org/abs/1906.05272)

Currently, we have only implemented the object branch and its respective loss.

### Requirements

Prepare an environment with python=3.8, tensorflow=2.3.1.

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

### Training

To train a geo prior model use the script `train.py`:
```bash
python train.py --train_data_json=../data/train2018.json \
    --train_location_info_json=PATH_TO_BE_CONFIGURED/train2018_locations.json \
    --val_data_json=PATH_TO_BE_CONFIGURED/val2018.json \
    --val_location_info_json=PATH_TO_BE_CONFIGURED/val2018_locations.json \
    --model_dir=PATH_TO_BE_CONFIGURED/geo_prior_ckp/ \
    --random_seed=42
```

Other training hyperparams can also be passed as flags. For more parameter information, please refer to `train.py`.

### Evaluation

To evaluate a model use the script `eval.py`:
```bash
python eval.py --test_data_json=PATH_TO_BE_CONFIGURED/val2018.json \
    --test_location_info_json=PATH_TO_BE_CONFIGURED/val2018_locations.json \
    --cnn_predictions_file=PATH_TO_BE_CONFIGURED/inat2018_val_preds_sparse.npz \
    --ckpt_dir=PATH_TO_BE_CONFIGURED/geo_prior_ckp/
```

### Results

| Prior                                   | Classifier* | Dataset  | Accuracy |
|-----------------------------------------|-------------|----------|----------|
| No Prior [1]                            | InceptionV3 | iNat2018 | 60.20    |
| Geo Prior (no photographer) [1]         | InceptionV3 | iNat2018 | 72.84    |
| Geo Prior (no photographer) [(ours)](https://drive.google.com/file/d/1lQ3X1x3cAu-o0hvg1eribtY0oKfDjtFx/view?usp=sharing) | InceptionV3 | iNat2018 | 71.53    |
| Geo Prior + BN (no photographer) (ours) | InceptionV3 | iNat2018 | 71.91    |

*Classifier predictions are from original paper [1] and they can be found in [2].

### Source

[1] Original paper: https://arxiv.org/abs/1906.05272

[2] Official PyTorch code: https://github.com/macaodha/geo_prior

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)
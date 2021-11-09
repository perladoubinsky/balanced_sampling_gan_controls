# Multi-attribute balanced sampling for disentangled GAN controls

In this work, we propose a simple and general approach to avoid the post-processing step often necessary in latent space editing methods to disentangle the controls after extraction from the latent space of GANs. We apply it in the context of InterFaceGAN a well-known method which allows to extract controls associated with facial attributes. Our method simply consists in balancing the semantics of the dataset before learning the boundary. For more details, please check [our paper](https://arxiv.org/abs/2111.00909).

Other differences with InterFaceGAN :
* We select samples for which there is a high confidence across all attributes (since we balance the data w.r.t. to every available attribute, we need to make sure the labelling is accurate)
* We compute the boundary using the line connecting the centroids of the two classes instead of using SVMs (as it gives better results on balanced data).

## Generate data

In addition to generate the data, you can also generate the prediction for each attribute by specifying ``--generate_prediction``. Before hand, you need to download [our pre-trained attribute prediction model trained on CelebA](https://drive.google.com/file/d/12ZpxZIuoTZYIMkhZQFKrDo6waEG8ejNq/view?usp=sharing) and put it in the folder ``predictors/pretrain``. We trained it using a multi-task setting with a cross-entropy loss for each attribute (Accuracy for 'Smile': 0.93, 'Gender': 0.98, 'Glasses': 0.99, 'Age': 0.86). The attributes scores are then given in the form of a python dictionary.

```bash
python generate_data.py 
       -m pggan_celebahq \
       -o data/pggan_celebahq \
       -n 1000000 \
       --generate_prediction
```

## Learn a boundary

You need to specify :
- the attribute you want to control (choose among 'Smile', 'Age', 'Gender', 'Glasses').
- the confidence threshold (used to select the most confident samples).
- the number of samples to use to compute the boundary (we recommend to inspect the contingency matrix and choose the number of samples according to the number of samples representing the rarest combination).


And pass the dictionary ``scores_dict.npy`` containing the scores for all attributes (obtained at previous step).

```bash
python train_boundary_balanced_sampling.py \
        -o boundaries_balanced_sampling/pggan_celebahs_gender \
        -c data/pggan_celebahq/z.npy \
        -s data/pggan_celebahq/scores_dict.npy 
        -a 'Gender' \
        -n 1000 \
        -t 0.9 \
        --boundary_name 'pggan_celebahq_gender_boundary.npy'
```

You can find already computed boundaries in the fodler ``boundaries_balanced_sampling``.

## Edit

```bash
python edit.py \
        -m pggan_celebahq \
        -b boundaries_balancing/pggan_celebahq_gender_boundary.npy \
        -n 1 \
        -o results/pggan_celebahq_gender_editing
```

## Multi-attribute balanced sampling

You can find our sampling function `contingency_sample()` in `mab_sampling.py`.

## Acknowledgements

This code is built upon [InterfaceGAN](https://github.com/genforce/interfacegan) and [Higan](https://github.com/genforce/higan) (for the predictors).


## Citation

```bibtex
@misc{doubinsky2021multiattribute,
      title={Multi-Attribute Balanced Sampling for Disentangled GAN Controls}, 
      author={Perla Doubinsky and Nicolas Audebert and Michel Crucianu and Hervé Le Borgne},
      year={2021},
      eprint={2111.00909},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
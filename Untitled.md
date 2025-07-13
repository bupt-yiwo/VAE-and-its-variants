<h1 align="center">
  <b>VAE-and-its-variants</b><br>
</h1>




This project is for **personal learning and experimentation**, containing implementations of **VAE (Variational Autoencoder)** and its various **variants**, such as β-VAE, VQ-VAE, and more.

> 🔗 The code is primarily adapted from [AntixK's PyTorch-VAE repository](https://github.com/AntixK/PyTorch-VAE/tree/master).
>
> 🎨 The dataset used is the [Anime Face Dataset](https://github.com/bchao1/Anime-Face-Dataset) by bchao1.



### Requirements

- Python (any version)
- PyTorch (any version)
- Pytorch Lightning  (any version)
- CUDA enabled computing device



### Usage

**Run**

```
$ cd PyTorch-VAE
$ python run.py -c configs/<config-file-name.yaml>
```

**View TensorBoard Logs**

```
$ cd logs/<experiment name>/version_<the version you want>
$ tensorboard --logdir .
```



### Results

| Model                                                        | Paper                                    | Reconstruction |
| ------------------------------------------------------------ | ---------------------------------------- | -------------- |
| VAE ([Code][vae_code], [Config][vae_config])                 | [Link](https://arxiv.org/abs/1312.6114)  | ![][1]         |
| VQ-VAE (*K = 512, D = 64*) ([Code][vqvae_code], [Config][vqvae_config]) | [Link](https://arxiv.org/abs/1711.00937) | ![][2]         |

### TODO

- [x] VanillaVAE
- [x] VQVAE



[vae_code]: models/vanilla_vae.py
[vqvae_code]: models/vq_vae.py
[vae_config]: configs/vanilla_vae.yaml
[vqvae_config]: configs/vq_vae.yaml
[1]: images/recons_VanillaVAE_Epoch_99.png
[2]: images/recons_VQ_VAE_Epoch_4.png

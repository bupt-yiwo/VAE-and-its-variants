du -sh /wangxiao/Protein/Data/0-4
    /wangxiao/Protein/Data/0-4

​    /wangxiao/Protein/Data/4-10



​    96_48_val_npy/

​     96_48_train_npy



老师，您可以联系超算平台管理员增加一下磁盘配额吗，磁盘配额超了，我在那个平台上没找到申请增加磁盘配额的申请入口。统计数据的时候有几个MAP发现之前处理失败了，现在还有几个MAP没处理，应该都是挺大的。error writing '/wangxiao/test_3G_file': Disk quota exceeded

| DATA                       | SIZE |
| -------------------------- | ---- |
| 96_48_val_npy（box数据）   | 1.9T |
| 96_48_train_npy（box数据） | 11T  |
| 0-4_mrc（map数据）         | 4.8T |
| 4-10_mrc（map数据）        | 1.9T |



然后就是我之前跟您说的那个mrc文件里有些特别大的问题，是大的structure（扫出来的box非常多），如果根据之前的随机采样不太合适（总会采样到大MAP，但是实际上小map是大多数情况），所以我改了一下，按照box数量的区间采样，每次在某个区间采样一定比例的，比如说这个epoch采样100个box，那在每个区间都采样25个，最终分配下来200个epoch后对于大多数的小MAP（0-100）都可以过4遍左右，100-1000两遍左右，1000-100000的刚好过1遍。

| Box count range | Number of  train maps/boxes | Number of  val maps/boxes |
| --------------- | --------------------------- | ------------------------- |
| 0-100           | 19129 / 391,893             | 3828 / 75548              |
| 100-1000        | 3518  /  858,700            | 706  /  177661            |
| 1000-5000       | 441    /  897,949           | 84    /  172320           |
| 5000-100000     | 60      /  832,676          | 12    /  140122           |

总 MAP 数: 
总数据量（所有 MAP 样本数之和）: 2981218
目录下实际文件数: 2981229

每个区间的 MAP 数量分布：
: 个 MAP
: 个 MAP
: 个 MAP
: 个 MAP
1000001+: 0 个 MAP



总 MAP 数: 
总数据量（所有 MAP 样本数之和）: 565651
目录下实际文件数: 565652

每个区间的 MAP 数量分布：
0-100:  个 MAP
100-1000: 个 MAP
1000-5000: 个 MAP
5000-1000001: 个 MAP
1000001+: 0 个 MAP

0-100，100-1000，1000-5000，5000-1000000









































<h1 align="center">
  <b>PyTorch VAE</b><br>
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



[vae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
[vqvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
[vae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/vae.yaml
[vqvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/vq_vae.yaml
[1]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/Vanilla%20VAE_25.png
[2]: https://github.com/AntixK/PyTorch-VAE/blob/master/assets/recons_Vanilla%20VAE_25.png

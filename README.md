# Pytorch code for OVPT

**OVPT: Optimal Viewset Pooling Transformer for 3D Object Recognition**

### Requirement

This code is tested on Python 3.6 and Pytorch 1.6.0 

### Dataset

First download the 20 views ModelNet10 and ModelNet40 dataset provided by [[rotationnet]](https://github.com/kanezaki/pytorch-rotationnet) and put it under `data`

`https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet10v2png_ori2.tar`
`https://data.airc.aist.go.jp/kanezaki.asako/data/modelnet40v2png_ori4.tar`

Then `python ov.py` to construct the optimal viewset

### Train ande test:

`python train.py`

The code is heavily borrowed from [[mvcnn-new]](https://github.com/jongchyisu/mvcnn_pytorch).

## Reference

Su, H., Maji, S., Kalogerakis, E., Learned-Miller, E.: Multi-view convolutional neural networks for 3d shape recognition. In: 2015 IEEE International Conference on Computer Vision (ICCV). (2015) 945–953

Kanezaki, A., Matsushita, Y., Nishida, Y.: Rotationnet: Joint object categorization and pose estimation using multiviews from unsupervised viewpoints. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. (2018) 5010–5019


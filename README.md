# jsai2020_fGAN
以下URLの発表に対する、参考実装です。類似手法のと比較や本手法自体の改良などの研究目的にのみ利用可能です。
This code is supplimental of the URL.You can use this for research purposes only such as comparison with similar methods and improvement of this method itself. 

https://confit.atlas.jp/guide/event/jsai2020/subject/3Rin4-23/tables?cryptoId=


# How to use
Here is 4 programs for each experiment

- 1D_normal/train.py:  fGAN only on 1D data.
- 1D_invG/train.py: fGAN+InvG on 1D data.
- 2D_normal/train.py: fGAN only on 2D data.
- 2D_invG/train.py:  fgan+InvG on 2D data.

May be you have to specify GPU No with --device argument like this
 ./train.py --device 0


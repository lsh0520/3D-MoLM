# 3D-MoLM: Towards 3D Molecule-Text Interpretation in Language Models

## We will release the checkpoints and curated datasets once accepted

## Requirements

python==3.8

* Install PyTorch with cuda-11.7 using conda by following the instructions in [link](https://pytorch.org/get-started/locally/)
* Install flash-attention by running `pip install flash-attn --no-build-isolation`. You might need to install the following dependencies for building the flash-attention module:
    * `pip install packaging ninja`
    * `conda install -c "nvidia/label/cuda-11.7.1" cuda-nvcc`
    * `conda install -c "nvidia/label/cuda-11.7.1" cuda-libraries-dev`
* Install the lastest version of opendela by runing `pip install git+https://github.com/thunlp/OpenDelta.git`
* Install lavis: `pip install rouge_score nltk salesforce-lavis`
* Install others: `pip install -U transformers pytorch-lightning`
* Install the lastest version of deepspeed: `pip install git+https://github.com/microsoft/DeepSpeed.git`
* Download nltk corpus:
```
import nltk
nltk.download('wordnet')
```


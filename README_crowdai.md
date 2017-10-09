installation
------------
Use the py2 branch of rllab

Anaconda Linux version: https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh

```shell
git submodule init
git submodule update

PYENV_VERSION=anaconda2-4.4.0 conda env create -f environment.yml

# make sure to install into the new conda env, either with pyenv's
# version stuff, or "activate rllab"
conda install --channel kidzik opensim
conda install --channel conda-forge lapack
pip install git+https://github.com/stanfordnmbl/osim-rl.git
pip install wandb
```

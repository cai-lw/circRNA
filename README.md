# circRNA
A course project of *Computational Biology* at Tsinghua University in the Spring semester of 2016

## Requirements

* Keras
* Theano or Tensorflow (as Keras backend)
* scikit-learn
* biopython

## Data preparation

Place these raw data files in `raw_data/`:

* `hg19.fa`
* `hg19_Alu.bed`
* `hsa_hg19_Rybak2015.bed`
* `all_exons.bed`

Then, run

```shell
python preprocess.py
```

and clean data will be generated in `clean_data/`

## Run

```shell
python model.py -v VERIFICATION_GROUP [--alu] [--debug]
```

where `VERIFICATION_GROUP` is the index of the group used for verification, and should be an integer between 0-9.

Note that switching between using and not using Alu information (`--alu` flag on and off) causes the model to recompile, which may take up to several minutes.

Parameters of network and training are in `config.ini`. 

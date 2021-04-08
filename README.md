# RNAProt

RNAProt is a computational RBP binding site prediction framework based on recurrent neural networks (RNNs). Conceived as an end-to-end method, RNAProt includes all necessary functionalities, from dataset generation over model training to the evaluation of binding preferences and binding site prediction. Various input types and features are supported, accompanied by comprehensive statistics and visualizations to inform the user about datatset characteristics and learned model properties.


## Table of contents

- [The RNAProt framework](#the-rnaprot-framework)
- [Installation](#installation)
    - [Conda](#conda)
    - [Conda package installation](#conda-package-installation)
    - [Manual installation](#manual-installation)
- [Test runs](#test-runs)
    - [Test example with FASTA sequences as input](#test-example-with-fasta-sequences-as-input)
    - [Test example with genomic regions as input](#test-example-with-genomic-regions-as-input)
    - [Test example with additional features](#test-example-with-additional-features)
- [Documentation](#documentation)
    - [Program modes](#program-modes)
    - [Supported features](#supported-features)
    - [Inputs](#inputs)
    - [Outputs](#outputs)


## The RNAProt framework


RNAProt utilizes RBP binding sites identified by CLIP-seq and related protocols to train an RNN-based model. The model is then used to predict new binding sites on given input RNA sequences. The following figure illustrates the RNAProt framework and its general workflow:


<img src="docs/framework_overview.png" alt="RNAProt framework overview"
	title="RNAProt framework overview" width="550" />


Yellow boxes mark necessary framework inputs, blue boxes the five program modes of RNAProt, and green boxes the framework outputs. Arrows show the dependencies between inputs, modes, and outputs. RNAProt accepts RBP binding sites in FASTA or BED format (transcript or genomic regions). The latter one also requires a genomic sequence file (.2bit format) and a genomic annotations file (GTF format).
Additional inputs are available, depending on the binding site input type as well as the selected features. For more details on inputs, modes, supported features, and outputs, see the [Documentation](#documentation).



## Installation

RNAProt was tested on Ubuntu (18.04 LTS), with Nvidia driver >=440, CUDA >=10, and various Nvidia graphics cards (RTX 2080 Ti, RTX 2070, GTX 1060, GTX 1030). We thus assume that you have a similar system available and running. While RNAProt runs fine without a dedicated GPU, we definitely recommend having an Nvidia graphics card with CUDA support for speeding up model training (specifically we recommend a >= GTX 1060 or a similar newer model, with >= 4 GB RAM). Regarding main memory, we recommend at least 8 GB RAM.
In the following we show how to install RNAProt via Conda package (easiest way + recommended), or alternatively manually (not too difficult either). In any case, you first need Conda running on your computer.

### Conda

If you do not have Conda yet, you can e.g. install miniconda, a free + lightweight Conda installer. Get miniconda [here](https://docs.conda.io/en/latest/miniconda.html), choose the Python 3.8 Miniconda3 Linux 64-bit installer and follow the installation instructions. In the end, Conda should be evocable on the command line via (possibly in a different version):

```
$ conda --version
conda 4.9.2
```

### Conda package installation

RNAProt is available as Conda package [here](https://anaconda.org/bioconda/rnaprot). This is the most convenient way to install RNAProt, since Conda takes care of all the dependencies. Note however that the Conda package version might not always be the latest release (but we work hard to not let this happen).

We recommend to create a Conda environment inside which we will then install RNAProt:

```
conda create -n rnaprotenv python=3.8 -c conda-forge -c bioconda
conda activate rnaprotenv
conda install -c bioconda rnaprot
```

NOTE that the bioconda installation only includes the CPU version of RNAProt (no GPU support). If you have a GPU supporting CUDA and want to take advantage of it (very much recommended for fast model training!), just install in addition:

```
conda install -c conda-forge pytorch-gpu=1.8
```

Now RNAProt should be available inside the environment:


```
rnaprot -h
```

Finally, if you have a compatible GPU, we want to check whether the GPU (CUDA and Nvidia CUDA Compiler nvcc) is available for RNAProt:

```
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

In our test case this delivers:

```
$ python -c "import torch; print(torch.__version__)"
1.7.1
$ python -c "import torch; print(torch.cuda.is_available())"
True
$ python -c "import torch; print(torch.version.cuda)"
10.2
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

This is great news, meaning that we can RNAProt with GPU support.



### Manual installation

To manually install RNAProt, we first create a Conda environment (as described [above](#conda)). Once inside the environment, we need to install the following dependencies for the GPU version (GPU with CUDA support required):

```
conda install -c conda-forge pytorch-gpu=1.8
conda install -c conda-forge seaborn
conda install -c bioconda viennarna
conda install -c bioconda bedtools
conda install -c bioconda logomaker
conda install -c conda-forge hpbandster
conda install -c conda-forge markdown=3.2.2
conda install -c conda-forge plotly
conda install -c conda-forge scikit-learn
conda install -c bioconda ushuffle
conda install -c bioconda ucsc-twobitinfo
conda install -c bioconda ucsc-twobittofa
conda install -c bioconda ucsc-bigwigaverageoverbed
```

If you don't have a CUDA supporting GPU (and you're not planning on getting one any time soon either), you don't need to install the additional GPU dependencies. To install pyTorch without GPU support, simply exchange the above pytorch call with:

```
conda install -c conda-forge pytorch-cpu=1.8
```

Concerning version numbers, RNAProt was tested with the following versions: pytorch=1.8.0, seaborn=0.11.1, viennarna=2.4.17, bedtools=2.30.0, logomaker=0.8, hpbandster=0.7.4, markdown=3.2.2, plotly=4.14.3, and scikit-learn=0.24.1.


Finally, to install RNAProt, we simply clone the repository and execute the installation script inside the folder:

```
git clone https://github.com/BackofenLab/RNAProt.git
cd RNAProt
python -m pip install . --ignore-installed --no-deps -vv
```

Now we can run RNAProt from any given folder (just remember to re-activate the environment once you open a new shell):

```
rnaprot -h
```


## Test runs

Once installed, we can do some quick test runs. 


### Test example with FASTA sequences as input

We first train a sequence model, using a provided set of positive and negative FASTA sequences sampled from the PARCLIP PUM2 dataset (3,000 positives, 3,000 negatives, all sequences with length 81 nt). In the following we will mainly use default parameters, but note that there are many options available for each program mode. To learn more about the mode options, refer to the [Documentation](#documentation), or simply list all mode options, e.g. for `rnaprot train`, by typing:

```
rnaprot train -h
```

Before training a model, we need to generate an RNAProt training dataset. For this we go to the cloned repository folder (clone + enter if not already there), and use the FASTA sequences supplied in the `test` folder as training data. To get training set statistics, we also enable `--report`:


```
git clone https://github.com/BackofenLab/RNAProt.git
cd RNAProt
rnaprot gt --in test/PUM2_PARCLIP.positives.fa --neg-in test/PUM2_PARCLIP.negatives.fa --out PUM2_PARCLIP_gt_out --report
```

We can then take a look at the `report.rnaprot_gt.html` inside `test_gt_out`, informing us about similarities and differences between the positive and negative set. The content of the HTML report depends on selected features (e.g. structure, conservation scores, region annotations), and the input type given to `rnaprot gt` (FASTA sequences, genomic sites BED, or transcript sites BED). Here for example we can compare k-mer statistics of the positive and negative set, observing that the positives tend to contain more AA, UU, and AU repeat sites. This likely also contributes to the lower sequence complexity of the postive set.


Next we train a model on the created dataset, using the default hyperparameters. For this we simply run `rnaprot train` with the `rnaprot gt` output folder as input. We also enable `--verbose-train`, to see the learning progress over the number of epochs:

```
rnaprot train --in PUM2_PARCLIP_gt_out --out PUM2_PARCLIP_train_out --verbose-train
```

In the end we get a summary for the trained model, e.g. reporting the model validation AUC, the training runtime, and set hyperparameters. Note that if you want to obtain the generalization peformance of the model on the given dataset, you need to run `rnaprot train` in cross validation mode (10-fold by default) by adding `--cv`:

```
rnaprot train --in PUM2_PARCLIP_gt_out --out PUM2_PARCLIP_cv_train_out --cv --verbose-train
```

To visualize what our just-trained model has learned, we next run `rnaprot eval`, which requires both the `rnaprot gt` and `rnaprot train` output folders:

```
rnaprot eval --gt-in PUM2_PARCLIP_gt_out --train-in PUM2_PARCLIP_train_out --out PUM2_PARCLIP_eval_out
```

This will plot a sequence logo informing us about global preferences, as well as profiles for the top 25 scoring sites (default setting). The profiles contain the saliency map and single mutations track, giving us an idea what local information the model regards as important for each of the 25 sites. As with the other modes, more options are available (e.g. `--report` for additional statistics, comparing two models, or specifying motif sizes and which profiles to plot).


Now that we have a model, we naturally want to use it for prediction. For this we first create a prediction dataset, choosing the lncRNA *NORAD* for window prediction. *NORAD* was shown to act as a [decoy for PUMILIO proteins](https://doi.org/10.1016/j.cell.2015.12.017) (PUM1/PUM2). We therefore use its provided FASTA sequence as input:

```
rnaprot gp --in test/NORAD_lncRNA.fa --train-in PUM2_PARCLIP_train_out --out PUM2_PARCLIP_gp_out --report
```

Note that the input can be any number of sequences, genomic regions, or transcript regions (also see examples below).

By default, RNAProt predicts whole sites, i.e., we would get one score returned for the whole lncRNA. To run the window prediction, we use `--mode 2`, and also plot the top window profiles containing the reported peak regions:

```
rnaprot predict --in PUM2_PARCLIP_gp_out --train-in PUM2_PARCLIP_train_out --out PUM2_PARCLIP_NORAD_predict_out --mode 2 --plot-top-profiles
```

Now we can take a look at the predicted peak regions (BED, TSV), or observe the profiles just like for `rnaprot eval`. The predicted peak regions are stored in BED format, as well as in a table file with additional information (.tsv). For details on output formats, see the [Documentation](#documentation). Note that while model prediction itself is very fast, plotting (especially getting the single mutation infos) takes some time for each peak region. So if you predict on a large set of input sites or sequences, you might want to disable plotting (or just exercise patience and wait). Knowing this, you can also predict only on certain input sites or a small subset (e.g. site_id1, site_id2), by adding `--site-id site_id1 site_id2`.


### Test example with genomic regions as input

To create datasets based on genomic or transcript regions, we first need to download two additional files. Specifically, we need a GTF file (containing genomic annotations), as well as a .2bit file (containing the genomic sequences). Note that we used Ensembl GTF files to test RNAProt, and therefore recommend using these. You can find them [here](http://www.ensembl.org/info/data/ftp/index.html) for many model organisms. The .2bit genome file we will download from [UCSC](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips). For this example, we choose the human genome + annotations (hg38 assembly):

```
wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.2bit
wget http://ftp.ensembl.org/pub/release-103/gtf/homo_sapiens/Homo_sapiens.GRCh38.103.gtf.gz
```

Unfortunately, we sometimes experienced cases where the GTF file was not fully downloaded. You can check this by browsing the file with:

```
less Homo_sapiens.GRCh38.103.gtf.gz
```

We would expect something like this appearing as first rows:

```
#!genome-build GRCh38.p13
#!genome-version GRCh38
#!genome-date 2013-12
#!genome-build-accession NCBI:GCA_000001405.28
#!genebuild-last-updated 2020-08
```

If the output is cryptic instead, you need to do it again. Next we download some genomic RBP binding regions identified by eCLIP from [ENCODE](https://www.encodeproject.org/). The ENCODE website contains a huge collection of eCLIP datasets for various RBPs. For this example, we again download PUM2 binding sites, choosing the IDR peaks identified by ENCODE's CLIPper pipeline (PUM2 K562 eCLIP dataset ID: ENCFF880MWQ, PUM2 K562 IDR peaks dataset ID: ENCFF880MWQ). We unzip it and change the format to 6-column BED which RNAProt likes best:

```
wget https://www.encodeproject.org/files/ENCFF880MWQ/@@download/ENCFF880MWQ.bed.gz
gunzip -c ENCFF880MWQ.bed.gz | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$7"\t"$6}' > PUM2_K562_IDR_peaks.bed
```

Note that we move the log2 fold change value from column 7 (original file) to column 5, which is used by RNAProt to filter and select sites in case of overlaps. By default, `rnaprot gt` removes overlapping sites by selecting only the highest-scoring site from a set of overlapping sites. To disable this, set `--allow-overlaps`. If there are no column 5 scores given (or all the same), filtering of overlaps is disabled by default. Moreover, positive sites that do not overlap with gene regions are by default filtered out too. To disable this, set `--no-gene-filter`.

Next we create a training dataset, by supplying the downloaded GTF and .2bit file:

```
rnaprot gt --in PUM2_K562_IDR_peaks.bed --out PUM2_K562_IDR_gt_out --gtf Homo_sapiens.GRCh38.103.gtf.gz --gen hg38.2bit --report
```

Thanks to the given GTF file, the HTML report will now also include information on target gene regions and biotypes. Note that by default, `rnaprot gt` centers the input BED regions, and extends them on both sides by the set `--seq-ext` (by default 40). If you want to keep the original site lengths, set `--mode 2 --seq-ext 0`. In this case, you might also want to filter the `--in` sites by `--max-len` or `--min-len`, e.g. `--max-len 100`. Of course you can also extend the original sites, e.g. by setting `--mode 2 --seq-ext 10`. Alternatively, you can set `--mode 3` to use the region upstream ends and extend by `--seq-ext`.

Now we can train a model and evaluate it just like in the example above:

```
rnaprot train --in PUM2_K562_IDR_gt_out --out PUM2_K562_IDR_train_out --verbose-train

rnaprot eval --gt-in PUM2_K562_IDR_gt_out --train-in PUM2_K562_IDR_train_out --out PUM2_K562_IDR_eval_out --report
```

For prediction, we could again use the folder generated by `rnaprot gp` from the above FASTA sequences + *NORAD* example. However, since we now have the GTF + genome .2bit file, we can also get its genomic or transcript region directly from these files (no need to download FASTA sequences or a gene BED file). As with `rnaprot gt`, `rnaprot gp` accepts sequences, genomic regions, or transcript regions as input. To get its genomic or transcript region, we just need its gene or transcript ID (as long as it is in the GTF file), and then use one of the two helper scripts installed together with RNAProt:

```
gtf_extract_gene_regions.py --ids ENSG00000260032 --gtf Homo_sapiens.GRCh38.103.gtf.gz --out ENSG00000260032.bed

gtf_extract_transcript_regions.py --ids ENST00000565493 --gtf Homo_sapiens.GRCh38.103.gtf.gz --out ENST00000565493.bed
```

The two scripts also accept > 1 ID (either on the command line or more practically given as a file with one ID per row). In the case of *NORAD*, both transcript and gene region have the same length, since *NORAD* contains no introns and only one annotated isoform:

```
$ cat ENSG00000260032.bed
chr20	36045617	36051018	ENSG00000260032	0	-
$ cat ENST00000565493.bed
ENST00000565493	0	5401	ENST00000565493	0	+
```

We can now use any of the two as input to `rnaprot gp`. As one would expect, extracting and providing the transcript region will result in predictions only on the transcript sequence (always excluding introns), while providing the gene region will result in predictions on the whole gene region (usually including introns). In general, we can specify any genomic or transcript (sub)region for prediction, as long as it is annotated in the GTF file. For now we will use the gene region, on which `rnaprot predict` will then predict and return peak regions directly with genomic coordinates (also contained inside the profiles for easier orientation):

```
rnaprot gp --in ENSG00000260032.bed --out NORAD_lncRNA_gene_gp_out --gtf Homo_sapiens.GRCh38.103.gtf.gz --gen hg38.2bit --train-in PUM2_K562_IDR_train_out

rnaprot predict --in NORAD_lncRNA_gene_gp_out --train-in PUM2_K562_IDR_train_out --out PUM2_K562_NORAD_predict_out --mode 2 --plot-top-profiles
```

Note that in this example we did not filter out sites from `PUM2_K562_IDR_peaks.bed` that overlap with the *NORAD* gene region prior to training. These sites indeed exist, as we can see in the `rnaprot gt` HTML report (Target region overlap statistics), so for an unbiased prediction we need remove them first. For this we use bedtools intersectBed, and again run `rnaprot gt`, `rnaprot train`, and `rnaprot predict`:

```
intersectBed -a PUM2_K562_IDR_peaks.bed -b ENSG00000260032.bed -v -s > PUM2_K562_IDR_peaks_no2norad.bed 
rnaprot gt --in PUM2_K562_IDR_peaks_no2norad.bed --out PUM2_K562_IDR_no2norad_gt_out --gtf Homo_sapiens.GRCh38.103.gtf.gz --gen hg38.2bit --report
rnaprot train --in PUM2_K562_IDR_no2norad_gt_out --out PUM2_K562_IDR_no2norad_train_out --verbose-train
rnaprot predict --in NORAD_lncRNA_gene_gp_out --train-in PUM2_K562_IDR_no2norad_train_out --out PUM2_K562_IDR_no2norad_predict_out --mode 2 --plot-top-profiles
```



### Test example with additional features

RNAProt supports various additional position(nucleotide)-wise features to learn from, such as secondary structure, region annotations (including user-defined ones), or conservation scores (see [Documentation](#documentation) for details). For this we have to specify what features to include in `rnaprot gt` and `rnaprot gp`, and depending on the feature also provide additional files. For model training (`rnaprot train`) we can then specify what features to use for training, from the features included in `rnaprot gt`. This has the advantage that features need to be extracted or computed only once, and that various feature combinations and parameter settings can be with `rnaprot train`.

In this example, we want to include secondary structure on top of the sequence information. We will again use a provided dataset, containing 2,274 potential Roquin binding sites (also termed CDEs for constitutive decay elements) from [Braun et al. 2018](https://doi.org/10.1093/nar/gky908). 
The CDEs were predicted using a biologically verified consensus structure consisting of a 6-8 bp long stem capped with a YRN (Y: C or U, R: A or G, N: any base) tri-nucleotide loop. We also note that the sequence conservation is rather low (specifically for the stem portion), making it an ideal test case for the benefits of including secondary structure information. We first generate a training set, by enabling structure calculation (`--str`) and using the CDE sites provided in the cloned repository folder:

```
rnaprot gt --in test/CDE_sites.bed --out CDE_sites_gt_out --gtf Homo_sapiens.GRCh38.103.gtf.gz --gen hg38.2bit --allow-overlaps --no-gene-filter --str --report
```

Structure calculation can be further customized by changing the RNAplfold parameters (`--plfold-u 3 --plfold-l 50 --plfold-w 70` by default). Regarding the type of structure information, RNAplfold calculates the probabilities of structural elements for each site position, which are then used as feature channels for training and prediction. For genomic or transcript `--in` sites, RNAProt automatically extends the sites by `--plfold-w` on both sides (or less at transcript ends) for a precise structure calculation.
Whether to use the probabilities or a one-hot encoding can be further specified in training (`--str-mode`, four options). Note that we use `--allow-overlaps` and `--no-gene-filter`, disabling the filtering of sites based on no gene overlap or overlap with other sites. These two options guarantee that all `--in` sites will be part of the generated training set. Now we want to train a model on the generated dataset:

```
rnaprot train --in CDE_sites_gt_out --out CDE_sites_str_train_out --verbose-train --epochs 300
```

Here we increased the maximum number of epochs to 300, since for smaller datasets model performance can sometimes still improve beyond the default 200 epochs. This can be easily monitored with `--verbose-train` enabled. In addition, increasing `--patience` might sometimes be necessary, to prevent model training with the model stuck early on in the training process from stopping (although we experienced this only very rarely).
Also note that if we do not specify what features to use, RNAProt will use all features present in `CDE_sites_gt_out` for training. Thus, to train a sequence-only model, we would need to specify:

```
rnaprot train --in CDE_sites_gt_out --out CDE_sites_onlyseq_train_out --only-seq --verbose-train
```

To create a prediction set for the structure model, we use the UCP3 gene (transcript ID ENST00000314032), for which the authors validated two CDE sites in its 3'UTR (see [Fig.2A](https://doi.org/10.1093/nar/gky908) blue and red hairpin).

```
rnaprot gp --in test/ENST00000314032.fa --train-in CDE_sites_str_train_out --out CDE_sites_str_gp_out
```

Note that `rnaprot gp` automatically detects what features were used for training the model, enabling structure prediction with set parameters for the prediction set generation. Depending on the additional feature, we thus might have to supply additional input files for extracting the respective feature information. This would be the case for conservation scores (`--phastcons`, `--phylop`), or user-defined features (`--feat-in`). After creating the prediction set we run a window prediction on the transcript:


```
rnaprot predict --in CDE_sites_str_gp_out --train-in CDE_sites_str_train_out --out CDE_sites_str_predict_out --mode 2 --plot-top-profiles --thr 1
```

In our case the model successfully predicted the two verified binding sites (all together 4 sites predicted) on the transcript (using threshold level `--thr 1`). The first verified loop is at transcript position 1,371 to 1,373 (loop nucleotides), the second loop from 1,404 to 1,406, with the second hairpin having a higher folding probability and score. To check this, we take a look at the reported peak regions (BED, TSV), or have a look at the plotted profiles, which show the sites annotated with transcript coordinates (or genomic coordinates in case of genomic sites as input) to quickly identify regions of interest. 
The `test/` folder also includes the model we used to predict, which you can easily apply yourself to compare:

```
unzip test/cde_sites_str_model_folder.zip

rnaprot predict --in CDE_sites_str_gp_out --train-in cde_sites_hg38_extlr40_w70l50_str_train_out --out CDE_sites_test_model_str_predict_out --mode 2 --plot-top-profiles --thr 1
```

This also shows how easy it is to share models. Once the model is trained, the `rnaprot train --out` folder can be copied and reused. To zip the train folder you can run:

```
zip my_cde_model.zip CDE_sites_str_train_out/*
```


Note however that predictions between two models can vary, since negative sites generation is random. Moreover, even models trained on the same positive and negative sites are slightly different from one another and thus can lead to slightly different predictions. This is because model training incorporates stochastic processes, like the random initialization of network weights, or the application of dropout.


## Documentation


This documentation provides details on all the RNAProt (version 0.3) framework parts:
[program modes](#program-modes), [supported features](#supported-features), [inputs](#inputs), and [outputs](#outputs).


### Program modes

RNAProt is divided into five different program modes: training set generation (`rnaprot gt`), model training (`rnaprot train`), model evaluation (`rnaprot eval`), prediction set generation (`rnaprot gp`), and model prediction (`rnaprot predict`).


An overview of the modes can be obtained by:


```
$ rnaprot -h
usage: rnaprot [-h] [-v] {train,eval,predict,gt,gp} ...

Modelling RBP binding preferences to predict RPB binding sites.

positional arguments:
  {train,eval,predict,gt,gp}
                        Program modes
    train               Train a binding site prediction model
    eval                Evaluate properties learned from training set
    predict             Predict binding sites (whole sites or profiles)
    gt                  Generate training data set
    gp                  Generate prediction data set

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

```

The following sections describe each mode in more detail.


#### Training set generation

The following command line arguments are available in `rnaprot gt` mode:

```
$ rnaprot gt -h
usage: rnaprot gt [-h] --in str --out str [--gtf str] [--gen str]
                  [--mode {1,2,3}] [--mask-bed str] [--seq-ext int]
                  [--thr float] [--rev-filter] [--max-len int] [--min-len int]
                  [--keep-ids] [--allow-overlaps] [--no-gene-filter]
                  [--neg-comp-thr float] [--neg-factor {2,3,4,5}]
                  [--keep-add-neg] [--neg-in str] [--shuffle-k {1,2,3}]
                  [--report] [--theme {1,2}] [--eia] [--eia-ib] [--eia-n]
                  [--eia-all-ex] [--tr-list str] [--phastcons str]
                  [--phylop str] [--tra] [--tra-codons] [--tra-borders]
                  [--rra] [--str] [--plfold-u int] [--plfold-l int]
                  [--plfold-w int] [--feat-in str] [--feat-in-1h]
                  [--feat-in-norm]

optional arguments:
  -h, --help            show this help message and exit
  --gtf str             Genomic annotations GTF file (.gtf or .gtf.gz)
  --gen str             Genomic sequences .2bit file
  --mode {1,2,3}        Define mode for --in BED site extraction. (1) Take the
                        center of each site, (2) Take the complete site, (3)
                        Take the upstream end for each site. Note that --min-
                        len applies only for --mode 2 (default: 1)
  --mask-bed str        Additional BED regions file (6-column format) for
                        masking negatives (e.g. all positive RBP CLIP sites)
  --seq-ext int         Up- and downstream sequence extension length of sites
                        (site definition by --mode) (default: 40)
  --thr float           Minimum site score (--in BED column 5) for filtering
                        (assuming higher score == better site) (default: None)
  --rev-filter          Reverse --thr filtering (i.e. the lower the better,
                        e.g. for p-values) (default: False)
  --max-len int         Maximum length of --in sites (default: 300)
  --min-len int         Minimum length of --in sites (only effective for
                        --mode 2). If length < --min-len, take center and
                        extend to --min-len. Use uneven numbers for equal up-
                        and downstream extension (default: 21)
  --keep-ids            Keep --in BED column 4 site IDs. Note that site IDs
                        have to be unique (default: False)
  --allow-overlaps      Do not select for highest-scoring sites in case of
                        overlapping sites (default: False)
  --no-gene-filter      Do not filter positives based on gene coverage (gene
                        annotations from --gtf) (default: False)
  --neg-comp-thr float  Sequence complexity (Shannon entropy) threshold for
                        filtering random negative regions (default: 0.5)
  --neg-factor {2,3,4,5}
                        Determines number of initial random negatives to be
                        extracted (== --neg-factor n times # positives)
                        (default: 2)
  --keep-add-neg        Keep additional negatives (# controlled by --neg-
                        factor) instead of outputting same numbers of positive
                        and negative sites (default: False)
  --neg-in str          Negative genomic or transcript sites in BED (6-column
                        format) or FASTA format (unique IDs required). Use
                        with --in BED/FASTA. If not set, negatives are
                        generated by shuffling --in sequences (if --in FASTA)
                        or random selection of genomic or transcript sites (if
                        --in BED)
  --shuffle-k {1,2,3}   Supply k for k-nucleotide shuffling of --in sequences
                        to generate negative sequences (if no --neg-in
                        supplied) (default: 2)
  --report              Output an .html report providing various training set
                        statistics and plots (default: False)
  --theme {1,2}         Set theme for .html report (1: palm beach, 2: midnight
                        sunset) (default: 1)

required arguments:
  --in str              Genomic or transcript RBP binding sites file in BED
                        (6-column format) or FASTA format. If --in FASTA, only
                        --str is supported as additional feature. If --in BED,
                        --gtf and --gen become mandatory
  --out str             Output training data folder (== input folder to
                        rnaprot train)

additional annotation arguments:
  --eia                 Add exon-intron annotations to genomic regions
                        (default: False)
  --eia-ib              Add intron border annotations to genomic regions (in
                        combination with --eia) (default: False)
  --eia-n               Label regions not covered by intron or exon regions as
                        N instead of labelling them as introns (I) (in
                        combination with --eia) (default: False)
  --eia-all-ex          Use all annotated exons in --gtf file, instead of
                        exons of most prominent transcripts or exon defined by
                        --tr-list. Set this and --tr-list will be effective
                        only for --tra (default: False)
  --tr-list str         Supply file with transcript IDs (one ID per row) for
                        exon-intron labeling (using the corresponding exon
                        regions from --gtf). By default, exon regions of the
                        most prominent transcripts (automatically selected
                        from --gtf) are used (default: False)
  --phastcons str       Genomic .bigWig file with phastCons conservation
                        scores to add as annotations
  --phylop str          Genomic .bigWig file with phyloP conservation scores
                        to add as annotations
  --tra                 Add transcript region annotations (5'UTR, CDS, 3'UTR,
                        None) to genomic and transcript regions (default:
                        False)
  --tra-codons          Add start and stop codon annotations to genomic or
                        transcript regions (in combination with --tra)
                        (default: False)
  --tra-borders         Add transcript and exon border annotations to
                        transcript regions (in combination with --tra)
                        (default: False)
  --rra                 Add repeat region annotations for genomic or
                        transcript regions retrieved from --gen .2bit
                        (default: False)
  --str                 Add secondary structure probabilities feature
                        (calculate with RNAplfold) (default: False)
  --plfold-u int        RNAplfold -u parameter value (default: 3)
  --plfold-l int        RNAplfold -L parameter value (default: 50)
  --plfold-w int        RNAplfold -W parameter value (default: 70)
  --feat-in str         Provide tabular file with additional position-wise
                        genomic region features (infos and paths to BED files)
                        to add
  --feat-in-1h          Use one-hot encoding for all additional position-wise
                        features from --feat-in table, ignoring type
                        definitions in --feat-in table (default: False)

```

Note that if genomic or transcript regions are supplied via `--in`, `--gen` and `--gtf` become required arguments.



#### Model training

The following command line arguments are available in `rnaprot train` mode:

```
$ rnaprot train -h
usage: rnaprot train [-h] --in IN_FOLDER --out OUT_FOLDER [--only-seq]
                     [--use-phastcons] [--use-phylop] [--use-eia] [--use-tra]
                     [--use-rra] [--use-str] [--str-mode {1,2,3,4}]
                     [--use-add-feat] [--cv] [--cv-k {5,10}]
                     [--val-size float] [--add-test] [--test-ids str]
                     [--keep-order] [--plot-lc] [--verbose-train]
                     [--force-cpu] [--epochs int] [--patience int]
                     [--batch-size int] [--lr float] [--weight-decay float]
                     [--n-rnn-layers int] [--n-hidden-dim int] [--dr float]
                     [--n-fc-layers {1,2}] [--model-type {1,2,3,4}] [--embed]
                     [--embed-dim int] [--run-bohb] [--bohb-n int]
                     [--bohb-min-budget int] [--bohb-max-budget int]
                     [--bohb-workers int] [--verbose-bohb]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --in IN_FOLDER        Input training data folder (output of rnaprot gt)
  --out OUT_FOLDER      Model training results output folder

feature definition arguments:
  --only-seq            Use only sequence feature. By default all features
                        present in --in are used (default: False)
  --use-phastcons       Add phastCons conservation scores. Set --use-xxx to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)
  --use-phylop          Add phyloP conservation scores. Set --use-xxx to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)
  --use-eia             Add exon-intron annotations. Set --use-xxx to define
                        which features to add on top of sequence feature (by
                        default all --in features are used)
  --use-tra             Add transcript region annotations. Set --use-xxx to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)
  --use-rra             Add repeat region annotations. Set --use-xxx to define
                        which features to add on top of sequence feature (by
                        default all --in features are used)
  --use-str             Add secondary structure features (type defined by
                        --str-mode). Set --use-xxx to define which features to
                        add on top of sequence feature (by default all --in
                        features are used)
  --str-mode {1,2,3,4}  Define secondary structure feature representation: 1)
                        use probabilities of five structural elements
                        (E,I,H,M,S) 2) same as 1) but encoded as one-hot
                        (element with highest probability gets 1, others 0) 3)
                        use unpaired probabilities 4) same as 3) but encoded
                        as one-hot (default: 1)
  --use-add-feat        Add additional feature annotations. Set --use-xxx to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)

model definition arguments:
  --cv                  Run cross validation in combination with set
                        hyperparameters to evaluate model generalization
                        performance (default: False)
  --cv-k {5,10}         Cross validation k for evaluating generalization
                        performance (use together with --cv) (default: 10)
  --val-size float      Validation set size for training final model as
                        percentage of all training sites. NOTE that if --add-
                        test is set, the test set will have the same size (so
                        if --val-size 0.2, train on 60 percent, validate on 20
                        percent, and test on 20 percent) (default: 0.2)
  --add-test            Use a part of the training set as a test set to
                        evaluate final model. Test set size is controlled by
                        --val-size (default: False)
  --test-ids str        Provide file with test IDs to be used as a test set
                        for testing final model. Test IDs need to be part of
                        --in training set. Not compatible with --add-test,
                        --cv, or --gm-cv
  --keep-order          Use same train-validation(-test) split for each call
                        to train final model. Test split only if --add-test or
                        --test-ids (default: False)
  --plot-lc             Plot learning curves (training vs validation loss) for
                        each tested hyperparameter combination (default:
                        False)
  --verbose-train       Enable verbose output during model training to show
                        performance over epochs (default: False)
  --force-cpu           Run on CPU regardless of CUDA available or not
                        (default: False)
  --epochs int          Maximum number of training epochs (default: 200)
  --patience int        Number of epochs to wait for further improvement on
                        validation set before stopping (default: 30)
  --batch-size int      Gradient descent batch size (default: 50)
  --lr float            Learning rate of optimizer (default: 0.001)
  --weight-decay float  Weight decay of optimizer (default: 0.0005)
  --n-rnn-layers int    Number of RNN layers (default: 1)
  --n-hidden-dim int    Number of RNN layer dimensions (default: 32)
  --dr float            Rate of dropout applied after RNN layers (default:
                        0.5)
  --n-fc-layers {1,2}   Number of fully connected layers following RNN layers
                        (default: 1)
  --model-type {1,2,3,4}
                        RNN model type to use. 1: GRU, 2: LSTM, 3: biGRU, 4:
                        biLSTM (default: 1)
  --embed               Use embedding layer for sequence feature, instead of
                        one-hot encoding (default: False)
  --embed-dim int       Dimension of embedding layer (default: 10)
  --run-bohb            Use BOHB to run a hyperparameter optimization. NOTE
                        that this will overwrite set hyperparameters, and
                        trains the final model with the found best
                        hyperparameter setting. ALSO NOTE that this will take
                        some time (!) (default: False)
  --bohb-n int          Number of BOHB iterations (default: 80)
  --bohb-min-budget int
                        BOHB minimum budget (default: 5)
  --bohb-max-budget int
                        BOHB maximum budget (default: 40)
  --bohb-workers int    Number of BOHB worker threads for local multi-core
                        parallel computing (default: 1)
  --verbose-bohb        Enable verbose output for BOHB hyperparameter
                        optimization. By default only warnings are print out
                        (default: False)

```


#### Model evaluation

The following command line arguments are available in `rnaprot eval` mode:

```
$ rnaprot eval -h 
usage: rnaprot eval [-h] --train-in IN_TRAIN_FOLDER --gt-in IN_GT_FOLDER --out
                    OUT_FOLDER [--nr-top-profiles int]
                    [--lookup-profile LIST_LOOKUP_IDS [LIST_LOOKUP_IDS ...]]
                    [--bottom-up]
                    [--nr-top-sites LIST_NR_TOP_SITES [LIST_NR_TOP_SITES ...]]
                    [--motif-size LIST_MOTIF_SIZES [LIST_MOTIF_SIZES ...]]
                    [--report] [--add-train-in str] [--theme {1,2}]
                    [--plot-format {1,2}]

optional arguments:
  -h, --help            show this help message and exit
  --nr-top-profiles int
                        Specify number of top predicted sites to plot profiles
                        for (default: 25)
  --lookup-profile LIST_LOOKUP_IDS [LIST_LOOKUP_IDS ...]
                        Provide site ID(s) for which to plot the feature
                        profile in addition to --nr-top-profiles (e.g.
                        --lookup-profile site_id1 site_id2 ). Site ID needs to
                        be in positive set from --gt-in
  --bottom-up           Plot bottom profiles as well (default: False)
  --nr-top-sites LIST_NR_TOP_SITES [LIST_NR_TOP_SITES ...]
                        Specify number(s) of top-predicted sites used for
                        motif extraction. Provide multiple numbers (e.g. --nr-
                        top-sites 100 200 500) to extract one motif plot from
                        each site set (default: 200)
  --motif-size LIST_MOTIF_SIZES [LIST_MOTIF_SIZES ...]
                        Motif size(s) (widths) for extracting and plotting
                        motifs. Provide multiple sizes (e.g. --motif-size 5 7
                        9) to extract a motif for each size (default: 7)
  --report              Generate an .html report containing various additional
                        statistics and plots (default: False)
  --add-train-in str    Second model training folder (output of rnaprot train)
                        for comparing prediction scores of both models on
                        --gt-in positive dataset. Note that if dataset
                        features of the two models are not identical,
                        comparison might be less informative
  --theme {1,2}         Set theme for .html report (1: palm beach, 2: midnight
                        sunset) (default: 1)
  --plot-format {1,2}   Plotting format of motifs and profiles (does not
                        affect plots generated for --report). 1: png, 2: pdf
                        (default: 1)

required arguments:
  --train-in IN_TRAIN_FOLDER
                        Input model training folder (output of rnaprot train)
  --gt-in IN_GT_FOLDER  Input training data folder (output of rnaprot gt)
  --out OUT_FOLDER      Evaluation results output folder

```


#### Prediction set generation

The following command line arguments are available in `rnaprot gp` mode:

```
$ rnaprot gp -h
usage: rnaprot gp [-h] --in str --train-in str --out str [--gtf str]
                  [--gen str] [--mode {1,2,3}] [--seq-ext int] [--gene-filter]
                  [--report] [--theme {1,2}] [--tr-list str] [--eia-all-ex]
                  [--phastcons str] [--phylop str] [--feat-in str]

optional arguments:
  -h, --help       show this help message and exit
  --gtf str        Genomic annotations GTF file (.gtf or .gtf.gz)
  --gen str        Genomic sequences .2bit file
  --mode {1,2,3}   Define mode for --in BED site extraction. (1) Take the
                   center of each site, (2) Take the complete site, (3) Take
                   the upstream end for each site. Use --seq-ext to extend
                   center sites again (default: 2)
  --seq-ext int    Up- and downstream sequence extension length of --in sites
                   (if --in BED, site definition by --mode) (default: False)
  --gene-filter    Filter --in sites based on gene coverage (gene annotations
                   from --gtf) (default: False)
  --report         Output an .html report providing various training set
                   statistics and plots (default: False)
  --theme {1,2}    Set theme for .html report (1: palm beach, 2: midnight
                   sunset) (default: 1)

required arguments:
  --in str         Genomic or transcript RBP binding sites file in BED
                   (6-column format) or FASTA format. If --in FASTA, only
                   --str is supported as additional feature. If --in BED,
                   --gtf and --gen become mandatory
  --train-in str   Training input folder (output folder of rnaprot train) to
                   extract the same features for --in sites which were used to
                   train the model (info stored in --train-in folder)
  --out str        Output prediction dataset folder (== input folder to
                   rnaprot predict)

additional annotation arguments:
  --tr-list str    Supply file with transcript IDs (one ID per row) for exon-
                   intron labeling (using the corresponding exon regions from
                   --gtf). By default, exon regions of the most prominent
                   transcripts (automatically selected from --gtf) are used
                   (default: False)
  --eia-all-ex     Use all annotated exons in --gtf file, instead of exons of
                   most prominent transcripts defined by --tr-list.
                   Set this and --tr-list will be effective only for --tra.
                   NOTE that by default --eia-all-ex is disabled, even if
                   --train-in model was trained with --eia-all-ex (default:
                   False)
  --phastcons str  Genomic .bigWig file with phastCons conservation scores to
                   add as annotations
  --phylop str     Genomic .bigWig file with phyloP conservation scores to add
                   as annotations
  --feat-in str    Provide tabular file with additional position-wise genomic
                   region features (infos and paths to BED files) to add. BE
                   SURE to use the same file as used for generating the
                   training dataset (rnaprot gt --feat-in) for training the
                   model from --train-in!

```

Note that `rnaprot gp` will try to reuse file paths used for training the `--train-in` model. This includes `--tr-list`, `--phastcons`, `--phylop`, and `--feat-in`. These can be overwritten by setting providing the paths on the command line.



#### Model prediction

The following command line arguments are available in `rnaprot predict` mode:

```
$ rnaprot predict -h
usage: rnaprot predict [-h] --in IN_FOLDER --train-in IN_TRAIN_FOLDER --out
                       str [--mode {1,2}] [--plot-top-profiles]
                       [--plot-format {1,2}] [--thr {1,2,3}]
                       [--site-id LIST_SITE_IDS [LIST_SITE_IDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --mode {1,2}          Define prediction mode. (1) predict whole sites, (2)
                        predict binding sites on longer sequences using moving
                        window predictions (default: 1)
  --plot-top-profiles   Plot top window profiles (default: False)
  --plot-format {1,2}   Plotting format of top window profiles. 1: png, 2: pdf
                        (default: 1)
  --thr {1,2,3}         Define site score threshold level for reporting peak
                        regions in --mode 2 (window prediction). 1: relaxed,
                        2: standard, 3: strict (default: 2)
  --site-id LIST_SITE_IDS [LIST_SITE_IDS ...]
                        Provide site ID(s) on which to predict (e.g. --site-id
                        site_id1 site_id2). By default predict on all --in
                        sites

required arguments:
  --in IN_FOLDER        Input prediction data folder (output of rnaprot gp)
  --train-in IN_TRAIN_FOLDER
                        Input model training folder containing model file and
                        parameters (output of rnaprot train)
  --out str             Prediction results output folder

```

### Supported features

RNAProt supports the following position-wise features, which can be utilized for training and prediction in addition to the sequence feature: secondary structure information (structural element probabilities), conservation scores (phastCons and phyloP), exon-intron annotation, transcript region annotation, repeat region annotation, and also user-defined region annotations. The following table lists the features available for each of the three input types (FASTA sequences, genomic regions BED, transcript regions BED):


|   Feature       | Sequences | Genomic regions | Transcript regions |
| :--------------: | :--------------: | :--------------: | :--------------: |
| **structure**    | YES | YES | YES |
| **conservation scores**    | NO | YES | YES |
| **exon-intron annotation**    | NO | YES | NO |
| **transcript region annotation**    | NO | YES | YES |
| **repeat region annotation**    | NO | YES | YES |
| **user-defined**    | NO | YES | YES |


#### Secondary structure information

RNAProt can include structure information in the form of unpaired probabilities for different loop contexts (probabilities for the nucleotide being paired or inside external, hairpin, internal or multi loops). The probabilities are calculated using the ViennaRNA Python 3 API (ViennaRNA 2.4.17) and RNAplfold with its sliding window approach, with user-definable parameters (by default these are window size = 70, maximum base pair span length = 50, and probabilities for regions of length u = 3, `--plfold-u 3 --plfold-l 50 --plfold-w 70`). In order to include structure information, `--str` has to be set when calling `rnaprot gt`. For training, `rnaprot train` by default uses all features it can find in the training set folder. To specify specific features for training, one can add `--use-str` (or in general --use-xxx). `rnaprot train` also offers four different modes for including the structure information (`--str-mode`). Here one can select between using all five context probabilities, using only paired and unpaired probabilities, or use the first two but encoded as one-hot.


#### Conservation scores

RNAProt supports two scores measuring evolutionary conservation (phastCons and phyloP). Conservation scores can be downloaded from the UCSC website, e.g. using the phastCons and phyloP scores generated from multiple sequence alignments of 99 vertebrate genomes to the human genome (hg38, [phastCons100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw) and [phyloP100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw) datasets). RNAProt accepts scores in bigWig (.bw) format (modes `rnaprot gt` and `rnaprot gp`, options `--phastcons` and `--phylop`). 
Note that `rnaprot gp` reuses the set file paths used for training (`rnaprot train`), as long as the file paths are still valid. If not, it will complain, or you can just overwrite the paths by setting `--phastcons` and `--phylop`.
To assign conservation scores to transcript regions, RNAProt first maps the transcript regions to the genome, using the provided GTF file. To use the conservation features for `rnaprot train`, set `--use-phastcons` or `--use-phylop`. Otherwise, if no other feature is specified, `rnaprot train` will train on all present features.




#### Exon-intron annotation

Exon-intron annotation in the form of one-hot encoded exon and intron labels can also be added to the node feature vectors.
Labels are assigned to each binding site position by taking a set of genomic exon regions and overlapping it with the genomic binding sites using bedtools. To unambiguously assign labels, RNAProt by default uses the most prominent isoform for each gene. The most prominent isoform for each gene gets selected through hierarchical filtering of the transcript information present in the input GTF file (tested with GTF files from [Ensembl](http://www.ensembl.org/info/data/ftp/index.html)): given that the transcript is part of the GENCODE basic gene set, RNAProt selects transcripts based on their transcript support level (highest priority), and by transcript length (longer isoform preferred). The extracted isoform exons are then used for region type assignment.
Note that this feature is only available for genomic regions, as it is not informative for transcript regions, which would contain only exon labels. Optionally, a user-defined isoform list can be supplied (`--tr-list`), substituting the list of most prominent isoforms for annotation. Regions not overlapping with introns or exons can also be labelled separately (instead of labelled as intron). In addition, instead of using the most prominent transcripts, `--eia-all-ex` allows the use of all exon regions from the GTF file for exon-intron labelling. For more details see mode options `--eia`, `--eia-ib`, `--eia-n`, and `--eia-all-ex`. To use exon-intron annotations for `rnaprot train`, set `--use-eia`. Otherwise, if no other feature is specified, `rnaprot train` will train a model with all present features.


#### Transcript region annotation

Similarly to exon-intron annotations, binding regions can be labelled based on their overlap with transcript regions. Labels are assigned based on UTR or CDS region overlap (5'UTR, CDS, 3'UTR, None), by taking the isoform information in the input GTF file. Again the list of most prominent isoforms is used for annotation, or alternatively a list of user-defined isoforms (`--tr-list`). Additional annotation options include start and stop codon or transcript and exon border labelling. For more details see mode options `--tra`, `--tra-codons`, and `--tra-borders`.
To use transcript region annotations for `rnaprot train`, set `--use-tra`. Otherwise, if no other feature is specified, `rnaprot train` will train a model with all present features.


#### Repeat region annotation

Repeat region annotation (`--rra` option) can also be added to the binding regions, analogously to other region type annotations. This information is derived directly from the 2bit formatted genomic sequence file supplied by `--gen`, where repeat regions identified by RepeatMasker and Tandem Repeats Finder are stored in lowercase letters. 
To use repeat region annotations for `rnaprot train`, set `--use-rra`. Otherwise, if no other feature is specified, `rnaprot train` will train a model with all present features.


#### User-defined region annotations

User-defined features in the form of region information (BED) for annotating transcript or genomic input regions can also be supplied. For this `rnaprot gt` and `rnaprot gp` require a table file with a specific format (feature ID, feature type, feature format, BED file path) provided via `--feat-in`. The table format is defined in the [Inputs](#inputs) section below. `rnaprot gt` also offers an option to force one-hot-encoding of all user-defined features (see mode option `--feat-in-1h`).
To utilize user-defined region annotations for `rnaprot train`, set `--use-add-feat`. Otherwise, if no other feature is specified, `rnaprot train` will train a model with all present features.


### Inputs

RNAProt accepts RBP binding sites in FASTA or BED format (transcript or genomic regions). The latter one also requires a genomic sequence file (.2bit format) and a genomic annotations file (GTF format).
Additional input files include BED files (negative sites or regions for masking), conservation scores in bigWig format, transcript lists, or user-defined feature tables.


#### Binding sites

RBP binding sites can be input in three different formats:

- Sequences (FASTA format)
- Genomic regions (6-column BED format)
- Transcript regions (6-column BED format)

The file content should thus look like:

```
$ head -4 sequences.fa
>seq_1
UUCUCACAUUGGCAUAGACAAGAUUGCAUUCACAGGGUCUACUGAGGUUGGAAAGCUUAUC
>seq_2
GGAUCAAAAGAUACAACAGUUAUCAUAUGGCAAGUUGAUCCGGAUACACACCUGCUAAAAC

$ head -4 genomic_sites.bed
chr7	5593730	5593791	gen_site_1	1.24262727654043	+
chr7	138460417	138460478	gen_site_2	1.24262727654043	+
chr7	73477388	73477449	gen_site_3	4.68741211921332	-
chr7	73442440	73442501	gen_site_4	3.65766477581927	-

$ head -4 transcript_sites.bed
ENST00000360876	1387	1448	tr_site_1	2.99156551237271	+
ENST00000325888	3965	4026	tr_site_2	4.51564577094684	+
ENST00000360876	1320	1381	tr_site_3	3.67261511728524	+
ENST00000325888	3092	3153	tr_site_4	3.05759538270791	+

```

Note that RNAProt by default creates unique site IDs. Optionally, the original sequence or site IDs (BED column 4) can be kept (`--keep-ids` in `rnaprot gt`). Also note that BED column 5 contains the site score, which can be used for filtering (`--thr`). In case of p-values, reverse filtering can be enabled with `--rev-filter` (smaller value == better). Filtering by site length is also possible (`--max-len`, `--min-len`). Note that by default, `--in` sites in `rnaprot gt` are filtered based on gene coverage and overlaps with other `--in` sites! To disable these filters, set `--no-gene-filter` and `--allow-overlaps` (see mode options for more details).
BED files with genomic RBP binding regions for example can be downloaded from [ENCODE](https://www.encodeproject.org/) (eCLIP CLIPper peak regions).


#### Genome sequence

The human genome .2bit formatted genomic sequence file can be downloaded [here](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.2bit). This file is used to extract genomic and transcript region sequences, as well as repeat region annotations. For other organisms, have a look [here](https://hgdownload.soe.ucsc.edu/downloads.html). Note that it is easy to generate your own .2bit files from any given FASTA file.


#### Genome annotations

RNAProt was tested with GTF files obtained from Ensembl. See Ensembl's [download page](http://www.ensembl.org/info/data/ftp/index.html) to download the latest GTF file with genomic annotations. Note that RNAProt can directly read from gzipped GTF. The GTF file is used to extract gene region, exon-intron, and transcript region annotations.


#### User-defined region annotations

To supply user-defined region annotations, `rnaprot gt` (`rnaprot gp`) accepts a table file via `--feat-in`. Inside this, each row stores the information of one single feature to be added, beginning with a unique feature ID, the type of feature (C: categorical, N: numerical), the feature format (0: score, 1: probability, 2: p-value), and the path to the feature regions BED file. Assuming we want to use the CDE sites from [above](#test-example-with-additional-features) as a region feature for annotation, we would create a text file with the following content:


```
$ cat add_feat.in
CDE	N	1	test/CDE_sites.bed
```

Note that the columns have to be tab-separated. Here we put "CDE" as the feature ID, "N" as feature type for numerical (since `CDE_sites.bed` BED file column 5 contains probabilities which we want to use for annotation), and "1" to indicate that the values are probabilities. Last but not least, we specify the path to the BED file in column 4. Now all `--in` site positions  (`rnaprot gt`) which overlap with `CDE_sites.bed` get the numerical value (BED column 5 value) of the overlapping region inside `CDE_sites.bed` assigned. If we want a one-hot encoding instead, we need to specify:

```
$ cat add_feat.in
CDE	C	0	test/CDE_sites.bed
```

Alternatively, we can also run `rnaprot gt` with `--feat-in-1h`, to turn all `add_feat.in` features into one-hot encoding. This means that overlapping site positions get a "1" assigned, and non-overlapping a "0", just like for the standard region annotations (exon-intron regions, transcript regions, repeat regions). Note that `rnaprot gp` reuses the set `add_feat.in` file used for training (`rnaprot train`), as long as the file path is valid. In general, we suggest to use either one-hot encoding (C) or normalized BED column 5 values (e.g. probabilities from 0 to 1). If you set "N" and "2" (column 1 and 2, telling RNAProt that these are p-values), RNAProt will automatically convert them to probabilities, by using 1-p-value for the respective regions. As said we do not recommend using raw column 5 BED scores, since the values are not normalized, which likely will be suboptimal for learning.


#### Additional inputs

Additional input files are (depending on set mode):

- BED files (negative sites or regions for masking, 6-column BED format as described)
- A transcript ID list file for exon-intron or transcript region annotation
- Conservation scores in bigWig format

The transcript IDs list file (`--tr-list` option, for `rnaprot gt` and `rnaprot gp`) has the following format (one ID per row):

```
$ head -5 tr_list.in
ENST00000360876
ENST00000325888
ENST00000360876
ENST00000325888
ENST00000359863
```

Files containing phastCons and phyloP conservation scores can be downloaded from UCSC (for hg38 e.g. [phastCons100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw) and [phyloP100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw)). RNAProt accepts the files as inputs (modes `rnaprot gt` and `rnaprot gp`), with `--phastcons` and `--phylop`. Note that `rnaprot gp` reuses the set file paths used for training (`rnaprot train`), as long as the file paths are still valid. If not, it will complain, or you can just overwrite the paths by setting `--phastcons` and `--phylop`.


### Outputs

Depending on the executed program mode, various output files are generated:

- Reports on dataset statistics (.html) for `rnaprot gt`, `rnaprot eval`, and `rnaprot gp`.
- Sequence and additional feature profiles (png, pdf) for `rnaprot eval` and `rnaprot predict`
- Sequence and additional feature logos (.png, .pdf) for `rnaprot eval`
- Whole site predictions (.out) for `rnaprot predict` (`--mode 1`)
- Moving window peak region predictions (.bed, .tsv) for `rnaprot predict` (`--mode 2`)


#### HTML reports

For the dataset generation modes (`rnaprot gt`, `rnaprot gp`), HTML reports can be output which include detailed statistics and visualizations regarding the positive, negative, or test dataset (`--report` option). There are various comparative statistics available on: site lengths, sequence complexity, di-nucleotide distributions, k-mer statistics, target region biotype and overlap statistics, as well as additional statistics and visualizations for each chosen feature. The .html report file can be found in the mode output folder. For `rnaprot eval`, comparative statistics regarding positive and negative site scores are output, as well as a saliency peak distribution, or if `--add-train-in` is given a model comparison plot. 


#### Logos and profiles

In model evaluation mode (`rnaprot eval`), sequence and additional feature logos are output, as well as position-wise profiles (including saliency map and single mutation effects track) for a subset of training sites to illustrate binding preferences.


#### Prediction files

In model prediction mode (`rnaprot predict`), whole-site (`--mode 1`) or moving window peak region (`--mode 2`) predictions are output. Optionally, the top scoring windows can also be plotted as profiles just like for `rnaprot eval`.

In `--mode 2`, `peak_regions.bed` contains the predicted peak regions on the reference (depending on input sequence, transcript, or genomic coordinates), in 11-column BED format:


```
$ head -3 pum2_test_predict_out/peak_regions.bed
chr20	36050191	36050204	peak_1	3.9615161	-	36050145	36050224	0.13704819277108438	36050198	0.14824574
chr20	36049801	36049822	peak_2	4.028928	-	36049743	36049822	0.09839357429718887	36049817	0.32227173
chr20	36049711	36049724	peak_3	4.0312123	-	36049646	36049725	0.09738955823293183	36049718	0.0796903
```

Here the columns are: reference ID, reference peak region start position (0-based), reference peak region end position (1-based), peak ID, window score, strand (for transcript or sequence input always "+"), reference window start position (0-based), reference window end position (1-based), top saliency peak score (saliency), reference top saliency peak position (1-based), window score p-value (calculated from the positive training set scores distribution).

The same information can also be found in the `peak_regions.tsv` file (there all coordinates 1-based), which in addition contains the top saliency peak sequence and the window sequence:

```
$ head -3 pum2_test_predict_out/peak_regions.tsv
ref_id	peak_region_s	peak_region_e	strand	window_s	window_e	peak_id	window_score	win_sc_p_val	top_peak_score	top_peak_pos	top_peak_seq	window_seq
chr20	36050192	36050204	-	36050146	36050224	peak_1	3.9615161	0.13704819277108438	0.14824574	36050198	GUGUAUA	UACUGGCCGUUUAUGGAAGGCCUGUGUAUAUAAUAUGAAAAAGCUGCUCUCAACUCCACCCCAACCUUUUAAUAGAAAA
chr20	36049802	36049822	-	36049744	36049822	peak_2	4.028928	0.09839357429718887	0.32227173	36049817	GUAUAUA	GUGUAUAUAGUUGACAAUGCUAAGCUUUUUUGAAAUGUCUCUUCUUUUUAGAUGUUCUGAAGUGCCUGAUAUGUUAAAA
```



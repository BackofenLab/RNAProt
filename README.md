# RNAProt

RNAProt is a computational RBP binding site prediction framework based on recurrent neural networks (RNNs). Conceived as an end-to-end method, RNAProt includes all necessary functionalities, from dataset generation over model training to the evaluation of binding preferences and binding site prediction. Various input types and features are supported, accompanied by comprehensive statistics and visualizations to inform the user about datatset characteristics and learned model properties.


## Table of contents

- [RNAProt framework introduction](#the-rnaprot-framework)
- [RNAProt installation](#installation)
    - [Conda](#conda)
    - [Conda package installation](#conda-package-installation)
    - [Manual installation](#manual-installation)
- [Test runs](#test-runs)
    - [Test example with FASTA sequences as input](#test-example-with-fasta-sequences-as-input)
    - [Test example with genomic regions as input](#test-example-with-genomic-regions-as-input)
    - [Test example with additional features](#test-example-with-additional-features)
- [GraphProt2 documentation](#documentation)
    - [Program modes](#program-modes)
    - [Supported features](#supported-features)
    - [Inputs](#inputs)
    - [Outputs](#outputs)


## The RNAProt framework


RNAProt utilizes RBP binding sites identified by CLIP-seq and related protocols to train an RNN-based model. The model is then used to predict new binding sites on given input RNA sequences. The following figure illustrates the RNAProt framework and its general workflow:


<img src="docs/framework_overview.png" alt="RNAProt framework overview"
	title="RNAProt framework overview" width="550" />


Yellow boxes mark necessary framework inputs, blue boxes the five program modes of RNAProt, and green boxes the framework outputs. Arrows show the dependencies between inputs, modes, and outputs. RNAProt accepts RBP binding sites in FASTA or BED format. The latter one also requires a genomic sequence file (.2bit format) and a genomic annotations file (GTF format).


RNAProt requires at least three inputs: a set of RBP binding sites (either in BED or FASTA format), a genomic sequence file (.2bit format), and a genomic annotations file (GTF format). 
Binding sites can be supplied either as sequences, genomic regions, or as transcript regions (GTF file with corresponding transcript annotation required). 
Additional inputs are available, depending on the binding site input type as well as the selected features. For more details on inputs, modes, supported features, and outputs, see the documentation below.



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
conda create -n rnaprotenv python=3.8 -c conda-forge bioconda
conda activate rnaprotenv
conda install -c bioconda rnaprot
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

To manually install RNAProt, we first create a Conda environment (as described [above](#conda)). Once inside the environment, we need to install the following dependencies:

```
conda install -c conda-forge pytorch=1.7.1=cuda102py38h9f8c3ab_1 cudatoolkit=10.2
conda install -c conda-forge seaborn=0.11.1
conda install -c bioconda viennarna=2.4.17
conda install -c bioconda bedtools=2.29.2
conda install -c bioconda logomaker=0.8
conda install -c conda-forge hpbandster=0.7.4
conda install -c conda-forge markdown=3.2.2
conda install -c conda-forge plotly=4.14.3
conda install -c conda-forge scikit-learn=0.24.1
conda install -c bioconda ushuffle=1.2.2
conda install -c bioconda ucsc-twobitinfo
conda install -c bioconda ucsc-twobittofa
conda install -c bioconda ucsc-bigwigaverageoverbed
```

If you don't have a dedicated GPU (and you're not planning on getting one any time soon either), you don't need to install the additional GPU libraries. To install pyTorch without GPU support, simply exchange the above call with:

```
conda install -c conda-forge pytorch=1.7.1 cudatoolkit=10.2
```


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

Before training a model, we need to generate an RNAProt training dataset. For this we go to the cloned repository folder (clone + enter if not already there from first example), and use the FASTA sequences supplied in the `test` folder as training data. To get training set statistics, we also enable `--report`:


```
git clone https://github.com/BackofenLab/RNAProt.git
cd RNAProt
rnaprot gt --in test/PUM2_PARCLIP.positives.fa --neg-in test/PUM2_PARCLIP.negatives.fa --out PUM2_PARCLIP_gt_out --report
```

We can then take a look at the `report.rnaprot_gt.html` inside `test_gt_out`, informing us about similarities and differences between the positive and negative set. The content of the HTML report depends on selected features (e.g. structure, conservation scores, region annotations), and the input type given to `rnaprot gt` (FASTA sequences, genomic sites BED, or transcript sites BED). Here for example we can compare k-mer statistics of the positive and negative set, observing that the positives tend to contain more AA, UU, and AU repeat sites. This likely also contributes to the lower sequence complexity of the postive set.


Next we train a model on the created dataset, using default parameters. For this we simply run `rnaprot train` with the `rnaprot gt` output folder as input. We also enable `--verbose-train`, to see the learning progress over the number of epochs:

```
rnaprot train --in PUM2_PARCLIP_gt_out --out PUM2_PARCLIP_train_out --verbose-train
```

In the end we get a summary for the trained model, e.g. reporting the model validation AUC, the training runtime, and set hyperparameters. To visualize what our just-trained model has learned, we next run `rnaprot eval`, which requires both the `rnaprot gt` and `rnaprot train` output folders:

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

To create datasets based on genomic or transcript regions, we first need to download two additional files. Specifically, we need a GTF file (containing genomic annotations), as well as a .2bit file (containing the genomic sequences). Note that we used Ensembl GTF files to test RNAProt, and therefore recommend using these. You can find them [here](http://www.ensembl.org/info/data/ftp/index.html) for many major model organisms. The .2bit genome file we will download from [UCSC](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips). For this example, we choose the human genome + annotations (hg38 assembly):

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

If the output is cryptic instead, you need to try again. Next we download some genomic RBP binding regions identified by eCLIP from [ENCODE](https://www.encodeproject.org/). The ENCODE website contains a huge collection of eCLIP datasets for various RBPs. For this example, we again download PUM2 binding sites, choosing the IDR peaks identified by ENCODE's CLIPper pipeline (PUM2 K562 eCLIP dataset ID: ENCFF880MWQ, PUM2 K562 IDR peaks ID: ENCFF880MWQ). We unzip it and change the format to 6-column BED which RNAProt likes best:

```
wget https://www.encodeproject.org/files/ENCFF880MWQ/@@download/ENCFF880MWQ.bed.gz
gunzip -c ENCFF880MWQ.bed.gz | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$7"\t"$6}' > PUM2_K562_IDR_peaks.bed
```

Note that we move the log2 fold change value from column 7 (original file) to column 5, which is used by RNAProt to filter and select sites in case of overlaps. By default, `rnaprot gt` removes overlapping sites by selecting only the highest-scoring site from a set of overlapping sites. To disable this, set `--allow-overlaps`. If there are no column 5 scores given (or all the same), filtering of overlaps is disabled by default. Moreover, positive sites that do not overlap with gene regions by default are filtered. To disable this, set `--no-gene-filter`.

Next we create a training dataset, by supplying the downloaded GTF and .2bit file:

```
rnaprot gt --in PUM2_K562_IDR_peaks.bed --out PUM2_K562_IDR_gt_out --gtf Homo_sapiens.GRCh38.103.gtf.gz --gen hg38.2bit --report
```

Thanks to the given GTF file, the HTML report will now also include information on target gene regions and biotypes. Note that by default, `rnaprot gt` centers the input BED regions, and extends them on both sides by the set `--seq-ext` (by default 40). If you want to keep the original site lengths, set `--mode 2 --seq-ext 0`. Alternatively, you can set `--mode 3` to use the region upstream ends and extend by `--seq-ext`.

Now we can train a model and evaluate it just like in the example above:

```
rnaprot train --in PUM2_K562_IDR_gt_out --out PUM2_K562_IDR_train_out --verbose-train

rnaprot eval --gt-in PUM2_K562_IDR_gt_out --train-in PUM2_K562_IDR_train_out --out PUM2_K562_IDR_eval_out --report
```

For prediction, we could again use the folder generated by `rnaprot gp` from the above FASTA sequences + *NORAD* example. However, since we now have the GTF + genome .2bit file, we can also get its genomic or transcript region directly from these files (no need to download FASTA sequences or a gene BED file). As with `rnaprot gt`, `rnaprot gp` accepts sequences, genomic regions, or transcript regions as input. To get its genomic or transcript region, we just need its gene or transcript ID (as long as it is in the GTF file), and then can use one of the two helper scripts installed together with RNAProt:

```
gtf_extract_gene_regions.py --ids ENSG00000260032 --gtf Homo_sapiens.GRCh38.103.gtf.gz --out ENSG00000260032.bed

gtf_extract_transcript_regions.py --ids ENST00000565493 --gtf Homo_sapiens.GRCh38.103.gtf.gz --out ENST00000565493.bed
```

Of course the scripts also accept > 1 ID (either on the command line or more practically given as a file with one ID per row). In the case of *NORAD*, both transcript and gene region have the same length, since *NORAD* contains no introns and only one annotated isoform:

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

RNAProt supports various additional position(nucleotide)-wise features to learn from, such as secondary structure, region annotations (including user-defined ones), or conservation scores (see [Documentation](#documentation) for details). For this we have to specify what features to include in `rnaprot gt` and `rnaprot gp`, and depending on the feature also provide additional files. For model training (`rnaprot train`) we can then specify what features to use for training, from the features included in `rnaprot gt`. This has the advantage that features need to be extracted or computed only once, and that various feature combinations and parameter settings can be tested in training.

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

Here we increased the maximum number of epochs to 300, since for smaller datasets model performance can sometimes still improve beyond the default 200 epochs. This can be easily monitored with `--verbose-train` enabled). In addition, increasing `--patience` might sometimes be necessary, to prevent model training with the model stuck early on in the training process from stopping (although we experienced this only very rarely).
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

In our case the model successfully predicted the two verified binding sites (all together 4 sites predicted) on the transcript (using threshold level `--thr 1`). The first loop is at transcript position 1,371 to 1,373 (loop nucleotides), the second loop from 1,404 to 1,406, with the second hairpin having a higher folding probability and score. To check this, we take a look at the reported `peak_regions.bed` (or `peak_regions.tsv`), or have a look at the plotted profiles, which conveniently includes the transcript coordinates (or genomic coordinates in case of genomic sites as input) to quickly identify regions of interest. 
The `test/` folder also includes the model we used to predict, which you can easily apply yourself to compare:

```
unzip test/cde_sites_str_model_folder.zip

rnaprot predict --in CDE_sites_str_gp_out --train-in cde_sites_hg38_extlr40_w70l50_str_train_out --out CDE_sites_test_model_str_predict_out --mode 2 --plot-top-profiles --thr 1
```

This also shows how easy it is to share models. Once the model is trained, the `rnaprot train --out` folder can be copied and reused. Note however that predictions between two models can vary, since negative sites generation is random. Moreover, even models trained on the same positive and negative sites are slightly different from another and thus can lead to slightly different predictions. This is because model training incorporates stochastic processes, like the random initialization of network weights, or the application of dropout.





## Documentation


This documentation provides details on all the RNAProt (version 0.1) framework parts: program modes, supported features, inputs, and outputs.


### Program modes

GraphProt2 is divided into five different program modes: training set generation, prediction set generation, model training, model evaluation, and model prediction.


An overview of the modes can be obtained by:


```
$ rnaprot -h
usage: rnaprot [-h] [-v] {train,eval,predict,gt,gp} ...

Modelling RBP binding preferences to predict RPB binding sites.

positional arguments:
  {train,eval,predict,gt,gp}
                        Program modes
    train               Train a binding site prediction model
    eval                Evaluate properties learned from positive sites
    predict             Predict binding sites (whole sites or profiles)
    gt                  Generate training data set
    gp                  Generate prediction data set

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit

```

The following sections describe each mode in more detail.


#### Training set generation

The following command line arguments are available in `graphprot2 gt` mode:

```
graphprot2 gt -h
usage: graphprot2 gt [-h] --in str --out str [--gtf str] [--gen str]
                     [--mode {1,2,3}] [--mask-bed str] [--seq-ext int]
                     [--con-ext int] [--thr float] [--rev-filter]
                     [--max-len int] [--min-len int] [--keep-ids]
                     [--allow-overlaps] [--no-gene-filter] [--con-ext-pre]
                     [--neg-comp-thr float] [--neg-factor {2,3,4,5}]
                     [--keep-add-neg] [--neg-in str] [--shuffle-k {1,2,3}]
                     [--report] [--theme {1,2}] [--eia] [--eia-ib] [--eia-n]
                     [--tr-list str] [--phastcons str] [--phylop str] [--tra]
                     [--tra-codons] [--tra-borders] [--rra] [--str] [--bp-in]
                     [--plfold-u int] [--plfold-l int] [--plfold-w int]

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
  --seq-ext int         Up- and downstream sequence extension of sites (site
                        definition by --mode) with uppercase characters
                        (default: 30)
  --con-ext int         Up- and downstream context sequence extension of sites
                        (site definition by --mode) with lowercase characters.
                        Note that statistics (--report) are produced only for
                        uppercase sequence parts (defined by --seq-ext)
                        (default: False)
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
  --con-ext-pre         Add --con-ext extension before selecting for highest-
                        scoring sites in case of overlaps (not afterwards)
                        (default: False)
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
                        to generate negative sequences (if no --neg-fa
                        supplied) (default: 2)
  --report              Output an .html report providing various training set
                        statistics and plots (default: False)
  --theme {1,2}         Set theme for .html report (1: default, 2: midnight
                        blue) (default: 1)

required arguments:
  --in str              Genomic or transcript RBP binding sites file in BED
                        (6-column format) or FASTA format. If --in FASTA, only
                        --str is supported as additional feature. If --in BED,
                        --gtf and --gen become mandatory
  --out str             Output training data folder (== input folder to
                        graphprot2 train)

additional annotation arguments:
  --eia                 Add exon-intron annotations to genomic regions
                        (default: False)
  --eia-ib              Add intron border annotations to genomic regions (in
                        combination with --exon-intron) (default: False)
  --eia-n               Label regions not covered by intron or exon regions as
                        N instead of labelling them as introns (I) (in
                        combination with --exon-intron) (default: False)
  --tr-list str         Supply file with transcript IDs (one ID per row) for
                        exon intron labeling (using the corresponding exon
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
  --str                 Add base pairs and position-wise structural elements
                        probabilities features (calculate with RNAplfold)
                        (default: False)
  --bp-in               Supply a custom base pair annotation file for all --in
                        sites. This disables base pair calculation for the
                        positive set (default: False)
  --plfold-u int        RNAplfold -u parameter value (default: 3)
  --plfold-l int        RNAplfold -L parameter value (default: 100)
  --plfold-w int        RNAplfold -W parameter value (default: 150)

```

#### Prediction set generation

The following command line arguments are available in `graphprot2 gp` mode:

```
graphprot2 gp -h
usage: graphprot2 gp [-h] --in str --out str [--gtf str] [--gen str]
                     [--keep-ids] [--gene-filter] [--con-ext int] [--report]
                     [--theme {1,2}] [--eia] [--eia-ib] [--eia-n]
                     [--tr-list str] [--phastcons str] [--phylop str] [--tra]
                     [--tra-codons] [--tra-borders] [--rra] [--str]
                     [--bp-in str] [--plfold-u int] [--plfold-l int]
                     [--plfold-w int]

optional arguments:
  -h, --help       show this help message and exit
  --gtf str        Genomic annotations GTF file (.gtf or .gtf.gz)
  --gen str        Genomic sequences .2bit file
  --keep-ids       Keep --in BED column 4 site IDs. Note that site IDs have to
                   be unique (default: False)
  --gene-filter    Filter --in sites based on gene coverage (gene annotations
                   from --gtf) (default: False)
  --con-ext int    Up- and downstream context sequence extension of --in sites
                   with lowercase characters for whole site prediction
                   (graphprot predict --mode 1). Best use same --con-ext
                   values in gp+gt+train modes. Note that statistics
                   (--report) are produced only for uppercase sequence parts
                   (default: False)
  --report         Output an .html report providing various training set
                   statistics and plots (default: False)
  --theme {1,2}    Set theme for .html report (1: default, 2: midnight blue)
                   (default: 1)

required arguments:
  --in str         Genomic or transcript RBP binding sites file in BED
                   (6-column format) or FASTA format. If --in FASTA, only
                   --str is supported as additional feature. If --in BED,
                   --gtf and --gen become mandatory
  --out str        Output prediction dataset folder (== input folder to
                   graphprot2 predict)

additional annotation arguments:
  --eia            Add exon-intron annotations to genomic regions (default:
                   False)
  --eia-ib         Add intron border annotations to genomic regions (in
                   combination with --exon-intron) (default: False)
  --eia-n          Label regions not covered by intron or exon regions as N
                   instead of labelling them as introns (I) (in combination
                   with --exon-intron) (default: False)
  --tr-list str    Supply file with transcript IDs (one ID per row) for exon
                   intron labeling (using the corresponding exon regions from
                   --gtf). By default, exon regions of the most prominent
                   transcripts (automatically selected from --gtf) are used
                   (default: False)
  --phastcons str  Genomic .bigWig file with phastCons conservation scores to
                   add as annotations
  --phylop str     Genomic .bigWig file with phyloP conservation scores to add
                   as annotations
  --tra            Add transcript region annotations (5'UTR, CDS, 3'UTR, None)
                   to genomic and transcript regions (default: False)
  --tra-codons     Add start and stop codon annotations to genomic or
                   transcript regions (in combination with --tra) (default:
                   False)
  --tra-borders    Add transcript and exon border annotations to transcript
                   regions (in combination with --tra) (default: False)
  --rra            Add repeat region annotations for genomic or transcript
                   regions retrieved from --gen .2bit (default: False)
  --str            Add base pairs and position-wise structural elements
                   probabilities features (calculate with RNAplfold) (default:
                   False)
  --bp-in str      Supply a custom base pair annotation file for all --in
                   sites, disabling base pair calculation with RNAplfold
                   (default: False)
  --plfold-u int   RNAplfold -u parameter value (default: 3)
  --plfold-l int   RNAplfold -L parameter value (default: 100)
  --plfold-w int   RNAplfold -W parameter value (default: 150)

```

#### Model training

The following command line arguments are available in `graphprot2 train` mode:

```
graphprot2 train -h
usage: graphprot2 train [-h] --in IN_FOLDER --out OUT_FOLDER [--only-seq]
                        [--use-phastcons] [--use-phylop] [--use-eia]
                        [--use-tra] [--use-rra] [--use-str-elem-p] [--use-bps]
                        [--bps-mode {1,2}] [--bps-prob-cutoff float]
                        [--uc-context] [--gen-cv] [--gen-cv-k {5,10}]
                        [--gm-cv] [--train-cv] [--train-cv-k {5,10}]
                        [--train-vs float] [--batch-size int [int ...]]
                        [--epochs int] [--patience int] [--fc-hidden-dim int]
                        [--list-lr float [float ...]]
                        [--list-hidden-dim int [int ...]]
                        [--list-weight-decay float [float ...]]

optional arguments:
  -h, --help            show this help message and exit

required arguments:
  --in IN_FOLDER        Input training data folder (output of graphprot2 gt)
  --out OUT_FOLDER      Model training results output folder

feature definition arguments:
  --only-seq            Use only sequence feature. By default all features
                        present in --in are used as node attributes (default:
                        False)
  --use-phastcons       Add phastCons conservation scores. Set --use-x to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)
  --use-phylop          Add phyloP conservation scores. Set --use-x to define
                        which features to add on top of sequence feature (by
                        default all --in features are used)
  --use-eia             Add exon-intron annotations. Set --use-x to define
                        which features to add on top of sequence feature (by
                        default all --in features are used)
  --use-tra             Add transcript region annotations. Set --use-x to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)
  --use-rra             Add repeat region annotations. Set --use-x to define
                        which features to add on top of sequence feature (by
                        default all --in features are used)
  --use-str-elem-p      Add structural elements probabilities. Set --use-x to
                        define which features to add on top of sequence
                        feature (by default all --in features are used)
  --use-bps             Add base pairs to graph. Set --use-x to define which
                        features to add on top of sequence feature (by default
                        all --in features are used)
  --bps-mode {1,2}      Defines which base pairs are added to the graphs.
                        --bpp-mode 1 : base pairs with start or end in
                        viewpoint region. --bpp-mode 2 : only base pairs with
                        start+end in viewpoint (default: 1)
  --bps-prob-cutoff float
                        Base pair probability cutoff for filtering base pairs
                        added to the graph (default: 0.5)
  --uc-context          Convert lowercase context (if present, added by
                        graphprot2 gt --con-ext) to uppercase (default: False)

model definition arguments:
  --gen-cv              Run cross validation in combination with
                        hyperparameter optimization to evaluate generalization
                        performance (default: False)
  --gen-cv-k {5,10}     Cross validation k for evaluating generalization
                        performance (default: 10)
  --gm-cv               Treat data as generic model data (positive IDs with
                        specific format required). This turns on generic model
                        data cross validation, with every fold leaving one RBP
                        set out for testing (ignoring --gen-cv and --gen-cv-k)
                        (default: False)
  --train-cv            Run cross validation to train final model, with
                        hyperparameter optimization in each split and
                        selection of best parameters based their on average
                        performance on validation sets. By default final model
                        training is done for one split only (validation set
                        size controlled by --train-vs). Note that --train-cv
                        with many hyperparameter combinations considerably
                        increases run time (default: False)
  --train-cv-k {5,10}   Final model cross validation k. Use in combination
                        with --train-cv (default: 5)
  --train-vs float      Validation set size for training final model as
                        percentage of all training sites. Only effective if
                        --train-cv not set (with --train-cv validation set
                        size controlled by --train-cv-k) (default: 0.2)
  --batch-size int [int ...]
                        List of gradient descent batch sizes (default: 50)
  --epochs int          Number of training epochs (default: 200)
  --patience int        Number of epochs to wait for further improvement on
                        validation set before stopping (default: 30)
  --fc-hidden-dim int   Number of dimensions for fully connected layers
                        (default: 128)
  --list-lr float [float ...]
                        List of learning rates for hyperparameter optimization
                        (default: 0.0001)
  --list-hidden-dim int [int ...]
                        List of node feature dimensions in hidden layers for
                        hyperparameter optimization (default: 128)
  --list-weight-decay float [float ...]
                        List of weight decays for hyperparameter optimization
                        (default: 0.0001)

```

#### Model evaluation

The following command line arguments are available in `graphprot2 eval` mode:

```
graphprot2 eval -h
usage: graphprot2 eval [-h] --in IN_FOLDER --out OUT_FOLDER
                       [--nr-top-sites LIST_NR_TOP_SITES [LIST_NR_TOP_SITES ...]]
                       [--nr-top-profiles int]
                       [--motif-size LIST_MOTIF_SIZES [LIST_MOTIF_SIZES ...]]
                       [--motif-sc-thr float]
                       [--win-size LIST_WIN_SIZES [LIST_WIN_SIZES ...]]
                       [--plot-format {1,2}]

optional arguments:
  -h, --help            show this help message and exit
  --nr-top-sites LIST_NR_TOP_SITES [LIST_NR_TOP_SITES ...]
                        Specify number(s) of top predicted sites used for
                        motif extraction. Provide multiple numbers (e.g. --nr-
                        top-sites 100 500 1000) to extract one motif plot from
                        each site set (default: 500)
  --nr-top-profiles int
                        Specify number of top predicted sites to plot profiles
                        for (default: 25)
  --motif-size LIST_MOTIF_SIZES [LIST_MOTIF_SIZES ...]
                        Motif size(s) (widths) for extracting and plotting
                        motifs. Provide multiple sizes (e.g. --motif-size 5 7
                        9) to extract a motif for each size (default: 7)
  --motif-sc-thr float  Minimum profile score of position to be included in
                        motif (default: 0.3)
  --win-size LIST_WIN_SIZES [LIST_WIN_SIZES ...]
                        Windows size(s) for calculating position-wise scoring
                        profiles. Provide multiple sizes (e.g. --win-size 5 7
                        9) to compute average profiles (default: 7)
  --plot-format {1,2}   Plotting format. 1: png, 2: pdf (default: 1)

required arguments:
  --in IN_FOLDER        Input model training folder (output of graphprot2
                        train)
  --out OUT_FOLDER      Evaluation results output folder

```

#### Model prediction

The following command line arguments are available in `graphprot2 predict` mode:

```
graphprot2 predict -h
usage: graphprot2 predict [-h] --in IN_FOLDER --model-in MODEL_IN_FOLDER --out
                          str [--mode {1,2}]
                          [--win-size LIST_WIN_SIZES [LIST_WIN_SIZES ...]]
                          [--peak-ext int] [--con-ext int] [--thr float]
                          [--max-merge-dist int]

optional arguments:
  -h, --help            show this help message and exit
  --mode {1,2}          Define prediction mode. (1) predict whole sites, (2)
                        predict position-wise scoring profiles and extract
                        top-scoring sites from profiles (default: 1)
  --win-size LIST_WIN_SIZES [LIST_WIN_SIZES ...]
                        Windows size(s) for calculating position-wise scoring
                        profiles. Provide multiple sizes (e.g. --win-size 5 7
                        9) to compute average profiles (default: 11)
  --peak-ext int        Up- and downstream peak position extension for
                        extracting top-scoring sites from fixed-window
                        profiles (default: 30)
  --con-ext int         Up- and downstream context extension for extracting
                        top-scoring sites from fixed-window profiles. By
                        default uses --con-ext info from --model-in (if set in
                        graphprot2 train), but restricts it to a maximum of 50
                        (default: False)
  --thr float           Minimum profile position score for extracting peak
                        regions and top-scoring sites. Further increase e.g.
                        in case of too many or too broad peaks (default: 0.5)
  --max-merge-dist int  Maximum distance between two peaks for merging. Two
                        peakse get merged to one if they are <= --max-merge-
                        dist away from each other (default: 0)

required arguments:
  --in IN_FOLDER        Input prediction data folder (output of graphprot2 gp)
  --model-in MODEL_IN_FOLDER
                        Input model training folder containing model file and
                        parameters (output of graphprot2 train)
  --out str             Prediction results output folder

```

### Supported features

GraphProt2 currently supports the following position-wise features which can be utilized for training and prediction in addition to the sequence feature: secondary structure information (base pairs and structural element probabilities), conservation scores (phastCons and phyloP), exon-intron annotation, transcript region annotation, and repeat region annotation. The following table lists the features available for each of the three input types (sequences, genomic regions, transcript regions):


|   Feature       | Sequences | Genomic regions | Transcript regions |
| :--------------: | :--------------: | :--------------: | :--------------: |
| **structure**    | YES | YES | YES |
| **conservation scores**    | NO | YES | YES |
| **exon-intron annotation**    | NO | YES | NO |
| **transcript region annotation**    | NO | YES | YES |
| **repeat region annotation**    | NO | YES | YES |


#### Secondary structure information

GraphProt2 can include two kinds of structure information for a given RNA sequence: 1) base pairs and 2) unpaired probabilities for different loop contexts (probabilities for the nucleotide being paired or inside external, hairpin, internal or multi loops). Both are calculated using the ViennaRNA Python 3 API (ViennaRNA 2.4.14) and RNAplfold with its sliding window approach, with user-definable parameters (by default these are window size = 150, maximum base pair span length = 100, and probabilities for regions of length u = 3). The base pairs with a probability \>= a set threshold (default = 0.5) are then added to the sequence graph as edges between the nodes that represent the end points of the base pair, and the unpaired probability values are added to the node feature vectors. Alternatively, the user can also provide custom base pair information (`--bp-in` option, for `graphprot2 gt` and `graphprot2 gp`), e.g. derived from experimental data. For more details see mode options `--str`, `--use-bps`, `--use-str-elem-p`, `--bps-mode`, and `--bps-prob-cutoff`.


#### Conservation scores

GraphProt2 supports two scores measuring evolutionary conservation (phastCons and phyloP). For the paper, conservation scores were downloaded from the UCSC website, using the phastCons and phyloP scores generated from multiple sequence alignments of 99 vertebrate genomes to the human genome (hg38, [phastCons100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw) and [phyloP100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw) datasets). GraphProt2 accepts scores in bigWig (.bw) format (modes `graphprot2 gt` and `graphprot2 gp`, options `--phastcons` and `--phylop`). To assign conservation scores to transcript regions, transcript regions are first mapped to the genome, using the provided GTF file.


#### Exon-intron annotation

Exon-intron annotation in the form of one-hot encoded exon and intron labels can also be added to the node feature vectors.
Labels are assigned to each binding site position by taking a set of genomic exon regions and overlapping it with the genomic binding sites using bedtools (v. 2.29.0). To unambiguously assign labels, GraphProt2 by default uses the most prominent isoform for each gene. The most prominent isoform for each gene gets selected through hierarchical filtering of the transcript information present in the input GTF file (tested with GTF files from [Ensembl](http://www.ensembl.org/info/data/ftp/index.html)): given that the transcript is part of the GENCODE basic gene set, GraphProt2 selects transcripts based on their transcript support level (highest priority), and by transcript length (longer isoform preferred). The extracted isoform exons are then used for region type assignment.
Note that this feature is only available for genomic regions, as it is not informative for transcript regions, which would contain only exon labels. Optionally, a user-defined isoform list can be supplied (`--tr-list`), substituting the list of most prominent isoforms for annotation. Regions not overlapping with introns or exons can also be labelled separately (instead of labelled as intron). For more details see mode options `--eia`, `--eia-ib`, and `--eia-n`.


#### Transcript region annotation

Similarly to the exon-intron annotation, binding regions can be labelled based on their overlap with transcript regions. Labels are assigned based on UTR or CDS region overlap (5'UTR, CDS, 3'UTR, None), by taking the isoform information in the input GTF file. Again the list of most prominent isoforms is used for annotation, or alternatively a list of user-defined isoforms (`--tr-list`). Additional annotation options include start and stop codon or transcript and exon border labelling. For more details see mode options `--tra`, `--tra-codons`, and `--tra-borders`.


#### Repeat region annotation

Repeat region annotation (`--rra` option) can also be added to the binding regions, analogously to other region type annotations. This information is derived directly from the 2bit formatted genomic sequence file, where repeat regions identified by RepeatMasker and Tandem Repeats Finder are stored in lowercase letters. 


### Inputs

GraphProt2 requires at least three inputs: a set of RBP binding sites (BED or FASTA format), a genomic sequence file (.2bit format), and a genomic annotations file (GTF format). Additional input files include BED files (negative sites or regions for masking), conservation scores in bigWig format, transcript lists, or custom base pair information.


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

Note that GraphProt2 by default creates unique site IDs. Optionally, the original sequence or site IDs (BED column 4) can be kept (`--keep-ids` in `graphprot2 gt` and `graphprot2 gp`). Also note that BED column 5 contains the site score, which can be used for filtering (`--thr`). In case of p-values, reverse filtering can be enabled with `--rev-filter` (smaller value == better). Filtering by site length is also possible (`--max-len`, `--min-len`), as well as various region filters (`--no-gene-filter`, `--allow-overlaps`, see modes section for more details).

BED files with genomic RBP binding regions can for example be downloaded from [ENCODE](https://www.encodeproject.org/) (eCLIP CLIPper peak regions).


#### Genome sequence

The human genome .2bit formatedd genomic sequence file can be downloaded [here](https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.2bit). This file is used to extract genomic and transcript region sequences, as well as repeat region annotations.


#### Genome annotations

GraphProt2 was tested with GTF files obtained from Enseml. See Ensembl's [download page](http://www.ensembl.org/info/data/ftp/index.html) to download the latest GTF file with genomic annotations. Note that GraphProt2 can directly read from gzipped GTF. The GTF file is used to extract exon-intron and transcript region annotations.


#### Additional inputs

Additional input files currently are (depending on set mode):

- BED files (negative sites or regions for masking, 6-column BED format as described)
- A transcript ID list file for exon-intron or transcript region annotation
- Conservation scores in bigWig format
- Custom base pair information file

The transcript IDs list file (`--tr-list` option, for `graphprot2 gt` and `graphprot2 gp`) has the following format (one ID per row):

```
$ head -5 tr_list.in
ENST00000360876
ENST00000325888
ENST00000360876
ENST00000325888
ENST00000359863
```

Files containing phastCons and phyloP conservation scores can be downloaded from UCSC (for hg38 e.g. [phastCons100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw) and [phyloP100way](http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw)). GraphProt2 accepts the files as inputs (modes `graphprot2 gt` and `graphprot2 gp`), with `--phastcons` and `--phylop`.


The custom base pair information file (`--bp-in` option, for `graphprot2 gt` and `graphprot2 gp`) can be provided with or without base pair probabilities:

```
$ head -5 bp_info1.in
>CLIP_1
1	9	0.133747
1	10	0.187047
1	20	0.038428
1	81	0.012488

$ head -5 bp_info2.in
>CLIP_1
1	9
1	10
1	20
1	81
```

So the format is sequence ID header (`>sequence_id`, same as in FASTA file), followed by the sequence base pairs, followed by the next ID header, followed by its base pairs, and so on. Each base pair has one row, with two or three columns. Column 1 and 2 (mandatory) are the base pair start and end coordinates (1-based index, where 1 is the first sequence position). Optionally, column 3 stores the base pair probability.


### Outputs

Depending on the executed program mode, various output files are generated:

- Reports on dataset statistics (.html) for `graphprot2 gt` and `graphprot2 gp`
- Sequence and additional feature profiles (png, pdf) for `graphprot2 eval`
- Sequence and additional feature logos (.png, .pdf) for `graphprot2 eval`
- Whole site predictions (.out) for `graphprot2 predict`
- Position-wise scoring profiles (.out), peak regions and top-scoring sites (.bed) for `graphprot2 predict`


#### HTML reports

For the dataset generation modes (`graphprot2 gt`, `graphprot2 gp`), HTML reports can be output which include detailed statistics and visualizations regarding the positive, negative, or test dataset (`--report` option). Currently there are comparative statistics available on: site lengths, sequence complexity, di-nucleotide distributions, k-mer statistics, target region biotype and overlap statistics, as well as additional statistics and visualizations for each chosen feature. The .html report file can be found in the mode output folder.


#### Logos and profiles

In model evaluation mode (`graphprot2 eval`), sequence and additional feature logos are output, as well as position-wise scoring profiles for a subset of training sites to illustrate binding preferences.


#### Prediction files

In model prediction mode (`graphprot2 predict`), position-wise scoring profiles or whole site predictions are output, and top-scoring sites are extracted from the generated profiles.






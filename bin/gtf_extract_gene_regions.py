#!/usr/bin/env python3

from rnaprot import rplib
import subprocess
import argparse
import os
import re


def setup_argument_parser():
    """Setup argparse parser."""
    help_description = """
    Extract gene regions in BED format (6-col) from (Ensembl) GTF.
    
    """
    # Define argument parser.
    p = argparse.ArgumentParser(add_help=False,
                                prog="gtf_extract_gene_regions.py",
                                description=help_description,
                                formatter_class=argparse.MetavarTypeHelpFormatter)

    # Required arguments.
    p.add_argument("-h", "--help",
                   action="help",
                   help="Print help message")
    p.add_argument("--ids",
                   dest="in_gene_ids",
                   type=str,
                   nargs='+',
                   required = True,
                   help = "Provide gene ID(s) to extract BED regions for (e.g. --ids ENSG00000223972 ENSG00000227232) from --gtf. NOTE1 that IDs must be provided without version number. NOTE2 --ids also accepts a file with gene IDs (one ID per row)")
    p.add_argument("--gtf",
                   dest="in_gtf",
                   type=str,
                   metavar='str',
                   required = True,
                   help = "Genomic annotations GTF file (.gtf or .gtf.gz)")
    p.add_argument("--out",
                   dest="out_bed",
                   type=str,
                   required = True,
                   help = "Output BED file")
    return p



if __name__ == '__main__':

    parser = setup_argument_parser()
    args = parser.parse_args()
    
    assert os.path.exists(args.in_gtf), "input .gtf file \"%s\" not found" %(args.in_gtf)
    
    gene_ids_dic = {}
    if len(args.in_gene_ids) == 1 and os.path.exists(args.in_gene_ids[0]):
        gene_ids_dic = rplib.read_ids_into_dic(args.in_gene_ids[0],
                                               check_dic=False)
        assert gene_ids_dic, "no gene IDs read in from %s" %(args.in_gene_ids[0])
    else:
        for gene_id in args.in_gene_ids:
            gene_ids_dic[gene_id] = 1
    assert gene_ids_dic, "no gene IDs read into gene_ids_dic"

    print("Extracting gene regions from GTF ... ")
    rplib.gtf_extract_gene_bed(args.in_gtf, args.out_bed,
                               gene_ids_dic=gene_ids_dic)

    bed_ids_dic = rplib.bed_get_region_ids(args.out_bed,
                                           check_dic=False)

    assert bed_ids_dic, "no gene regions extracted from --gtf. Gene IDs provided via --ids must be present in --gtf"

    c_extracted = 0
    for gene_id in gene_ids_dic:
        if gene_id not in bed_ids_dic:
            print("WARNING: no gene region extracted for --in gene ID %s" %(gene_id))
        else:
            c_extracted += 1
    if c_extracted:
        print("# of extracted gene regions:  %i" %(c_extracted))
        print("Gene regions written to:\n%s" %(args.out_bed))

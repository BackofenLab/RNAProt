#!/usr/bin/env python3

from rnaprot import rplib
import subprocess
import argparse
import os
import re


def setup_argument_parser():
    """Setup argparse parser."""
    help_description = """
    Extract transcript regions in BED format (6-col) from (Ensembl) GTF.
    
    """
    # Define argument parser.
    p = argparse.ArgumentParser(add_help=False,
                                prog="gtf_extract_transcript_regions.py",
                                description=help_description,
                                formatter_class=argparse.MetavarTypeHelpFormatter)

    # Required arguments.
    p.add_argument("-h", "--help",
                   action="help",
                   help="Print help message")
    p.add_argument("--ids",
                   dest="in_tr_ids",
                   type=str,
                   nargs='+',
                   required = True,
                   help = "Provide transcript ID(s) to extract BED regions for (e.g. --ids ENST00000456328 ENST00000488147) from --gtf. NOTE1 that IDs must be provided without version number. NOTE2 --ids also accepts a file with transcript IDs (one ID per row)")
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
    
    tr_ids_dic = {}
    if len(args.in_tr_ids) == 1 and os.path.exists(args.in_tr_ids[0]):
        tr_ids_dic = rplib.read_ids_into_dic(args.in_tr_ids[0],
                                               check_dic=False)
        assert tr_ids_dic, "no transcript IDs read in from %s" %(args.in_tr_ids[0])
    else:
        for tr_id in args.in_tr_ids:
            tr_ids_dic[tr_id] = 1
    assert tr_ids_dic, "no transcript IDs read into tr_ids_dic"

    print("Extracting transcript regions from GTF ... ")
    tr_len_dic = rplib.gtf_get_transcript_lengths(args.in_gtf)
    assert tr_len_dic, "no transcript lengths extracted from --gtf %s" %(args.in_gtf)

    c_miss = 0
    c_found = 0
    for tr_id in tr_ids_dic:
        if tr_id not in tr_len_dic:
            print("WARNING: --in transcript ID %s has no entry in --gtf" %(tr_id))
        else:
            c_found += 1
    assert c_found, "no --gtf entries for any --in transcript IDs"
    rplib.bed_sequence_lengths_to_bed(tr_len_dic, args.out_bed,
                                      ids_dic=tr_ids_dic)

    print("# of extracted transcript regions:  %i" %(c_found))
    print("Transcript regions written to:\n%s" %(args.out_bed))


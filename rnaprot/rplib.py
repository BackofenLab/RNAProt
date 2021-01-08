#!/usr/bin/env python3

from distutils.spawn import find_executable
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from math import log
import pandas as pd
import numpy as np
import statistics
import subprocess
import logomaker
import random
import torch
import gzip
import uuid
import sys
import re
import os


"""

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~ OPEN FOR BUSINESS ~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


AuthoR: uhlm [at] informatik [dot] uni-freiburg [dot] de


~~~~~~~~~~~~~
Run doctests
~~~~~~~~~~~~~

python3 -m doctest gp2lib.py
python3 -m doctest -v gp2lib.py


"""

################################################################################

def extract_vp_seq(seqs_dic, seq_id,
                   use_con_ext=False,
                   con_ext=100):
    """
    Extract viewpoint part (uppercase chars) from sequence with
    given sequence ID seq_id.
    If use_con_ext is set, viewpoint region will be extended by
    con_ext. Thus total length of returned sequence will be
    len(vp_region)+2*len(con_ext).
    Return sequence, start position + end position (both one-based)
    of extracted sequence.

    >>> seqs_dic = {"CLIP_01" : "acguACGUacgu", "CLIP_02" : "CCCCgggg"}
    >>> seq, s, e = extract_vp_seq(seqs_dic, "CLIP_01")
    >>> print(seq, s, e)
    ACGU 5 8
    >>> seq, s, e = extract_vp_seq(seqs_dic, "CLIP_01", use_con_ext=True, con_ext=2)
    >>> print(seq, s, e)
    guACGUac 3 10
    >>> seq, s, e = extract_vp_seq(seqs_dic, "CLIP_02", use_con_ext=True, con_ext=2)
    >>> print(seq, s, e)
    CCCCgg 1 6

    """
    # Check.
    if not seq_id in seqs_dic:
        print ("ERROR: seq_id \"%s\" not found in seqs_dic" % (seq_id))
        sys.exit()
    seq = seqs_dic[seq_id]
    m = re.search("([acgun]*)([ACGUN]+)([acgun]*)", seq)
    if m:
        us_seq = m.group(1)
        vp_seq = m.group(2)
        ds_seq = m.group(3)
        l_us = len(us_seq)
        l_vp = len(vp_seq)
        l_ds = len(ds_seq)
        # Viewpoint start + end.
        new_s = l_us+1
        new_e = l_us+l_vp
        new_seq = vp_seq
        if use_con_ext:
            new_us_seq = us_seq[-con_ext:]
            new_ds_seq = ds_seq[:con_ext]
            l_new_us = len(new_us_seq)
            l_new_ds = len(new_ds_seq)
            new_s = l_us-l_new_us+1
            new_e = l_us+l_vp+l_new_ds
            new_seq = new_us_seq+vp_seq+new_ds_seq
        return new_seq, new_s, new_e
    else:
        print ("ERROR: extract_vp_seq() viewpoint extraction failed for \"%s\"" % (seq_id))
        sys.exit()


################################################################################

def read_fasta_into_dic(fasta_file,
                        seqs_dic=False,
                        ids_dic=False,
                        dna=False,
                        report=1,
                        all_uc=False,
                        skip_data_id="set",
                        skip_n_seqs=True):
    """
    Read in FASTA sequences, store in dictionary and return dictionary.

    >>> test_fasta = "test_data/test.fa"
    >>> d = read_fasta_into_dic(test_fasta)
    >>> print(d)
    {'seq1': 'acguACGUacgu', 'seq2': 'ugcaUGCAugcaACGUacgu'}

    """
    if not seqs_dic:
        seqs_dic = {}
    seq_id = ""
    # Go through FASTA file, extract sequences.
    with open(fasta_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                assert seq_id not in seqs_dic, "non-unique FASTA header \"%s\" in \"%s\"" % (seq_id, fasta_file)
                if ids_dic:
                    if seq_id in ids_dic:
                        seqs_dic[seq_id] = ""
                else:
                    seqs_dic[seq_id] = ""
            elif re.search("[ACGTUN]+", line, re.I):
                m = re.search("([ACGTUN]+)", line, re.I)
                seq = m.group(1)
                if seq_id in seqs_dic:
                    if dna:
                        # Convert to DNA, concatenate sequence.
                        seq = seq.replace("U","T").replace("u","t")
                    else:
                        # Convert to RNA, concatenate sequence.
                        seq = seq.replace("T","U").replace("t","u")
                    if all_uc:
                        seq = seq.upper()
                    seqs_dic[seq_id] += seq
    f.closed
    # Check if sequences read in.
    assert seqs_dic, "no sequences read in (input FASTA file \"%s\" empty or mal-formatted?)" %(fasta_file)
    # If sequences with N nucleotides should be skipped.
    c_skipped_n_ids = 0
    if skip_n_seqs:
        del_ids = []
        for seq_id in seqs_dic:
            seq = seqs_dic[seq_id]
            if re.search("N", seq, re.I):
                if report == 1:
                    print ("WARNING: sequence with seq_id \"%s\" in file \"%s\" contains N nucleotides. Discarding sequence ... " % (seq_id, fasta_file))
                c_skipped_n_ids += 1
                del_ids.append(seq_id)
        for seq_id in del_ids:
            del seqs_dic[seq_id]
        assert seqs_dic, "no sequences remaining after deleting N containing sequences (input FASTA file \"%s\")" %(fasta_file)
        if c_skipped_n_ids:
            if report == 2:
                print("# of N-containing %s regions discarded:  %i" %(skip_data_id, c_skipped_n_ids))
    return seqs_dic


################################################################################

def string_vectorizer(seq,
                      s=False,
                      e=False,
                      convert_to_uc=False,
                      empty_vectors=False,
                      custom_alphabet=False):
    """
    Take string sequence, look at each letter and convert to one-hot-encoded
    vector. Optionally define start and end index (1-based) for extracting
    sub-sequences.
    Return array of one-hot encoded vectors.
    If empty_vectors=True, return list of empty vectors.
    Enable convert_to_uc to convert characters to uppercase before string
    vectorization. Good if context regions are part of given subsequence.

    >>> string_vectorizer("ACGU")
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    >>> string_vectorizer("")
    []
    >>> string_vectorizer("XX")
    [[0, 0, 0, 0], [0, 0, 0, 0]]
    >>> string_vectorizer("ABC", empty_vectors=True)
    [[], [], []]
    >>> string_vectorizer("aCGu", convert_to_uc=False)
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    >>> string_vectorizer("aCGu", convert_to_uc=True)
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    """
    alphabet=['A','C','G','U']
    if custom_alphabet:
        alphabet = custom_alphabet
    seq_l = len(seq)
    if convert_to_uc:
        seq = seq.upper()
    if empty_vectors:
        vector = []
        for letter in seq:
            vector.append([])
    else:
        vector = [[1 if char == letter else 0 for char in alphabet] for letter in seq]
    if s and e:
        if len(seq) < e or s < 1:
            print ("ERROR: invalid indices passed to string_vectorizer (s:\"%s\", e:\"%s\")" % (s, e))
            sys.exit()
        vector = vector[s-1:e]
    return vector


################################################################################

def char_vectorizer(char,
                    custom_alphabet=False):
    """
    Vectorize given nucleotide character. Convert to uppercase before
    vectorizing.

    >>> char_vectorizer("C")
    [0, 1, 0, 0]
    >>> char_vectorizer("g")
    [0, 0, 1, 0]
    >>> char_vectorizer("M", ['E', 'H', 'I', 'M', 'S'])
    [0, 0, 0, 1, 0]

    """
    alphabet = ['A','C','G','U']
    if custom_alphabet:
        alphabet = custom_alphabet
    char = char.upper()
    l = len(char)
    vector = []
    assert l == 1, "given char length != 1 (given char: \"%s\")" % (l)
    for c in alphabet:
        if c == char:
            vector.append(1)
        else:
            vector.append(0)
    return vector


################################################################################

def update_sequence_viewpoint_regions(seqs_dic, vp_s_dic, vp_e_dic):
    """
    Update sequence viewpoint regions, i.e. regions marked by vp_s_dic and
    vp_e_dic, converting nucleotides to uppercase.

    >>> seqs_dic = {"S1" : "acgtACGTacgt"}
    >>> vp_s_dic = {"S1" : 4}
    >>> vp_e_dic = {"S1" : 9}
    >>> update_sequence_viewpoint_regions(seqs_dic, vp_s_dic, vp_e_dic)
    >>> seqs_dic
    {'S1': 'acgTACGTAcgt'}
    >>> seqs_dic = {"S2" : "acgtacgtACGTac"}
    >>> vp_s_dic = {"S2" : 5}
    >>> vp_e_dic = {"S2" : 16}
    >>> update_sequence_viewpoint_regions(seqs_dic, vp_s_dic, vp_e_dic)
    >>> seqs_dic
    {'S2': 'acgtACGTACGTAC'}

    """
    for seq_id in seqs_dic:
        vp_s = vp_s_dic[seq_id]
        vp_e = vp_e_dic[seq_id]
        seq = seqs_dic[seq_id]
        # Extract and update case for sequence parts.
        us_seq = seq[:vp_s-1].lower()
        vp_seq = seq[vp_s-1:vp_e].upper()
        ds_seq = seq[vp_e:].lower()
        # Store new sequence in given dictionary.
        seqs_dic[seq_id] = us_seq + vp_seq + ds_seq


################################################################################

def update_sequence_viewpoint(seq, vp_s, vp_e):
    """
    Update sequence viewpoint, i.e. region marked by vp_s (start) and vp_e
    (end), converting viewpoint to uppercase, and rest to lowercase.
    NOTE that vp_s and vp_e are expected to be 1-based index.

    >>> seq = "acgtACGTacgt"
    >>> update_sequence_viewpoint(seq, 4, 9)
    'acgTACGTAcgt'
    >>> seq = "acgtacgtACGTac"
    >>> update_sequence_viewpoint(seq, 5, 16)
    'acgtACGTACGTAC'

    """
    assert seq, "seq empty"
    assert vp_s <= vp_e, "vp_s > vp_e"
    us_seq = seq[:vp_s-1].lower()
    vp_seq = seq[vp_s-1:vp_e].upper()
    ds_seq = seq[vp_e:].lower()
    new_seq = us_seq + vp_seq + ds_seq
    return new_seq


################################################################################

def extract_viewpoint_regions_from_fasta(seqs_dic,
                                         vp_s_dic=False,
                                         vp_e_dic=False,
                                         center_vp=False,
                                         id2se_dic=None,
                                         get_se_dic=False,
                                         vp_ext=False):
    """
    Extract viewpoint start and end positions from FASTA dictionary.
    Return dictionaries for start+end (1-based indices, key:fasta_id).
    Set center_vp to center the extracted viewpoints, and vp_ext to
    bring all viewpoints to same length 1+2*vp_ext

    get_se_dic:
        Store viewpoint starts and ends for each ID in dictionary with
        format: ID -> [start, end], return this dictionary instead of
        two vp dics.

    >>> seqs_dic = {"id1": "acguACGUacgu", "id2": "ACGUacgu"}
    >>> vp_s, vp_e = extract_viewpoint_regions_from_fasta(seqs_dic)
    >>> vp_s["id1"] == 5
    True
    >>> vp_e["id1"] == 8
    True
    >>> vp_s["id2"] == 1
    True
    >>> vp_e["id2"] == 4
    True
    >>> vp_s, vp_e = extract_viewpoint_regions_from_fasta(seqs_dic, center_vp=True)
    >>> vp_s["id1"] == 7
    True
    >>> vp_e["id1"] == 7
    True
    >>> vp_s, vp_e = extract_viewpoint_regions_from_fasta(seqs_dic, center_vp=True, vp_ext=2)
    >>> vp_s["id1"] == 5
    True
    >>> vp_e["id1"] == 9
    True
    >>> vp_s, vp_e = extract_viewpoint_regions_from_fasta(seqs_dic, center_vp=True, vp_ext=3)
    >>> vp_s["id2"] == 1
    True
    >>> vp_e["id2"] == 6
    True
    >>> extract_viewpoint_regions_from_fasta(seqs_dic, get_se_dic=True)
    {'id1': [5, 8], 'id2': [1, 4]}

    """
    if not vp_s_dic:
        vp_s_dic = {}
    if not vp_e_dic:
        vp_e_dic = {}
    if id2se_dic is None:
        id2se_dic = {}
    # Sanity check vp_ext.
    if vp_ext:
        if vp_ext < 0 or vp_ext > 100:
            print ("ERROR: vp_ext set too high (vp_ext=%i) Set vp_ext between 0 and 100." % (vp_ext))
            sys.exit()
    # Get viewpoint starts+ends (= uppercase nucleotide labels) for each sequence.
    for seq_id, seq in sorted(seqs_dic.items()):
        l_seq = len(seq)
        m = re.search("([acgun]*)([ACGUN]+)", seq)
        if m:
            l_us = len(m.group(1))
            l_vp = len(m.group(2))
            # If viewpoint length 0.
            if not l_vp:
                print ("ERROR: no viewpoint found for \"%s\"" % (seq_id))
                sys.exit()
            vp_s = l_us+1
            vp_e = l_us+l_vp
            # If center_vp, center viewpoint region.
            if center_vp:
                 vp_s = round(vp_s + (l_vp/2))
                 vp_e = vp_s
            # If vp_ext, extend vp region by vp_ext (new vp region).
            if vp_ext:
                vp_s = vp_s - vp_ext
                if vp_s < 1:
                    vp_s = 1
                vp_e = vp_e + vp_ext
                if vp_e > l_seq:
                    vp_e = l_seq
            # Store coordinates in hash.
            if get_se_dic:
                id2se_dic[seq_id] = [vp_s, vp_e]
            else:
                vp_s_dic[seq_id] = vp_s
                vp_e_dic[seq_id] = vp_e
        else:
            print ("ERROR: viewpoint extraction failed for \"%s\"" % (seq_id))
            sys.exit()
    if get_se_dic:
        assert id2se_dic, "id2se_dic empty"
        return id2se_dic
    else:
        return vp_s_dic, vp_e_dic


################################################################################

def extract_uc_region_coords_from_fasta(seqs_dic,
                                        uc_coords_dic=False):
    """
    Extract uppercase (viewpoint) start + end positions from FASTA dictionary.
    Return dictionary with vectors [start,end] (1-based indices, key:fasta_id).

    >>> seqs_dic = {"id1": "acguACGUacgu", "id2": "ACGUacgu"}
    >>> extract_uc_region_coords_from_fasta(seqs_dic)
    {'id1': [5, 8], 'id2': [1, 4]}

    """
    assert seqs_dic, "given seqs_dic empty"
    if not uc_coords_dic:
        uc_coords_dic = {}
    # Get uppercase region coordinates for each sequence.
    for seq_id, seq in sorted(seqs_dic.items()):
        l_seq = len(seq)
        m = re.search("([acgun]*)([ACGUN]+)", seq)
        if m:
            l_us = len(m.group(1))
            l_vp = len(m.group(2))
            # If viewpoint length 0.
            assert l_vp, "no viewpoint found for \"%s\"" % (seq_id)
            vp_s = l_us+1
            vp_e = l_us+l_vp
            uc_coords_dic[seq_id] = [vp_s, vp_e]
        else:
            assert False, "viewpoint extraction failed for ID \"%s\" with sequence \"%s\"" % (seq_id, seq)
    return uc_coords_dic


################################################################################

def read_str_elem_p_into_dic(str_elem_p_file,
                             p_to_str=False,
                             str_elem_p_dic=False):

    """
    Read in structural elements unpaired probabilities for each sequence
    position. Available structural elements:
    p_unpaired, p_external, p_hairpin, p_internal, p_multiloop, p_paired
    Input Format:
    >sequence_id
    p_unpaired<t>p_external<t>p_hairpin<t>p_internal<t>p_multiloop<t>p_paired
    Read values into dictionary with sequence ID -> 2d list mapping, and
    return dictionary.

    p_to_str:
        Read in probs as strings.

    Example input:
    >CLIP_01
    0.9	0.1	0.2	0.4	0.2	0.1
    0.8	0.2	0.3	0.2	0.1	0.2
    Resulting dictionary:
    d = {'CLIP_01': [[0.9, 0.1, 0.2, 0.4, 0.2, 0.1], [0.8, 0.2, 0.3, 0.2, 0.1, 0.2]]}
    print(d["CLIP_01"][0])
    [0.9, 0.1, 0.2, 0.4, 0.2, 0.1]
    print(d["CLIP_01"][0][0])
    0.9

    >>> str_elem_up_test = "test_data/test.elem_p.str"
    >>> read_str_elem_p_into_dic(str_elem_up_test)
    {'CLIP_01': [[0.9, 0.1, 0.2, 0.4, 0.2, 0.1], [0.8, 0.2, 0.3, 0.2, 0.1, 0.2]]}

    """
    if not str_elem_p_dic:
        str_elem_p_dic = {}
    seq_id = ""
    # Read in structural elements probabilities from file.
    with open(str_elem_p_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                str_elem_p_dic[seq_id] = []
            else:
                pl = line.strip().split("\t")
                if p_to_str:
                    str_elem_p_dic[seq_id].append(pl)
                else:
                    fpl = [float(p) for p in pl]
                    str_elem_p_dic[seq_id].append(fpl)
    f.closed
    return str_elem_p_dic


################################################################################

def read_con_into_dic(con_file,
                      sc_to_str=False,
                      con_dic=False):
    """
    Read in conservation scores (phastCons or phyloP) and store scores for
    each sequence ID in a vector. Resulting dictionary:
    sequence ID -> scores vector

    Example.con file format:
    >CLIP_01
    0.1
    0.2
    >CLIP_02
    0.4
    0.5
    ...

    con_dic:
        If given, add entries to existing dictionary.

    >>> con_test = "test_data/test.pp.con"
    >>> read_con_into_dic(con_test)
    {'CLIP_01': [0.1, 0.2], 'CLIP_02': [0.4, 0.5]}

    """
    if not con_dic:
        con_dic = {}
    seq_id = ""
    # Read in scores.
    with open(con_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                con_dic[seq_id] = []
            else:
                score = line.strip()
                if sc_to_str:
                    con_dic[seq_id].append(score)
                else:
                    score = float(score)
                    if not score % 1:
                        score = int(score)
                    con_dic[seq_id].append(score)
    f.closed
    assert con_dic, "con_dic empty"
    return con_dic


################################################################################

def mean_normalize(x, mean_x, max_x, min_x):
    """
    Mean normalization of input x, given dataset mean, max, and min.

    >>> mean_normalize(10, 10, 15, 5)
    0.0
    >>> mean_normalize(15, 20, 30, 10)
    -0.25

    Formula from:
    https://en.wikipedia.org/wiki/Feature_scaling

    """
    # If min=max, all values the same, so return x.
    if (max_x - min_x) == 0:
        return x
    else:
        return ( (x-mean_x) / (max_x - min_x) )


################################################################################

def min_max_normalize(x, max_x, min_x,
                      borders=False):
    """
    Min-max normalization of input x, given dataset max and min.

    >>> min_max_normalize(20, 30, 10)
    0.5
    >>> min_max_normalize(30, 30, 10)
    1.0
    >>> min_max_normalize(10, 30, 10)
    0.0
    >>> min_max_normalize(0.5, 1, 0, borders=[-1, 1])
    0.0

    Formula from:
    https://en.wikipedia.org/wiki/Feature_scaling

    """
    # If min=max, all values the same, so return x.
    if (max_x - min_x) == 0:
        return x
    else:
        if borders:
            assert len(borders) == 2, "list of 2 values expected"
            a = borders[0]
            b = borders[1]
            assert a < b, "a should be < b"
            return a + (x-min_x)*(b-a) / (max_x - min_x)
        else:
            return (x-min_x) / (max_x - min_x)


################################################################################

def read_bpp_into_dic(bpp_file, vp_dic,
                      bpp_dic=False,
                      con_ext=False,
                      bps_mode=1):
    """
    Read in base pair probabilities and store information in list for
    each sequence, where region to extract values from is defined by
    viewpoint (vp) start+end given in vp_dic.
    Return dictionary with base pair+probability list for each sequence
    (key: sequence id, value: "bp_start-bp_end,bp_prob").
    bps_mode: define which base pairs get extracted.
    bps_mode=1 : bps in extended vp region with start or end in base vp
    bps_mode=2 : only bps with start+end in base vp

    >>> bpp_test = "test_data/test.bpp"
    >>> vp_dic = {"CLIP_01": [150, 250]}
    >>> read_bpp_into_dic(bpp_test, vp_dic, bps_mode=1)
    {'CLIP_01': ['130-150,0.33', '160-200,0.44', '240-260,0.55']}
    >>> read_bpp_into_dic(bpp_test, vp_dic, bps_mode=2)
    {'CLIP_01': ['160-200,0.44']}

    """
    if not bpp_dic:
        bpp_dic = {}
    seq_id = ""
    # Go through base pairs file, extract sequences.
    with open(bpp_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                assert seq_id in vp_dic, "bpp_file ID \"%s\" not found in vp_dic" % ( seq_id)
                bpp_dic[seq_id] = []
            else:
                m = re.search("(\d+)\t(\d+)\t(.+)", line)
                s = int(m.group(1))
                e = int(m.group(2))
                bpp = float(m.group(3))
                bpp_se = "%s-%s,%s" % (s,e,bpp)
                if bps_mode == 1:
                    #if (s >= (vp_dic[seq_id][0]-con_ext) and (e <= (vp_dic[seq_id][1]) and e >= vp_dic[seq_id][0])) or (e <= (vp_dic[seq_id][1]+con_ext) and (s <= (vp_dic[seq_id][1]) and s >= vp_dic[seq_id][0])):
                    if (e >= vp_dic[seq_id][0] and e <= vp_dic[seq_id][1]) or (s >= vp_dic[seq_id][0] and s <= vp_dic[seq_id][1]):
                        bpp_dic[seq_id].append(bpp_se)
                elif bps_mode == 2:
                    if s >= vp_dic[seq_id][0] and e <= vp_dic[seq_id][1]:
                        bpp_dic[seq_id].append(bpp_se)
                else:
                    assert False, "ERROR: invalid bps_mode given (valid values: 1,2)"
    f.closed
    return bpp_dic


################################################################################

def convert_graph_to_string(g):
    """
    Convert graph into string of graph edges for string comparisons.
    E.g. "1-2,2-3,3-5,,4-5,4-8,5-6,6-7,7-8,"
    This graph has backbone edges from 1 to 8 plus one basepair edge from 4-8.

    Expected output:
Oc:clodxdl'.,;:cclooddxxkkkkkOOOOOOOOOOOOOkkkkO00O00000000O0KNNNNNNNNNNNNNNNNNNN
Kl:cloddo;.';;:cloodddxxkkkOOOOOOOO00OOOOOkkkOO00000000000000KNNNNNNNNNNNNNNNNNN
Xd:clool:..,;:clloodddxxkkOOOOOOOO000OOO0OOkkkO000000000000000KNNNNNNNNNNNNNNNNN
NOccclc:'.',;::cllooddxxkkkOOOOOOOO00O0000OOOOO0000000000000000KNNNNNNNNNNNNNNNN
WKoccc:,..,;;;::ccllooddxxkkOOOOOOOOOO00000OOO000000000000OkxxdxOXNNNNNNNNNNNNNN
NNkcc:;'.',........',,;:cclodxkkkxkkkkkkkkkxxxxxxddoolcc;,'.....'oXNNNNNNNNNNNNN
NN0o::;'..               ...........................        ...'';kNNNNNNNNNNNNN
NNNxc:;'.  ..',,;;;,,'...                                .,:ldkOOOOKNNNNNNNNNNNN
NNNKo:;'.';:cloodxxkkkkkxol:,'.                       .;lx00KKKKKKKKKXNNNNNNNNNN
NNNNkc:;;:::clooddxxkkOOO0000Oxoc,.          .  ..  .cx0KKKKKKKKKKXKKKXNNNNNNNNN
NNNNKdcc:;::cclloodxxxkkOOO0000KK0xl'.          . .;xO0KKKKKKKKKKKKKKKKXNNNNNNNN
NNNNNOo:;;;;::cclooddxxkkkOOO000KKKK0d,.         .cxO00KKKKKKKKKKKKKK00KNNNNNNNN
NNNNNXx:;;;;;::ccloooddxxkkkOO000KKKKK0l,.      .:xkO00KKKKKKKKKKKK00000XNNNNNNN
NNNNNXd;;;;;;:::clloooddxxxkkOOO00KKKKK00d.    .;dxkO00KKKKKKKKK00000000KXNNNNNN
NNNNNKl;;;;;;:::cclloooddxxxkkkOO000KKKKK0o.   'cdxkOO0000000000000000000XNNNNNN
NNNNNO:;;;;;;:::ccllloodddxxxkkkOO000KKKKK0:  .;ldxkkOOO00000000000000000KXNNNNN
NNNNXd;;;;;;;;::cclllooodddxxxkkkOOO0000K00d..':lodxkkkOOO00000OOOOO0OO000XNNNNN
NNNN0l;;;;;;;;::ccclllooodddxxxkkkOOO000000k,.;:loddxxkkkOOOOOOOOOOOOOOOO0KNNNNN
NNNNx:;;;;;;;;::cccllloooddddxxxkkkOOOOO000k:';:clodddxxkkkOOOOOOOOOOOOOOO0XNNNN
NNNKl;;;;;;;;;:::ccllllooodddxxxxkkkkkOOOOOx:';:ccloodddxxkkOOOOOOOOOOOOOO0KNNNN
NNNk:;;;;;;;;;:::cclllloooddddxxxxxkkkkkkkkd;';::ccllooddxxkkkOOOOOOOOOOOOOKNNNN
NWKo;;;;;;;;;;:::ccllllooooddddxxxxxkkkkkxxc'',;::ccllooddxxkkkOOOOOOOOOOOO0XNNN
NN0c;;;;;;;;;;:::cclllllooooddddxxxxxxxxxxo,.',,;::ccllooddxxkkkOOOOOOOOOOO0KNNN
NNk:;;;;;;;;;;;:::ccllllloooooddddxxdddddo;..'',,;::ccclooddxxxkkkkkkOkkOOOO0XNN
WXd;;;;;::::::::::ccclllllooooodddddddooodl;''',,;;:::cclooddxxxkkkkkkkkOOOO0XNN
WKo;;;;:::::::::::cccclllloooooooooooloxOKN0c,,,,;;;:::cclloodddxxxkkkkkOOOOOKNN
W0l;;;;:::ccccccccccccclllllllllllllox0XNNWXo'',;;;;;;::cccllooddxxkkkkkOOOOO0XW
W0c;;;;:::ccccccccclllllllllllllcc:oKNNNNNWXc.....'',;;;:::ccloodxxxkkkOOOOOO0XN
W0c;;;;:::ccccllllllllooooooollccc:dXWNNNNWK:........'',;;:clloddxxkkkkOOOOOOOKN
WKl;;;;:::ccllllloooooddddddddddddoxKWNNNNWKc...'',,,;;::cllooddxxkkkkOOOOOOOOKN
WKo;;;;;::ccllllooodddddxxxxxxxxxxdxKWNWNNWNo'''',;::cccclloooddxxkkkkOOOOOOOOKN
WXd;;;;;::cclllloooddddxxxxxxxxxxxdkXWNWWWWWO:'',,;:ccllllooodddxxkkkkkOOOOOOOKW

    """
    g_string = ""
    for s,e in g.edges:
        g_string += "%i-%i," % (s,e)
    return g_string


################################################################################

def read_ids_into_dic(ids_file,
                      ids_dic=False):
    """
    Read in IDs file, where each line stores one ID.

    >>> test_ids_file = "test_data/test.ids"
    >>> ids_dic = read_ids_into_dic(test_ids_file)
    >>> print(ids_dic)
    {'clip1': 1, 'clip2': 1, 'clip3': 1}

    """
    if not ids_dic:
        ids_dic = {}
    # Read in file content.
    with open(ids_file) as f:
        for line in f:
            row_id = line.strip()
            ids_dic[row_id] = 1
    f.closed
    assert ids_dic, "IDs dictionary ids_dic empty"
    return ids_dic


################################################################################

def bed_convert_coords(reg_s, reg_e, ref_s, ref_e, pol):
    """
    Convert BED coordinates derived from a region within a subsequence
    of the reference (chromsome or transcript) to chromosome or
    transcript reference coordinates. Return start + end (BED format)
    positions.

    reg_s:
        Region within subsequence start position,
        with 0 to len(subsequence) coordinates
    reg_e:
        Region within subsequence end position,
        with 0 to len(subsequence) coordinates
    ref_s:
        Subsequence start position on reference sequence
    ref_e:
        Subsequence end position on reference sequence
    pol:
        Polarity on reference (use + for transcripts, + or - for
        chromosomal regions)

    >>> bed_convert_coords(10, 20, 1000, 2000, "+")
    (1010, 1020)
    >>> bed_convert_coords(10, 20, 1000, 2000, "-")
    (1980, 1990)

    """
    assert pol == "+" or pol == "-", "invalid polarity given"
    assert reg_s < reg_e, "Invalid BED coordinates given: reg_s >= reg_e"
    assert ref_s < ref_e, "Invalid BED coordinates given: ref_s >= ref_e"
    new_s = ref_s + reg_s
    new_e = ref_s + reg_e
    if pol == "-":
        new_s = ref_e - reg_e
        new_e = ref_e - reg_s
    return new_s, new_e


################################################################################

def list_extract_peaks(in_list,
                       max_merge_dist=0,
                       coords="list",
                       sc_thr=0):
    """
    Extract peak regions from list.
    Peak region is defined as region >= score threshold.
    Return list of lists with format:
    [pr_s, pr_e, pr_top_pos, pr_top_sc]

    coords=bed  :  peak start 0-based, peak end 1-based.
    coords=list :  peak start 0-based, peak end 0-based.

    >>> test_list = [-1, 0, 2, 4.5, 1, -1, 5, 6.5]
    >>> list_extract_peaks(test_list)
    [[1, 4, 3, 4.5], [6, 7, 7, 6.5]]
    >>> list_extract_peaks(test_list, sc_thr=2)
    [[2, 3, 3, 4.5], [6, 7, 7, 6.5]]
    >>> list_extract_peaks(test_list, sc_thr=2, coords="bed")
    [[2, 4, 4, 4.5], [6, 8, 8, 6.5]]
    >>> list_extract_peaks(test_list, sc_thr=10)
    []
    >>> test_list = [2, -1, 3, -1, 4, -1, -1, 6, 9]
    >>> list_extract_peaks(test_list, max_merge_dist=2)
    [[0, 4, 4, 4], [7, 8, 8, 9]]
    >>> list_extract_peaks(test_list, max_merge_dist=3)
    [[0, 8, 8, 9]]

    """
    # Check.
    assert len(in_list), "Given list is empty"
    # Peak regions list.
    peak_list = []
    # Help me.
    inside = False
    pr_s = 0
    pr_e = 0
    pr_top_pos = 0
    pr_top_sc = -100000
    for i, sc in enumerate(in_list):
        # Part of peak region?
        if sc >= sc_thr:
            # At peak start.
            if not inside:
                pr_s = i
                pr_e = i
                inside = True
            else:
                # Inside peak region.
                pr_e = i
            # Store top position.
            if sc > pr_top_sc:
                pr_top_sc = sc
                pr_top_pos = i
        else:
            # Before was peak region?
            if inside:
                # Store peak region.
                #peak_infos = "%i,%i,%i,%f" %(pr_s, pr_e, pr_top_pos, pr_top_sc)
                peak_infos = [pr_s, pr_e, pr_top_pos, pr_top_sc]
                peak_list.append(peak_infos)
                inside = False
                pr_top_pos = 0
                pr_top_sc = -100000
    # If peak at the end, also report.
    if inside:
        # Store peak region.
        peak_infos = [pr_s, pr_e, pr_top_pos, pr_top_sc]
        peak_list.append(peak_infos)
    # Merge peaks.
    if max_merge_dist and len(peak_list) > 1:
        iterate = True
        while iterate:
            merged_peak_list = []
            added_peaks_dic = {}
            peaks_merged = False
            for i, l in enumerate(peak_list):
                if i in added_peaks_dic:
                    continue
                j = i + 1
                # Last element.
                if j == len(peak_list):
                    if i not in added_peaks_dic:
                        merged_peak_list.append(peak_list[i])
                    break
                # Compare two elements.
                new_peak = []
                if (peak_list[j][0] - peak_list[i][1]) <= max_merge_dist:
                    peaks_merged = True
                    new_top_pos = peak_list[i][2]
                    new_top_sc = peak_list[i][3]
                    if peak_list[i][3] < peak_list[j][3]:
                        new_top_pos = peak_list[j][2]
                        new_top_sc = peak_list[j][3]
                    new_peak = [peak_list[i][0], peak_list[j][1], new_top_pos, new_top_sc]
                # If two peaks were merged.
                if new_peak:
                    merged_peak_list.append(new_peak)
                    added_peaks_dic[i] = 1
                    added_peaks_dic[j] = 1
                else:
                    merged_peak_list.append(peak_list[i])
                    added_peaks_dic[i] = 1
            if not peaks_merged:
                iterate = False
            peak_list = merged_peak_list
            peaks_merged = False
    # If peak coordinates should be in .bed format, make peak ends 1-based.
    if coords == "bed":
        for i in range(len(peak_list)):
            peak_list[i][1] += 1
            peak_list[i][2] += 1 # 1-base best score position too.
    return peak_list


################################################################################

def is_tool(name):
    """Check whether tool "name" is in PATH."""
    return find_executable(name) is not None


################################################################################

def count_file_rows(in_file):
    """
    Count number of file rows for given input file.

    >>> test_file = "test_data/test1.bed"
    >>> count_file_rows(test_file)
    7
    >>> test_file = "test_data/empty_file"
    >>> count_file_rows(test_file)
    0

    """
    check_cmd = "cat " + in_file + " | wc -l"
    output = subprocess.getoutput(check_cmd)
    row_count = int(output.strip())
    return row_count


################################################################################

def bed_check_six_col_format(bed_file):
    """
    Check whether given .bed file has 6 columns.

    >>> test_bed = "test_data/test1.bed"
    >>> bed_check_six_col_format(test_bed)
    True
    >>> test_bed = "test_data/empty_file"
    >>> bed_check_six_col_format(test_bed)
    False

    """

    six_col_format = False
    with open(bed_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) == 6:
                six_col_format = True
            break
    f.closed
    return six_col_format


################################################################################

def count_file_rows(in_file,
                    nr_cols=False):
    """
    Count number of file rows. If nr_cols set, demand certain (nr_cols) number
    of columns (separated by tab), in order for row to be counted.

    >>> test_file = "test_data/test1.bed"
    >>> count_file_rows(test_file)
    7
    >>> test_file = "test_data/empty_file"
    >>> count_file_rows(test_file)
    0

    """
    c = 0
    with open(in_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            if nr_cols:
                if len(cols) == nr_cols:
                    c += 1
            else:
                c += 1
    f.closed
    return c


################################################################################

def bpp_callback(v, v_size, i, maxsize, what, data=None):
    """
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.13).
    So RNA.py needs to be in PYTHONPATH (it is if installed via conda).

    """
    import RNA
    if what & RNA.PROBS_WINDOW_BPP:
        data.extend([{'i': i, 'j': j, 'p': p} for j, p in enumerate(v) if (p is not None) and (p >= 0.01)])


################################################################################

def up_split_callback(v, v_size, i, maxsize, what, data):
    """
    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.13).
    So RNA.py needs to be in PYTHONPATH (it is if installed via conda).

    """
    import RNA
    if what & RNA.PROBS_WINDOW_UP:
        what = what & ~RNA.PROBS_WINDOW_UP
        dat = []
        # Non-split case:
        if what == RNA.ANY_LOOP:
                dat = data
        # all the cases where probability is split into different loop contexts
        elif what == RNA.EXT_LOOP:
                dat = data['ext']
        elif what == RNA.HP_LOOP:
                dat = data['hp']
        elif what == RNA.INT_LOOP:
                dat = data['int']
        elif what == RNA.MB_LOOP:
                dat = data['mb']
        dat.append({'i': i, 'up': v})


################################################################################

def calc_str_elem_up_bpp(in_fasta, out_bpp, out_str,
                         out_miss=False,
                         missing_ids_dic=None,
                         report=True,
                         stats_dic=None,
                         id2ucr_dic=False,
                         plfold_u=3,
                         plfold_l=100,
                         plfold_w=150):
    """
    Calculate structural elements probabilities (different loop contexts),
    as well as base pairs and their probabilities, using ViennaRNA.

    This uses the Python3 API (RNA.py) of ViennaRNA (tested with v 2.4.14).
    So RNA.py needs to be in PYTHONPATH, which it is,
    if e.g. installed via:
    conda install -c bioconda viennarna=2.4.14

    If no base pairs found for sequence, still print out ID header to
    out_bpp (just no base pair rows following).

    in_fasta:
        Input FASTA file
    out_bpp:
        Output base pair probabilities file
    out_str:
        Output position-wise structural elements probabilities file
    out_miss:
        Output file to store FASTA IDs for which no BPs were found
    missing_ids_dic:
        Optionally, store missing IDs in missing_ids_dic. Like out_miss
        obsolete, since all sequence IDs are output even if no base pairs
        found for a given sequence.
    stats_dic:
        If not None, extract statistics from structure data and store
        in stats_dic.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        Set to define regions for which to generate element probability
        stats, stored in stats_dic.
    plfold_u:
        RNAplfold -u parameter value
    plfold_l:
        RNAplfold -L parameter value
    plfold_w:
        RNAplfold -W parameter value

    """

    # Import RNA.py library.
    try:
        import RNA
    except:
        assert False, "ViennaRNA Python3 API library RNA.py not in PYTHONPATH"
    # For real.
    import RNA

    # Check input.
    assert os.path.isfile(in_fasta), "cannot open target FASTA file \"%s\"" % (in_fasta)

    # Read in FASTA file.
    seqs_dic = read_fasta_into_dic(in_fasta)

    # If stats dictionary given, compute statistics during run.
    if stats_dic is not None:
        stats_dic["bp_c"] = 0
        stats_dic["seqlen_sum"] = 0
        stats_dic["nobpsites_c"] = 0
        stats_dic["seq_c"] = len(seqs_dic)
        pu_list = []
        ps_list = []
        pe_list = []
        ph_list = []
        pi_list = []
        pm_list = []
        pbp_list = []

    # Output files.
    OUTBPP = open(out_bpp,"w")
    OUTSTR = open(out_str,"w")
    if out_miss:
        OUTMISS = open(out_miss,"w")

    # Floor float, for centering probabilities (important when u > 1).
    i_add = int(plfold_u/2)

    # Sequence counter.
    c_seq = 0

    # Calculate base pair and structural elements probabilities.
    if report:
        print("Calculate base pair and structural elements probabilities ... ")

    for seq_id, seq in sorted(seqs_dic.items()):

        md = RNA.md()
        md.max_bp_span = plfold_l
        md.window_size = plfold_w

        # Get base pairs and their probabilities.
        data = []
        # Different loop context probabilities.
        data_split = {'ext': [], 'hp': [], 'int': [], 'mb': [] }

        fc = RNA.fold_compound(seq, md, RNA.OPTION_WINDOW)
        # Get base pairs and their probabilities.
        fc.probs_window(plfold_u, RNA.PROBS_WINDOW_BPP, bpp_callback, data)
        # Get different loop context probabilities.
        fc.probs_window(plfold_u, RNA.PROBS_WINDOW_UP | RNA.PROBS_WINDOW_UP_SPLIT, up_split_callback, data_split)

        # If base pairs found.
        if data:
            # Output base pair probabilities.
            OUTBPP.write(">%s\n" %(seq_id))
            for prob in data:
                p = prob['p']
                i = prob['i']
                j = prob['j']
                OUTBPP.write("%i\t%i\t%f\n" %(i,j,p))
                if stats_dic:
                    stats_dic["bp_c"] += 1
                    pbp_list.append(p)
        else:
            if report:
                print("WARNING: no base pairs found for \"%s\"" %(seq_id))
            # Still print header.
            OUTBPP.write(">%s\n" %(seq_id))
            if stats_dic:
                stats_dic["nobpsites_c"] += 1
            if out_miss:
                OUTMISS.write("%s\n" %(seq_id))
            if missing_ids_dic is not None:
                missing_ids_dic[seq_id] = 1

        # Store individual probs for sequence in lists.
        ups = []
        ups_e = []
        ups_h = []
        ups_i = []
        ups_m = []
        ups_s = []

        for i,e in enumerate(seq):
            data_i = i + 1
            p_e = 0
            p_h = 0
            p_i = 0
            p_m = 0
            if data_split['ext'][i]['up'][plfold_u]:
                p_e = data_split['ext'][i]['up'][plfold_u]
            if data_split['hp'][i]['up'][plfold_u]:
                p_h = data_split['hp'][i]['up'][plfold_u]
            if data_split['int'][i]['up'][plfold_u]:
                p_i = data_split['int'][i]['up'][plfold_u]
            if data_split['mb'][i]['up'][plfold_u]:
                p_m = data_split['mb'][i]['up'][plfold_u]
            # Total unpaired prob = sum of different loop context probs.
            p_u = p_e + p_h + p_i + p_m
            if p_u > 1:
                p_u = 1
            # Paired prob (stacked prob).
            p_s = 1 - p_u
            ups.append(p_u)
            ups_e.append(p_e)
            ups_h.append(p_h)
            ups_i.append(p_i)
            ups_m.append(p_m)
            ups_s.append(p_s)

        # Center the values and output for each sequence position.
        OUTSTR.write(">%s\n" %(seq_id))
        l_seq = len(seq)
        if stats_dic is not None:
            stats_dic["seqlen_sum"] += l_seq
        for i, c in enumerate(seq):
            # At start, end, and middle.
            if i < i_add:
                p_u = ups[plfold_u-1]
                p_e = ups_e[plfold_u-1]
                p_h = ups_h[plfold_u-1]
                p_i = ups_i[plfold_u-1]
                p_m = ups_m[plfold_u-1]
                p_s = ups_s[plfold_u-1]
            elif i >= (l_seq - i_add):
                p_u = ups[l_seq-1]
                p_e = ups_e[l_seq-1]
                p_h = ups_h[l_seq-1]
                p_i = ups_i[l_seq-1]
                p_m = ups_m[l_seq-1]
                p_s = ups_s[l_seq-1]
            else:
                p_u = ups[i+i_add]
                p_e = ups_e[i+i_add]
                p_h = ups_h[i+i_add]
                p_i = ups_i[i+i_add]
                p_m = ups_m[i+i_add]
                p_s = ups_s[i+i_add]
            # Output centered values.
            pos = i+1 # one-based sequence position.
            #OUTSTR.write("%i\t%f\t%f\t%f\t%f\t%f\t%f\n" %(pos,p_u,p_e,p_h,p_i,p_m,p_s))
            OUTSTR.write("%f\t%f\t%f\t%f\t%f\n" %(p_e,p_h,p_i,p_m,p_s))
            if stats_dic:
                if id2ucr_dic:
                    # If id2ucr_dic, record values only for uppercase part of sequence.
                    uc_s = id2ucr_dic[seq_id][0]
                    uc_e = id2ucr_dic[seq_id][1]
                    if pos >= uc_s and pos <= uc_e:
                        pu_list.append(p_u)
                        ps_list.append(p_s)
                        pe_list.append(p_e)
                        ph_list.append(p_h)
                        pi_list.append(p_i)
                        pm_list.append(p_m)
                else:
                    pu_list.append(p_u)
                    ps_list.append(p_s)
                    pe_list.append(p_e)
                    ph_list.append(p_h)
                    pi_list.append(p_i)
                    pm_list.append(p_m)

        c_seq += 1
        if report:
            if not c_seq % 100:
                print("%i sequences processed" %(c_seq))

    OUTBPP.close()
    OUTSTR.close()
    if out_miss:
        OUTMISS.close()

    # Calculate stats if stats_dic set.
    if stats_dic:
        # Mean values.
        stats_dic["U"] = [statistics.mean(pu_list)]
        stats_dic["S"] = [statistics.mean(ps_list)]
        stats_dic["E"] = [statistics.mean(pe_list)]
        stats_dic["H"] = [statistics.mean(ph_list)]
        stats_dic["I"] = [statistics.mean(pi_list)]
        stats_dic["M"] = [statistics.mean(pm_list)]
        stats_dic["bp_p"] = [statistics.mean(pbp_list)]
        # Standard deviations.
        stats_dic["U"] += [statistics.stdev(pu_list)]
        stats_dic["S"] += [statistics.stdev(ps_list)]
        stats_dic["E"] += [statistics.stdev(pe_list)]
        stats_dic["H"] += [statistics.stdev(ph_list)]
        stats_dic["I"] += [statistics.stdev(pi_list)]
        stats_dic["M"] += [statistics.stdev(pm_list)]
        stats_dic["bp_p"] += [statistics.stdev(pbp_list)]


################################################################################

def bed_read_rows_into_dic(in_bed,
                           exon_regions=False,
                           id2exonc_dic=None,
                           id2len_dic=None,
                           two_ids_dic=False):
    """
    Read in .bed file rows into dictionary.
    Mapping is region ID -> bed row.
    If exon_regions=True, a set of regions defines a transcript if column 5
    ID is the same for all these regions. Then by start,end + polarity,
    exon IDs are added. E.g. "id" -> "id_e1", "id_e2" ...

    exon_regions:
        Allow column 5 IDs to be non-unique, and treated as exon regions.
        This results in new IDs for these regions (_e1, _e2 ...)
    id2exonc_dic:
        Site ID to exon count dictionary.
    two_ids_dic:
        Dictionary with site ID -> sequence ID, used for filtering sites.
        Thus, row has to have both site and sequence ID to be kept.

    >>> id2exonc_dic = {}
    >>> id2len_dic = {}
    >>> id2row_dic = {}
    >>> test_bed = "test_data/test2.bed"
    >>> id2row_dic = bed_read_rows_into_dic(test_bed, id2exonc_dic=id2exonc_dic, exon_regions=True, id2len_dic=id2len_dic)
    >>> id2exonc_dic
    {'reg1': 4, 'reg2': 3, 'reg3': 1}
    >>> id2len_dic
    {'reg1': 2200, 'reg2': 2100, 'reg3': 60000}

    """

    newid2row_dic = {}
    id2pol_dic = {}
    id2sc_dic = {}
    id2starts_dic = {}

    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            seq_id = cols[0]
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_l = site_e - site_s
            site_id = cols[3]
            site_sc = cols[4]
            site_pol = cols[5]
            # Store polarity, if not exon regions, check if site_id is unique.
            if not exon_regions:
                # If site_id non-unique.
                assert site_id not in id2pol_dic, "non-unique site ID (\"%s\") found in \"%s\"" %(site_id, in_bed)
            id2pol_dic[site_id] = site_pol
            id2sc_dic[site_id] = site_sc
            # Make new row string.
            new_row = "%s\t%i\t%i" %(seq_id, site_s, site_e)
            # Make new site ID, with column 5 ID + start position.
            new_site_id = "%s,%i" %(site_id, site_s)
            # Check whether new ID is unique.
            assert new_site_id not in newid2row_dic, "non-unique site ID + start position combination (\"%s\") found in \"%s\"" %(new_site_id, in_bed)
            # Store row for each new unique ID.
            newid2row_dic[new_site_id] = new_row
            # Store site starts list for each site ID.
            if site_id not in id2starts_dic:
                id2starts_dic[site_id] = [site_s]
            else:
                id2starts_dic[site_id].append(site_s)
            # Calculate total lengths for each site_id.
            if id2len_dic is not None:
                if site_id in id2len_dic:
                    id2len_dic[site_id] += site_l
                else:
                    id2len_dic[site_id] = site_l
    f.closed

    # Store sites with new site IDs.
    id2row_dic = {}
    for site_id in id2starts_dic:
        site_pol = id2pol_dic[site_id]
        # If site ID only has one .bed region.
        if len(id2starts_dic[site_id]) == 1:
            site_s = id2starts_dic[site_id][0]
            new_site_id = "%s,%i" %(site_id, site_s)
            id2row_dic[site_id] = newid2row_dic[new_site_id] + "\t%s\t%s\t%s" %(site_id, id2sc_dic[site_id], site_pol)
            if id2exonc_dic is not None:
                id2exonc_dic[site_id] = 1
        else:
            # Sort start positions ascending (+ strand) or descending (- strand).
            if site_pol == "+":
                id2starts_dic[site_id].sort()
            else:
                id2starts_dic[site_id].sort(reverse=True)
            exon_c = 0
            for site_s in id2starts_dic[site_id]:
                new_site_id = "%s,%i" %(site_id, site_s)
                row = newid2row_dic[new_site_id]
                exon_c += 1
                exon_id = site_id + "_e%i" %(exon_c)
                id2row_dic[exon_id] = newid2row_dic[new_site_id]  + "\t%s\t0\t%s" %(exon_id, site_pol)
                if id2exonc_dic is not None:
                    if site_id in id2exonc_dic:
                        id2exonc_dic[site_id] += 1
                    else:
                        id2exonc_dic[site_id] = 1
    return id2row_dic


################################################################################

def bed_write_row_dic_into_file(id2row_dic, out_bed,
                                id2out_dic=None):
    """
    Write .bed row dictionary (column 5 ID as key, .bed row as string)
    into .bed file.
    Example dictionary:
    {'reg1_e1': 'chr1\t1000\t1100\treg1_e1\t0\t+', ... }

    id2out_dic:
        IDs dictionary for which to output regions.

    """
    assert id2row_dic, "given id2row_dic empty"
    OUTBED = open(out_bed, "w")
    c_out = 0
    for site_id in id2row_dic:
        if id2out_dic is not None:
            if not site_id in id2out_dic:
                continue
        c_out += 1
        out_row = id2row_dic[site_id] + "\n"
        OUTBED.write(out_row)
    OUTBED.close()
    assert c_out, "nothing was output"


################################################################################

def bed_extract_sequences_from_2bit(in_bed, out_fa, in_2bit,
                                    lc_repeats=False,
                                    convert_to_rna=False):
    """
    Extract sequences from genome (provide genome .2bit file).
    twoBitToFa executable needs to be in PATH. Store extracted
    sequences in out_fa.

    convert_to_rna:
        If true, read in extracted sequences and convert to RNA.
    lc_repeats:
        If True, do not convert repeat regions to uppercas and output.

    >>> in_bed = "test_data/test_seq_extr.sites.bed"
    >>> tmp_2bit_fa = "test_data/test_seq_extr.sites.2bit.tmp.fa"
    >>> tmp_seq_fa = "test_data/test_seq_extr.sites.seq.tmp.fa"
    >>> exp_fa = "test_data/test_seq_extr.sites.exp.fa"
    >>> in_fa = "test_data/test_seq_extr.sequences.fa"
    >>> in_2bit = "test_data/test_seq_extr.sequences.2bit"
    >>> id2row_dic = bed_read_rows_into_dic(in_bed)
    >>> seqs_dic = read_fasta_into_dic(in_fa, dna=True)
    >>> id2seq_dic = extract_transcript_sequences(id2row_dic, seqs_dic, revcom=True)
    >>> fasta_output_dic(id2seq_dic, tmp_seq_fa)
    >>> bed_extract_sequences_from_2bit(in_bed, tmp_2bit_fa, in_2bit)
    >>> diff_two_files_identical(tmp_seq_fa, exp_fa)
    True
    >>> diff_two_files_identical(tmp_2bit_fa, exp_fa)
    True

    """
    # Check for twoBitToFa.
    assert is_tool("twoBitToFa"), "twoBitToFa not in PATH"

    # Run twoBitToFa and check.
    check_cmd = "twoBitToFa"
    if not lc_repeats:
        check_cmd += " -noMask"
    check_cmd += " -bed=" + in_bed + " " + in_2bit + " " + out_fa
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "twoBitToFa is complaining:\n%s\n%s" %(check_cmd, output)
    if convert_to_rna:
        # Read in tmp_fa into dictionary (this also converts sequences to RNA).
        seqs_dic = read_fasta_into_dic(out_fa)
        # Output RNA sequences.
        fasta_output_dic(seqs_dic, out_fa,
                         split=True)


################################################################################

def generate_random_fn(file_ending):
    """
    Generate a random file name for temporary files.

    """
    random_id = uuid.uuid1()
    random_fn = str(random_id) + ".tmp." . file_ending
    return random_fn


################################################################################

def bed_check_unique_ids(bed_file):
    """
    Check whether .bed file (6 column format with IDs in column 4)
    has unique column 4 IDs.

    >>> test_bed = "test_data/test1.bed"
    >>> bed_check_unique_ids(test_bed)
    True
    >>> test_bed = "test_data/test2.bed"
    >>> bed_check_unique_ids(test_bed)
    False

    """

    check_cmd = "cut -f 4 " + bed_file + " | sort | uniq -d"
    output = subprocess.getoutput(check_cmd)
    if output:
        return False
    else:
        return True


################################################################################

def bed_check_unique_ids_two_files(bed_file1, bed_file2):
    """
    Check whether .bed file (6 column format with IDs in column 4)
    has unique column 4 IDs.

    >>> test_bed1 = "test_data/test1.bed"
    >>> test_bed2 = "test_data/test4.bed"
    >>> bed_check_unique_ids_two_files(test_bed1, test_bed2)
    True
    >>> bed_check_unique_ids_two_files(test_bed1, test_bed1)
    False

    """

    check_cmd = "cut -f 4 " + bed_file1 + " " + bed_file2 + " | sort | uniq -d"
    output = subprocess.getoutput(check_cmd)
    if output:
        return False
    else:
        return True


################################################################################

def fasta_check_fasta_file(fasta_file):
    """
    Check whether given FASTA file is in FASTA format.

    >>> test_bed = "test_data/test1.bed"
    >>> fasta_check_fasta_file(test_bed)
    False
    >>> test_fa = "test_data/test.fa"
    >>> fasta_check_fasta_file(test_fa)
    True

    """
    # R.U.O.K ?
    assert os.path.isfile(fasta_file), "cannot open fasta_file \"%s\"" % (fasta_file)
    # Check first 50 FASTA entries.
    check_cmd = 'grep -A 1 ">" ' + fasta_file + ' | head -100 '
    output = subprocess.getoutput(check_cmd)
    if not output:
        return False
    # Go over output.
    lines = output.split("\n")
    c_headers = 0
    c_seqs = 0
    for line in lines:
        if re.search(">.+", line):
            c_headers += 1
        elif re.search("[ACGTUN]+", line, re.I):
            c_seqs += 1
        else:
            return False
    if c_headers and c_seqs:
        if c_headers == c_seqs:
            return True
        else:
            return False


################################################################################

def bed_get_region_lengths(bed_file,
                           id2pol_dic=None):
    """
    Read in .bed file, store and return region lengths in dictionary.
    key   :  region ID (.bed col4)
    value :  region length (.bed col3-col2)
    Additionally, also store polarities in id2pol_dic.

    >>> test_file = "test_data/test3.bed"
    >>> bed_get_region_lengths(test_file)
    {'CLIP1': 10, 'CLIP2': 15}

    """
    id2len_dic = {}
    with open(bed_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_pol = cols[5]
            site_l = site_e - site_s
            assert site_id not in id2len_dic, "column 4 IDs not unique in given .bed file \"%s\"" %(bed_file)
            if id2pol_dic is not None:
                id2pol_dic[site_id] = site_pol
            id2len_dic[site_id] = site_l
    f.closed
    assert id2len_dic, "No IDs read into dictionary (input file \"%s\" empty or malformatted?)" % (bed_file)
    return id2len_dic


################################################################################

def get_core_id_to_part_counts_dic(ids_dic,
                                   label_dic=None):
    """
    Get core ID to part count dictionary.
    E.g. for
    id1_p1, id1_p2
    we get:
    id1 -> 2
    Works for _p (split region parts) and _e (exon regions) ID extensions.

    >>> ids_dic = {'id1_p1': 1, 'id1_p2': 1, 'id2': 1, 'id3_e1': 1, 'id3_e2': 1}
    >>> get_core_id_to_part_counts_dic(ids_dic)
    {'id1': 2, 'id2': 1, 'id3': 2}
    >>> in_bed = "test_data/test_con.bed"
    >>> ids_dic = bed_get_region_ids(in_bed)
    >>> get_core_id_to_part_counts_dic(ids_dic)
    {'site1': 1, 'site2': 1, 'site3': 2, 'site4': 2}

    """
    # Checker.
    assert ids_dic, "given ids_dic empty"
    id2c_dic = {}
    for site_id in ids_dic:
        # Check if site ID is split site ID with _e or _p.
        if re.search('.+_[pe]\d+$', site_id):
            m = re.search('(.+)_([pe])\d+$', site_id)
            core_id = m.group(1)
            label = m.group(2)
            if label_dic is not None:
                label_dic[core_id] = label
            if core_id in id2c_dic:
                id2c_dic[core_id] += 1
            else:
                id2c_dic[core_id] = 1
        else:
            assert site_id not in id2c_dic, "non-unique site ID \"%s\" in ids_dic" %(site_id)
            id2c_dic[site_id] = 1
    # Check and litter.
    assert id2c_dic, "nothing read into id2c_dic"
    return id2c_dic


################################################################################

def bed_core_id_to_part_counts_dic(in_bed):
    """
    Get core ID to part count dictionary.
    E.g. for
    id1_p1, id1_p2
    we get:
    id1 -> 2
    Works for _p (split region parts) and _e (exon regions) ID extensions.

    >>> test_bed = "test_data/test8.bed"
    >>> bed_core_id_to_part_counts_dic(test_bed)
    {'reg1': 1, 'reg2': 2, 'reg3': 1}

    """
    id2c_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_id = cols[3]
            # Check if site ID is split site ID with _e or _p.
            if re.search('.+_[pe]\d+$', site_id):
                m = re.search('(.+)_[pe]\d+$', site_id)
                core_id = m.group(1)
                if core_id in id2c_dic:
                    id2c_dic[core_id] += 1
                else:
                    id2c_dic[core_id] = 1
            else:
                assert site_id not in id2c_dic, "non-unique site ID \"%s\" in BED file \"%s\"" %(site_id, in_bed)
                id2c_dic[site_id] = 1
    f.closed
    assert id2c_dic, "id2c_dic empty"
    return id2c_dic


################################################################################

def bed_check_for_part_ids(in_bed):
    """
    Check for part IDs in given BED file in_bed.
    E.g. IDs like
    id1_p1, id1_p2
    Works for _p (split region parts) and _e (exon regions) ID extensions.
    Return True if part IDs found, else False.

    >>> test_bed = "test_data/test8.bed"
    >>> bed_check_for_part_ids(test_bed)
    True

    """
    id2c_dic = {}
    part_ids_found = False
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_id = cols[3]
            if re.search('.+_[pe]\d+$', site_id):
                m = re.search('(.+)_[pe]\d+$', site_id)
                core_id = m.group(1)
                if core_id in id2c_dic:
                    part_ids_found = True
                    break
                else:
                    id2c_dic[core_id] = 1
            else:
                assert site_id not in id2c_dic, "non-unique site ID \"%s\" in BED file \"%s\"" %(site_id, in_bed)
                id2c_dic[site_id] = 1
    f.closed
    return part_ids_found


################################################################################

def extract_conservation_scores(in_bed, out_con, con_bw,
                                stats_dic=None,
                                id2ucr_dic=False,
                                merge_split_regions=True,
                                report=False):
    """
    Extract conservation scores for genomic regions given as in_bed BED file.
    Scores are extracted from .bigWig con_bw file, tested for phastCons
    and phyloP .bigWig files, downloaded from:

    http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw
    http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/hg38.phastCons100way.bw

    Ouput conservation scores for each region position to out_con with format:
    >site_id
    0.01
    0.03
    0
    ...

    stats_dic:
        If not None, extract statistics on conservation scores and store
        in stats_dic.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        Set to define regions for which to extract conservation score
        stats, stored in stats_dic.
    merge_split_regions:
        If True, merge regions with IDs id1_p1, id1_p2 .. or id1_e1, id1_e2 ..
        The function thus looks for IDs with _e or _p attached to core ID,
        and combines the regions with incrementing numbers (p1 p2 ... ).
        This should be used for split sites like transcript sites over exon
        borders, or exon regions of a transcript, to get one contigious
        list of conservation scores for the site / transcript.
    report:
        If True, output some logging information.

    This function uses:
    bedtools makewindows, bigWigAverageOverBed

    bedtools makewindows
    ====================
    Split in_bed regions into 1-nt regions.
    Given test.bed with region:
    chr1	100	105	site1	0	+
    bedtools makewindows (bedtools makewindows -b test.bed -w 1 -i srcwinnum)
    outputs:
    chr1	100	101	site1_1
    chr1	101	102	site1_2
    chr1	102	103	site1_3
    chr1	103	104	site1_4
    chr1	104	105	site1_5

    >>> in_bed = "test_data/test_con.bed"
    >>> out_con = "test_data/test_con.tmp.con"
    >>> exp_con = "test_data/test_con.exp.con"
    >>> out2_con = "test_data/test_con2.tmp.con"
    >>> exp2_con = "test_data/test_con2.exp.con"
    >>> con_bw = "test_data/test_con.bw"
    >>> extract_conservation_scores(in_bed, out_con, con_bw, merge_split_regions=True)
    >>> diff_two_files_identical(out_con, exp_con)
    True
    >>> extract_conservation_scores(in_bed, out2_con, con_bw, merge_split_regions=False)
    >>> diff_two_files_identical(out2_con, exp2_con)
    True

    """

    # Checks.
    assert is_tool("bedtools"), "bedtools not in PATH"
    assert is_tool("bigWigAverageOverBed"), "bigWigAverageOverBed not in PATH"
    assert os.path.isfile(in_bed), "cannot open in_bed file \"%s\"" % (in_bed)
    assert os.path.isfile(con_bw), "cannot open con_bw file \"%s\"" % (con_bw)
    assert bed_check_unique_ids(in_bed), "in_bed \"%s\" column 4 IDs not unique" % (in_bed)

    # Get region IDs, lengths, strand polarities.
    id2pol_dic = {}
    id2len_dic = bed_get_region_lengths(in_bed, id2pol_dic=id2pol_dic)
    # Get part counts for IDs (>1 for id_p or id_e IDs) to get exons / split sites.
    label_dic = {} # Split site ID to used label (e, p).
    id2parts_dic = get_core_id_to_part_counts_dic(id2len_dic, label_dic=label_dic)

    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"
    random_id = uuid.uuid1()
    tmp_tab = str(random_id) + ".tmp.tab"

    if stats_dic is not None:
        stats_dic["mean"] = 0
        stats_dic["stdev"] = 0
        stats_dic["min"] = 0
        stats_dic["max"] = 0
        stats_dic["zero_pos"] = 0
        stats_dic["total_pos"] = 0
        sc_list = []

    if report:
        print("bedtools makewindows from input BED ... ")

    # Make 1-pos bed from in_bed.
    makewin_cmd = "bedtools makewindows -b %s -w 1 -i srcwinnum > %s" % (in_bed, tmp_bed)
    os.system(makewin_cmd)

    if report:
        print("Extract conservation scores from bigWig ... ")

    # Extract conservation scores from .bw.
    bw_cmd = "bigWigAverageOverBed %s %s %s" %(con_bw, tmp_bed, tmp_tab)
    os.system(bw_cmd)

    # Region ID to score list.
    id2sc_dic = {}
    with open(tmp_tab) as f:
        for line in f:
            cols = line.strip().split("\t")
            pos_id = cols[0]
            sc = float(cols[3])
            if sc == 0:
                sc = 0
            m = re.search("(.+)_(\d+)", pos_id)
            reg_id = m.group(1)
            # pos = m.group(2)
            pol = id2pol_dic[reg_id]
            if not reg_id in id2sc_dic:
                id2sc_dic[reg_id] = [sc]
            else:
                # Reverse score list for "-".
                if pol == "+":
                    id2sc_dic[reg_id] = id2sc_dic[reg_id] + [sc]
                else:
                    id2sc_dic[reg_id] = [sc] + id2sc_dic[reg_id]
    f.closed

    if report:
        print("Write scores to output file ... ")

    # Write output file.
    OUTCON = open(out_con,"w")

    """
    For each ID output conservation scores.
    In case the ID has several _p or _e, assemble the parts and output full
    region scores list.

    """
    # Assemble split regions.
    if merge_split_regions:
        for core_id in label_dic:
            part_c = id2parts_dic[core_id]
            label = label_dic[core_id]
            merged_sc = []
            for i in range(part_c):
                i += 1
                # Get part region ID.
                region_id = core_id + "_%s%i" %(label, i)
                if region_id not in id2sc_dic:
                    assert False, "splite site ID \"%s\" missing in id2sc_dic" %(region_i)
                merged_sc += id2sc_dic[region_id]
                del id2sc_dic[region_id]
            assert merged_sc, "merged_sc list empty"
            id2sc_dic[core_id] = merged_sc

    # Output regions.
    for reg_id in id2sc_dic:
        OUTCON.write(">%s\n" %(reg_id))
        for sc in id2sc_dic[reg_id]:
            OUTCON.write("%s\n" %(sc))
        # Store conservation score stats.
        if stats_dic:
            if id2ucr_dic:
                # If id2ucr_dic, record values only for uppercase part of sequence.
                uc_s = id2ucr_dic[reg_id][0]
                uc_e = id2ucr_dic[reg_id][1]
                for i,sc in enumerate(id2sc_dic[reg_id]):
                    pos = i+1
                    if pos >= uc_s and pos <= uc_e:
                        if sc == 0:
                            stats_dic["zero_pos"] += 1
                        sc_list.append(sc)
            else:
                for sc in id2sc_dic[reg_id]:
                    if sc == 0:
                        stats_dic["zero_pos"] += 1
                    sc_list.append(sc)
    OUTCON.close()

    if stats_dic:
        assert sc_list, "no scores stored in score list"
        stats_dic["mean"] = statistics.mean(sc_list)
        stats_dic["stdev"] = statistics.stdev(sc_list)
        stats_dic["total_pos"] = len(sc_list)
        stats_dic["min"] = min(sc_list)
        stats_dic["max"] = max(sc_list)

    # Remove tmp files.
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)
    if os.path.exists(tmp_tab):
        os.remove(tmp_tab)


################################################################################

def bed_get_exon_intron_annotations_from_gtf(tr_ids_dic, in_bed,
                                             in_gtf, eia_out,
                                             stats_dic=None,
                                             id2ucr_dic=False,
                                             own_exon_bed=False,
                                             n_labels=False,
                                             intron_border_labels=False):

    """
    By default get exon (E) and intron (I) labels for each position in
    given in_bed BED file. Get labels from in_gtf GTF, for transcripts
    with IDs stored in tr_ids_dic. Output site labels to eia_out.

    Raiga: "Kriegsch n paar Eia??"

    tr_ids_dic:
        Transcript IDs for which to extract exon+intron regions for labelling.
    in_bed:
        BED file with regions to label.
    in_gtf:
        GTF file for extracting exon/intron regions.
    eia_out:
        Output file with labels.
    stats_dic:
        If not None, extract exon-intron annotation statistics and store
        in stats_dic.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        Set to define regions for which to extract exon-intron annotation
        stats, stored in stats_dic.
    own_exon_bed:
        Supply own exon BED file. This disables n_labels and
        intron_border_labels annotations. Also tr_ids_dic is not used anymore
        for defining transcript / exon regions.
    n_labels:
        If True, label all positions not covered by intron or exon regions
        with "N".
    intron_border_labels:
        If True, label intron 5' and 3' end positions (labels "T" and "F").


    >>> in_bed = "test_data/test_eia.bed"
    >>> in_gtf = "test_data/test_eia.gtf"
    >>> out_exp1_bed = "test_data/test_eia.exp1.eia"
    >>> out_exp2_bed = "test_data/test_eia.exp2.eia"
    >>> out_exp3_bed = "test_data/test_eia.exp3.eia"
    >>> out_exp4_bed = "test_data/test_eia.exp4.eia"
    >>> out_tmp1_bed = "test_data/test_eia.tmp1.eia"
    >>> out_tmp2_bed = "test_data/test_eia.tmp2.eia"
    >>> out_tmp3_bed = "test_data/test_eia.tmp3.eia"
    >>> out_tmp4_bed = "test_data/test_eia.tmp4.eia"
    >>> tr_ids_dic = {'tr1': 1, 'tr2': 1}
    >>> bed_get_exon_intron_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, out_tmp1_bed)
    >>> diff_two_files_identical(out_tmp1_bed, out_exp1_bed)
    True
    >>> bed_get_exon_intron_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, out_tmp2_bed, n_labels=True)
    >>> diff_two_files_identical(out_tmp2_bed, out_exp2_bed)
    True
    >>> bed_get_exon_intron_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, out_tmp3_bed, intron_border_labels=True)
    >>> diff_two_files_identical(out_tmp3_bed, out_exp3_bed)
    True
    >>> bed_get_exon_intron_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, out_tmp4_bed, n_labels=True, intron_border_labels=True)
    >>> diff_two_files_identical(out_tmp4_bed, out_exp4_bed)
    True

    """
    if own_exon_bed:
        intron_border_labels = False
        n_labels = False
        exon_bed = own_exon_bed
    else:
        # Checker.
        assert tr_ids_dic, "given dictionary tr_ids_dic empty"
        random_id = uuid.uuid1()
        exon_bed = str(random_id) + ".tmp.bed"

    intron_bed = False
    if intron_border_labels or n_labels:
        random_id = uuid.uuid1()
        intron_bed = str(random_id) + ".intron.tmp.bed"
    random_id = uuid.uuid1()
    border_bed = str(random_id) + ".border.tmp.bed"
    random_id = uuid.uuid1()
    merged_bed = str(random_id) + ".merged.tmp.bed"
    random_id = uuid.uuid1()
    tmp_out = str(random_id) + ".tmp.out"

    if stats_dic is not None:
        stats_dic["E"] = 0
        stats_dic["I"] = 0
        stats_dic["total_pos"] = 0
        if n_labels:
            stats_dic["N"] = 0
        if intron_border_labels:
            stats_dic["F"] = 0
            stats_dic["T"] = 0

    # Get exon (+ intron) regions from GTF.
    if not own_exon_bed:
        gtf_extract_exon_bed(in_gtf, exon_bed,
                            out_intron_bed=intron_bed,
                            use_ei_labels=True,
                            tr_ids_dic=tr_ids_dic)

    # Extract intron border positions to BED.
    if intron_border_labels:
        bed_extract_start_end_pos(intron_bed, border_bed)

    # Merge label region files for overlapping.
    merge_list = []
    merge_list.append(exon_bed)
    if intron_bed:
        merge_list.append(intron_bed)
    if intron_border_labels:
        merge_list.append(border_bed)
    merge_files(merge_list, merged_bed)

    # Get sstart + end for each site ID.
    id2s_dic = {}
    id2e_dic = {}
    # Dictionary of lists, store position labels, init with "I" or "N".
    id2labels_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            id2s_dic[site_id] = site_s
            id2e_dic[site_id] = site_e
            site_l = site_e - site_s
            assert site_l, "invalid site length for row \"%s\" in in_bed \"%s\"" %(row, in_bed)
            if n_labels:
                id2labels_dic[site_id] = ["N"]*site_l
            else:
                id2labels_dic[site_id] = ["I"]*site_l
    f.closed
    assert id2s_dic, "nothing got read in. Given BED file in_bed \"%s\" empty?" %(in_bed)


    # Preferred labels, i.e. do not overwrite these if present at position.
    pref_labels_dic = {}
    if intron_border_labels:
        pref_labels_dic["F"] = 1
        pref_labels_dic["T"] = 1

    # Run overlap calculation to get exon overlapping regions.
    intersect_params = "-s -wb"
    intersect_bed_files(in_bed, merged_bed, intersect_params, tmp_out)

    """
    Example output:
    $ intersectBed -a sites.bed -b annot.bed -s -wb
    chr1	1000	1020	site1	0	+	chr1	980	1020	F	0	+
    chr1	1020	1023	site1	0	+	chr1	1020	1023	S	0	+
    chr1	1020	1050	site1	0	+	chr1	1020	1500	C	0	+
    """

    with open(tmp_out) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            s = int(cols[1]) + 1 # Make one-based.
            e = int(cols[2])
            site_id = cols[3]
            site_s = id2s_dic[site_id] + 1 # Make one-based.
            site_e = id2e_dic[site_id]
            site_pol = cols[5]
            label = cols[9]
            # + case.
            if site_pol == "+":
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = i - site_s
                        if id2labels_dic[site_id][li] not in pref_labels_dic:
                            id2labels_dic[site_id][li] = label
            else:
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = site_e - i
                        if id2labels_dic[site_id][li] not in pref_labels_dic:
                            id2labels_dic[site_id][li] = label
    f.closed

    # Output transcript region annotations to .eia file.
    OUTEIA = open(eia_out,"w")
    for site_id in id2labels_dic:
        # List to string.
        label_str = "".join(id2labels_dic[site_id])
        OUTEIA.write("%s\t%s\n" %(site_id, label_str))
        # Get label statistics.
        if stats_dic:
            if id2ucr_dic:
                # If uppercase part only, prune label_str.
                uc_s = id2ucr_dic[site_id][0]
                uc_e = id2ucr_dic[site_id][1]
                label_str = label_str[uc_s-1:uc_e]
            stats_dic["total_pos"] += len(label_str)
            occ_labels = ["F", "T"]
            for ocl in occ_labels:
                if re.search("%s" %(ocl), label_str):
                    stats_dic[ocl] += 1
            for l in label_str:
                if l not in occ_labels:
                    stats_dic[l] += 1
    OUTEIA.close()

    # Remove tmp files.
    if os.path.exists(exon_bed):
        if not own_exon_bed:
            os.remove(exon_bed)
    if intron_bed:
        if os.path.exists(intron_bed):
            os.remove(intron_bed)
    if os.path.exists(border_bed):
        os.remove(border_bed)
    if os.path.exists(merged_bed):
        os.remove(merged_bed)
    if os.path.exists(tmp_out):
        os.remove(tmp_out)


################################################################################

def extract_exon_intron_labels(in_bed, exon_bed, out_labels):
    """
    Overlap genomic regions .bed with exon regions .bed, and mark region
    positions with "I" (intron) or "E" (exon) labels based on the overlap
    with exon regions.

    >>> region_bed = "test_data/test3.bed"
    >>> exon_bed = "test_data/test4.bed"
    >>> exp_lab = "test_data/test.exon_intron_labels"
    >>> out_lab = "test_data/test.tmp.exon_intron_labels"
    >>> extract_exon_intron_labels(region_bed, exon_bed, out_lab)
    >>> diff_two_files_identical(out_lab, exp_lab)
    True

    """

    # Check.
    assert is_tool("bedtools"), "bedtools not in PATH"
    assert os.path.isfile(in_bed), "cannot open in_bed BED file \"%s\"" % (in_bed)
    assert os.path.isfile(exon_bed), "cannot open exon_bed BED file \"%s\"" % (exon_bed)
    assert bed_check_unique_ids(in_bed), "in_bed \"%s\" column 4 IDs not unique" % (in_bed)

    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"

    # Get polarity, start, end for each site ID.
    id2pol_dic = {}
    id2s_dic = {}
    id2e_dic = {}
    # Dictionary of lists, store position labels, init with "I".
    id2labels_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_pol = cols[5]
            id2pol_dic[site_id] = site_pol
            id2s_dic[site_id] = site_s
            id2e_dic[site_id] = site_e
            site_l = site_e - site_s
            assert site_l, "invalid site length for row \"%s\" in in_bed \"%s\"" %(row, in_bed)
            id2labels_dic[site_id] = ["I"]*site_l
    f.closed
    assert id2pol_dic, "No entries read into dictionary (input file \"%s\" empty or malformatted?)" % (in_bed)

    # Run overlap calculation to get exon overlapping regions.
    intersect_params = "-s"
    intersect_bed_files(in_bed, exon_bed, intersect_params, tmp_bed)
    with open(tmp_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            s = int(cols[1]) + 1 # Make one-based.
            e = int(cols[2])
            site_id = cols[3]
            site_s = id2s_dic[site_id] + 1 # Make one-based.
            site_e = id2e_dic[site_id]
            site_pol = id2pol_dic[site_id]
            # + case.
            if site_pol == "+":
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = i - site_s
                        id2labels_dic[site_id][li] = "E"
            else:
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = site_e - i
                        id2labels_dic[site_id][li] = "E"
    f.closed

    # Write labels to file.
    OUTLAB = open(out_labels,"w")
    for site_id in id2labels_dic:
        # List to string.
        label_str = "".join(id2labels_dic[site_id])
        OUTLAB.write("%s\t%s\n" %(site_id, label_str))
    OUTLAB.close()

    # Remove tmp files.
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)


################################################################################

def bed_extract_start_end_pos(in_bed, out_bed):
    """
    Extract region start and end positions from given in_bed BED.
    Output start and end position regions to out_bed BED.
    Output column 4 IDs will be F for 5' end position
    (strand info considered) and T for 3'end position.

    >>> in_bed = "test_data/test_start_end.bed"
    >>> out_exp_bed = "test_data/test_start_end.exp.bed"
    >>> out_tmp_bed = "test_data/test_start_end.tmp.bed"
    >>> bed_extract_start_end_pos(in_bed, out_tmp_bed)
    >>> diff_two_files_identical(out_exp_bed, out_tmp_bed)
    True

    """
    OUTPOS = open(out_bed,"w")
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            s = int(cols[1])
            e = int(cols[2])
            pol = cols[5]
            s_f = s
            e_f = s + 1
            s_t = e - 1
            e_t = e
            if pol == "-":
                s_f = e - 1
                e_f = e
                s_t = s
                e_t = s + 1
            # Output border positions.
            OUTPOS.write("%s\t%i\t%i\tF\t0\t%s\n" %(chr_id, s_f, e_f, pol))
            OUTPOS.write("%s\t%i\t%i\tT\t0\t%s\n" %(chr_id, s_t, e_t, pol))
    f.closed
    OUTPOS.close()


################################################################################

def intersect_bed_files(a_file, b_file, params, out_file,
                        sorted_out=False):
    """
    Intersect two .bed files, using intersectBed.

    """

    check_cmd = "intersectBed -a " + a_file + " -b " + b_file + " " + params + " > " + out_file
    if sorted_out:
        check_cmd = "intersectBed -a " + a_file + " -b " + b_file + " " + params + " | " + "sort -k1,1 -k2,2n > " + out_file
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "intersectBed has problems with your input:\n%s\n%s" %(check_cmd, output)


################################################################################

def bed_intersect_count_region_overlaps(a_file, b_file,
                                        b_f=0.75):
    """
    Intersect two .bed files, count how often -a regions overlap with -b
    regions. Return count dictionary (-a col4 ID -> overlap count)

    intersectBed -a genes.bed -b sites.bed -s -F 0.75
    chr1	1000	1050	ENSG1	0	+
    chr1	1500	1550	ENSG1	0	+

    >>> a_file = "test_data/test_intersect.genes.bed"
    >>> b_file = "test_data/test_intersect.sites.bed"
    >>> bed_intersect_count_region_overlaps(a_file, b_file)
    {'ENSG1': 2, 'ENSG2': 1}

    """

    count_dic = {}
    params = "-s -F %.2f" %(b_f)

    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_out = str(random_id) + ".intersect.tmp.out"

    check_cmd = "intersectBed -a " + a_file + " -b " + b_file + " " + params + " > " + tmp_out
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "intersectBed has problems with your input:\n%s\n%s" %(check_cmd, output)

    # Acquire information.
    with open(tmp_out) as f:
        for line in f:
            cols = line.strip().split("\t")
            reg_id = cols[3]
            if reg_id in count_dic:
                count_dic[reg_id] += 1
            else:
                count_dic[reg_id] = 1
    f.close()
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    assert count_dic, "no region counts read in. Possibly no overlaps?"
    return count_dic


################################################################################

def diff_two_files_identical(file1, file2):
    """
    Check whether two files are identical. Return true if diff reports no
    differences.

    >>> file1 = "test_data/file1"
    >>> file2 = "test_data/file2"
    >>> diff_two_files_identical(file1, file2)
    True
    >>> file1 = "test_data/test1.bed"
    >>> diff_two_files_identical(file1, file2)
    False

    """
    same = True
    check_cmd = "diff " + file1 + " " + file2
    output = subprocess.getoutput(check_cmd)
    if output:
        same = False
    return same


################################################################################

def fasta_read_in_ids(fasta_file,
                      return_type="list"):
    """
    Given a .fa file, read in header IDs in order appearing in file,
    and store in list.
    Also works with other files containing headers like ">id"

    >>> test_file = "test_data/test.fa"
    >>> fasta_read_in_ids(test_file)
    ['seq1', 'seq2']
    >>> fasta_read_in_ids(test_file, return_type="dictionary")
    {'seq1': 1, 'seq2': 1}

    """
    if return_type == "list":
        ids = []
    elif return_type == "dictionary":
        ids = {}
    else:
        assert False, "invalid return_type set"
    with open(fasta_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                if return_type == "list":
                    ids.append(seq_id)
                else:
                    ids[seq_id] = 1
    f.close()
    return ids


################################################################################

def filter_bed_row_dic_output(id2row_dic, keep_ids_dic, out_bed):
    """
    Filter bed rows dictionary, with key = site ID and value = bed row string.

    id2row_dic    Dictionary to filter with site ID -> bed row string
    keep_ids_dic  Dictionary storing site IDs to keep
    out_bed       Output .bed file

    """
    # Check.
    assert keep_ids_dic, "given keep_ids_dic empty"
    # Write labels to file.
    OUTBED = open(out_bed,"w")
    c_out = 0
    for site_id in id2row_dic:
        if site_id in keep_ids_dic:
            c_out += 1
            OUTBED.write("%s\n" %(id2row_dic[site_id]))
    OUTBED.close()
    assert c_out, "no remaining BED rows after filtering"


################################################################################

def bed_get_region_ids(bed_file):
    """
    Read in .bed file, return region/site IDs (column 5 IDs).

    >>> test_file = "test_data/test3.bed"
    >>> bed_get_region_ids(test_file)
    {'CLIP1': 1, 'CLIP2': 1}

    """
    ids_dic = {}
    with open(bed_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_id = cols[3]
            assert site_id not in ids_dic, "column 4 IDs not unique in given .bed file \"%s\"" %(bed_file)
            ids_dic[site_id] = 1
    f.closed
    assert ids_dic, "No IDs read into dictionary (input file \"%s\" empty or malformatted?)" % (bed_file)
    return ids_dic


################################################################################

def fasta_filter_entries(in_fasta, keep_ids_dic, out_fasta,
                         dna=False):
    """
    Filter FASTA file, keeping entries with IDs in keep_ids_dic.

    >>> in_fasta = "test_data/test.fa"
    >>> out_fasta = "test_data/test.tmp.fa"
    >>> exp_fasta = "test_data/test3.fa"
    >>> keep_ids_dic = {'seq2'}
    >>> fasta_filter_entries(in_fasta, keep_ids_dic, out_fasta, dna=True)
    >>> diff_two_files_identical(out_fasta, exp_fasta)
    True

    """
    # Check.
    assert keep_ids_dic, "given keep_ids_dic empty"
    # Read in FASTA file.
    fasta_dic = read_fasta_into_dic(in_fasta, ids_dic=keep_ids_dic, dna=dna)
    assert fasta_dic, "fasta_dic empty after filtering"
    # Write FASTA file.
    OUTFA = open(out_fasta,"w")
    for fa_id in fasta_dic:
        OUTFA.write(">%s\n%s\n" %(fa_id, fasta_dic[fa_id]))
    OUTFA.close()


################################################################################

def make_file_copy(in_file, out_file,
                   delete_in=False):
    """
    Make a file copy by copying in_file to out_file.

    """
    check_cmd = "cat " + in_file + " > " + out_file
    assert in_file != out_file, "cat does not like to cat file into same file (%s)" %(check_cmd)
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "cat did not like your input (in_file: %s, out_file: %s):\n%s" %(in_file, out_file, output)
    # Delete in_file.
    if delete_in:
        if os.path.exists(in_file):
            os.remove(in_file)


################################################################################

def move_rename_file(in_file, out_file):
    """
    Move / rename in_file to out_file.

    """
    check_cmd = "mv " + in_file + " " + out_file
    assert in_file != out_file, "mv does not like to mv file into same file (%s)" %(check_cmd)
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "mv did not like your input (in_file: %s, out_file: %s):\n%s" %(in_file, out_file, output)


################################################################################

def con_merge_exon_regions(in_con, id2exonc_dic, out_con,
                           id2len_dic=False):
    """
    Take a conservation scores (.con) file, and merge exon regions
    identified by id2exonc_dic. Output merged regions to out_con.
    id2exonc_dic format: site_id (key) -> exon_count (value)
    Only counts > 1 need to be merged, site_id counts == 1 are output
    unchanged.

    .con file format:
    >site_id1
    1	0	-0.101
    2	0	-0.303
    3	0	0.909
    ....
    >site_id2

    For exon regions we expect ID : site_id_e1, site_ide_e2 ...
    If id2len_dic given, compare the (concatenated) list lengths
    with the lengths in id2len_dic for sanity checking.

    >>> in_con = "test_data/test2.con"
    >>> out_con = "test_data/test2.tmp.con"
    >>> exp_con = "test_data/test2.exp.con"
    >>> id2exonc_dic = {'CLIP1': 1, 'CLIP2': 3}
    >>> id2len_dic = {'CLIP1': 10, 'CLIP2': 15}
    >>> con_merge_exon_regions(in_con, id2exonc_dic, out_con, id2len_dic=id2len_dic)
    >>> diff_two_files_identical(out_con, exp_con)
    True

    """

    # Check.
    assert id2exonc_dic, "given dictionary keep_ids_dic empty"
    assert os.path.isfile(in_con), "cannot open in_con \"%s\"" % (in_con)

    # Read in conservation scores (for each position a list of [val1, val2] ).
    con_dic = {}
    seq_id = ""
    # Go through .con file, extract phastCons, phyloP scores for each position.
    with open(in_con) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                if seq_id not in con_dic:
                    con_dic[seq_id] = []
                else:
                    assert False, "non-unique ID \"%s\" in in_con \"%s\"" %(seq_id, in_con)
            else:
                row = line.strip()
                cols = line.strip().split("\t")
                con_dic[seq_id].append([cols[1], cols[2]])
    f.closed
    assert con_dic, "con_dic empty"

    # Merge entries and write to out_con.
    OUTCON = open(out_con,"w")
    for site_id in id2exonc_dic:
        ex_c = id2exonc_dic[site_id]
        if ex_c > 1:
            # Go over exons.
            new_list = []
            for i in range(ex_c):
                i += 1
                exon_id = site_id + "_e%i" %(i)
                if exon_id in con_dic:
                    new_list += con_dic[exon_id]
                else:
                    assert False, "exon_id \"%\" missing in con_dic" %(site_id)
            if id2len_dic:
                if site_id in id2len_dic:
                    ll = len(new_list)
                    sl = id2len_dic[site_id]
                    assert sl == ll, "lengths not identical for site_id \"%s\" (%i != %i)" %(site_id, sl, ll)
                else:
                    assert False, "missing site_id \"%s\" in id2len_dic" %(site_id)
            OUTCON.write(">%s\n" %(site_id))
            pos = 0
            for l in new_list:
                pos += 1
                OUTCON.write("%i\t%s\t%s\n" %(pos, l[0], l[1]))
        elif ex_c == 1:
            # For single exon / regions.
            if site_id in con_dic:
                if id2len_dic:
                    if site_id in id2len_dic:
                        ll = len(con_dic[site_id])
                        sl = id2len_dic[site_id]
                        assert sl == ll, "lengths not identical for site_id \"%s\" (%i != %i)" %(site_id, sl, ll)
                    else:
                        assert False, "missing site_id \"%s\" in id2len_dic" %(site_id)
                OUTCON.write(">%s\n" %(site_id))
                pos = 0
                for l in con_dic[site_id]:
                    pos += 1
                    OUTCON.write("%i\t%s\t%s\n" %(pos, l[0], l[1]))
            else:
                assert False, "site_id \"%\" has ex_c = 1 but is missing in con_dic" %(site_id)
        else:
            assert False, "invalid ex_c (%i) given for site_id \"%s\"" %(ex_c, site_id)
    OUTCON.close()


################################################################################

def fasta_merge_exon_regions(in_fa, id2exonc_dic, out_fa,
                             id2len_dic=False):
    """
    Take a FASTA file, and merge exon regions identified by
    id2exonc_dic. Output merged regions to out_fa.
    id2exonc_dic format: site_id (key) -> exon_count (value)
    Only counts > 1 need to be merged, site_id counts == 1 are output
    unchanged.

    For exon regions we expect ID : site_id_e1, site_ide_e2 ...
    If id2len_dic given, compare the (concatenated) sequence lengths
    with the lengths in id2len_dic for sanity checking.

    >>> in_fa = "test_data/test4.fa"
    >>> out_fa = "test_data/test4.tmp.fa"
    >>> exp_fa = "test_data/test4.exp.fa"
    >>> id2exonc_dic = {'CLIP1': 1, 'CLIP2': 3}
    >>> id2len_dic = {'CLIP1': 10, 'CLIP2': 15}
    >>> fasta_merge_exon_regions(in_fa, id2exonc_dic, out_fa, id2len_dic=id2len_dic)
    >>> diff_two_files_identical(out_fa, exp_fa)
    True

    """

    # Check.
    assert id2exonc_dic, "given keep_ids_dic empty"
    assert os.path.isfile(in_fa), "cannot open in_fa \"%s\"" % (in_fa)

    # Read in FASTA file.
    fasta_dic = read_fasta_into_dic(in_fa)
    assert fasta_dic, "fasta_dic empty"

    # Merge sequences and write to out_fa.
    OUTFA = open(out_fa,"w")
    for site_id in id2exonc_dic:
        ex_c = id2exonc_dic[site_id]
        if ex_c > 1:
            # Go over exons.
            new_seq = ""
            for i in range(ex_c):
                i += 1
                exon_id = site_id + "_e%i" %(i)
                if exon_id in fasta_dic:
                    new_seq += fasta_dic[exon_id]
                else:
                    assert False, "exon_id \"%\" missing in fasta_dic (possibly sequence extraction from .2bit failed for this region)" %(site_id)
            if id2len_dic:
                if site_id in id2len_dic:
                    ll = len(new_seq)
                    sl = id2len_dic[site_id]
                    assert sl == ll, "lengths not identical for site_id \"%s\" (%i != %i)" %(site_id, sl, ll)
                else:
                    assert False, "missing site_id \"%s\" in id2len_dic" %(site_id)
            OUTFA.write(">%s\n%s\n" %(site_id, new_seq))
        elif ex_c == 1:
            # For single exon / regions.
            if site_id in fasta_dic:
                if id2len_dic:
                    if site_id in id2len_dic:
                        ll = len(fasta_dic[site_id])
                        sl = id2len_dic[site_id]
                        assert sl == ll, "lengths not identical for site_id \"%s\" (%i != %i)" %(site_id, sl, ll)
                    else:
                        assert False, "missing site_id \"%s\" in id2len_dic" %(site_id)
                OUTFA.write(">%s\n%s\n" %(site_id, fasta_dic[site_id]))
            else:
                assert False, "site_id \"%\" has ex_c = 1 but is missing in fasta_dic (possibly sequence extraction from .2bit failed for this region)" %(site_id)
        else:
            assert False, "invalid ex_c (%i) given for site_id \"%s\"" %(ex_c, site_id)
    OUTFA.close()


################################################################################

def exon_intron_labels_merge_exon_regions(in_file, id2exonc_dic, out_file,
                                          id2len_dic=False):
    """
    Take a .exon_intron_labels file, and merge exon regions identified by
    id2exonc_dic. Output merged regions to out_file.
    id2exonc_dic format: site_id (key) -> exon_count (value)
    Only counts > 1 need to be merged, site_id counts == 1 are output
    unchanged.

    .exon_intron_labels file format:
    CLIP1	IIIIIIEEEE
    CLIP2	EEEEEIIIIIIIIII
    ....

    For exon regions we expect ID : site_id_e1, site_ide_e2 ...
    If id2len_dic given, compare the (concatenated) label string lengths
    with the lengths in id2len_dic for sanity checking.

    >>> in_file = "test_data/test2.exon_intron_labels"
    >>> out_file = "test_data/test2.tmp.exon_intron_labels"
    >>> exp_file = "test_data/test2.exp.exon_intron_labels"
    >>> id2exonc_dic = {'CLIP1': 1, 'CLIP2': 3}
    >>> id2len_dic = {'CLIP1': 10, 'CLIP2': 15}
    >>> exon_intron_labels_merge_exon_regions(in_file, id2exonc_dic, out_file, id2len_dic=id2len_dic)
    >>> diff_two_files_identical(out_file, exp_file)
    True

    """
    # Check.
    assert id2exonc_dic, "given keep_ids_dic empty"
    assert os.path.isfile(in_file), "ERROR: Cannot open in_file \"%s\"" % (in_file)
    # Read in .exon_intron_labels file into dictionary.
    id2labels_dic = {}
    with open(in_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_id = cols[0]
            labels = cols[1]
            assert site_id not in id2labels_dic, "column 1 IDs not unique in given .exon_intron_labels file \"%s\"" %(in_file)
            id2labels_dic[site_id] = labels
    f.closed
    OUTLABELS = open(out_file,"w")
    for site_id in id2exonc_dic:
        ex_c = id2exonc_dic[site_id]
        if ex_c > 1:
            # Go over exons.
            new_labels = ""
            for i in range(ex_c):
                i += 1
                exon_id = site_id + "_e%i" %(i)
                if exon_id in id2labels_dic:
                    new_labels += id2labels_dic[exon_id]
                else:
                    assert False, "exon_id \"%\" missing in id2labels_dic" %(site_id)
            if id2len_dic:
                if site_id in id2len_dic:
                    ll = len(new_labels)
                    sl = id2len_dic[site_id]
                    assert sl == ll, "lengths not identical for site_id \"%s\" (%i != %i)" %(site_id, sl, ll)
                else:
                    assert False, "missing site_id \"%s\" in id2len_dic" %(site_id)
            OUTLABELS.write("%s\t%s\n" %(site_id, new_labels))
        elif ex_c == 1:
            # For single exon / regions.
            if site_id in id2labels_dic:
                if id2len_dic:
                    if site_id in id2len_dic:
                        ll = len(id2labels_dic[site_id])
                        sl = id2len_dic[site_id]
                        assert sl == ll, "lengths not identical for site_id \"%s\" (%i != %i)" %(site_id, sl, ll)
                    else:
                        assert False, "missing site_id \"%s\" in id2len_dic" %(site_id)
                OUTLABELS.write("%s\t%s\n" %(site_id, id2labels_dic[site_id]))
            else:
                assert False, "site_id \"%\" has ex_c = 1 but is missing in id2labels_dic" %(site_id)
        else:
            assert False, "invalid ex_c (%i) given for site_id \"%s\"" %(ex_c, site_id)
    OUTLABELS.close()


################################################################################

def exon_intron_labels_read_in_ids(in_file):
    """
    Given a .exon_intron_labels file, read in header IDs in dictionary.

    >>> test_file = "test_data/test.exon_intron_labels"
    >>> exon_intron_labels_read_in_ids(test_file)
    {'CLIP1': 1, 'CLIP2': 1}

    """
    # Check.
    assert os.path.isfile(in_file), "cannot open in_file \"%s\"" % (in_file)
    # Read in IDs.
    ids_dic = {}
    with open(in_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_id = cols[0]
            assert site_id not in ids_dic, "column 1 IDs not unique in given .exon_intron_labels file \"%s\"" %(in_file)
            ids_dic[site_id] = 1
    f.closed
    return ids_dic


################################################################################

def get_chromosome_lengths_from_2bit(in_2bit, out_lengths,
                                     std_chr_filter=False):
    """
    Get chromosome lengths from in_2bit .2bit file. Write lengths
    to out_lengths, with format:
    chr1	248956422
    chr10	133797422
    chr11	135086622
    ...
    Also return a dictionary with key=chr_id and value=chr_length.

    std_chr_filter:
        Filter / convert chromosome IDs with function check_convert_chr_id(),
        removing non-standard chromosomes, and convert IDs like 1,2,X,MT ..
        to chr1, chr2, chrX, chrM.

    """

    # Check for twoBitInfo.
    assert is_tool("twoBitInfo"), "twoBitInfo not in PATH"

    # Run twoBitInfo and check.
    check_cmd = "twoBitInfo " + in_2bit + " " + out_lengths
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "twoBitInfo is complaining:\n%s\n%s" %(check_cmd, output)

    # Read in lengths into dictionary.
    chr_len_dic = {}
    with open(out_lengths) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            chr_l = int(cols[1])
            # Check ID.
            if std_chr_filter:
                new_chr_id = check_convert_chr_id(chr_id)
                # If not standard chromosome ID or conversion failed, skip.
                if not new_chr_id:
                    continue
                else:
                    chr_id = new_chr_id
            assert chr_id not in chr_len_dic, "non-unique chromosome ID \"%s\" encountered in \"%s\"" %(chr_id, out_lengths)
            chr_len_dic[chr_id] = chr_l
    f.closed
    assert chr_len_dic, "chr_len_dic empty (\"%s\" empty? Chromosome IDs filter activated?)" %(out_lengths)

    return chr_len_dic


################################################################################

def get_center_position(start, end):
    """
    Get center position (1-based), given a (genomic) start (0-based) and
    end coordinate (1-based).

    >>> get_center_position(10, 11)
    11
    >>> get_center_position(1000,2000)
    1501
    >>> get_center_position(11, 20)
    17

    """
    # If region has length of 1, return end position.
    center_pos = end
    # Otherwise calculate new center position.
    if not end - start == 1:
        center_pos = round( ( (end - start) / 2 ) + start ) + 1
    return center_pos


################################################################################

def bed_merge_file(in_bed, out_bed,
                   custom_params_str=False):
    """
    Use mergeBed from bedtools to merge overlapping .bed entries, storing
    the region IDs to later pick one region for each set of overlapping
    regions.

    >>> in_bed = "test_data/test.sorted.bed"
    >>> out_bed = "test_data/test.sorted.merged.tmp.bed"
    >>> out_exp_bed = "test_data/test.sorted.merged.exp.bed"
    >>> bed_merge_file(in_bed, out_bed)
    >>> diff_two_files_identical(out_bed, out_exp_bed)
    True

    """
    # Check for bedtools.
    assert is_tool("bedtools"), "bedtools not in PATH"
    # Parameter string.
    params_str = '-s -c 4 -o distinct -delim ";"'
    if custom_params_str:
        params_str = custom_params_str
    check_cmd = "mergeBed -i " + in_bed + " " + params_str + " > " + out_bed
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "mergeBed is complaining:\n%s\n%s" %(check_cmd, output)


################################################################################

def bed_sort_file(in_bed, out_bed,
                  custom_params_str=False):
    """
    Use command line sort to sort the in_bed .bed file. Output sorted .bed
    file to out_bed.

    """
    # Parameter string.
    params_str = '-k1,1 -k2,2n'
    if custom_params_str:
        params_str = custom_params_str
    check_cmd = "sort " + params_str + " " + in_bed + " > " + out_bed
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "sort is complaining:\n%s\n%s" %(check_cmd, output)


################################################################################

def bed_sort_merge_output_top_entries(in_bed, out_bed,
                                      rev_filter=False):
    """
    Sort in_bed file, use mergeBed from bedtools to merge overlapping entries,
    then select for each overlapping set the entry with highest score and
    output it to out_bed.

    >>> in_bed = "test_data/test5.bed"
    >>> out_bed = "test_data/test5.tmp.bed"
    >>> exp_bed = "test_data/test5.exp.bed"
    >>> bed_sort_merge_output_top_entries(in_bed, out_bed)
    >>> diff_two_files_identical(out_bed, exp_bed)
    True

    """
    assert os.path.isfile(in_bed), "cannot open in_bed \"%s\"" % (in_bed)
    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"
    # Read in_bed rows into dictionary.
    id2row_dic = bed_read_rows_into_dic(in_bed)
    # Get region scores.
    id2sc_dic = bed_get_region_id_scores(in_bed)
    # Sort file.
    bed_sort_file(in_bed, out_bed)
    # Merge .bed.
    bed_merge_file(out_bed, tmp_bed)
    # Output file.
    OUTBED = open(out_bed,"w")
    # Open merged .bed file, and select top entry for each overlap set.
    with open(tmp_bed) as f:
        for line in f:
            cols = line.strip().split("\t")
            ids = cols[3].split(";")
            best_id = "-"
            best_sc = -666666
            if rev_filter:
                best_sc = 666666
            for site_id in ids:
                assert site_id in id2sc_dic, "site ID \"%s\" not found in id2sc_dic" % (site_id)
                site_sc = id2sc_dic[site_id]
                if rev_filter:
                    if site_sc < best_sc:
                        best_sc = site_sc
                        best_id = site_id
                else:
                    if site_sc > best_sc:
                        best_sc = site_sc
                        best_id = site_id
            assert best_id in id2row_dic, "site ID \"%s\" not found in id2row_dic" % (best_id)
            OUTBED.write(id2row_dic[best_id] + "\n")
    f.closed
    OUTBED.close()
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)


################################################################################

def bed_get_score_to_count_dic(in_bed):
    """
    Given an .bed file in_bed, store scores and count how many times each
    score appears. Return dictionary with score -> count mapping.

    >>> in_bed = "test_data/test1.bed"
    >>> bed_get_score_to_count_dic(in_bed)
    {'1': 2, '0': 2, '2': 1, '3': 2}

    """
    assert os.path.isfile(in_bed), "cannot open in_bed \"%s\"" % (in_bed)
    # Read in IDs.
    sc2c_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_sc = cols[4]
            if site_sc in sc2c_dic:
                sc2c_dic[site_sc] += 1
            else:
                sc2c_dic[site_sc] = 1
    f.closed
    return sc2c_dic


################################################################################

def bed_get_region_id_scores(in_bed, no_float=False):
    """
    Read in .bed file, and store scores for each region in dictionary
    (unique column 4 ID and column 5 score have to be present).
    Return dictionary with mappings region ID -> region score

    >>> test_bed = "test_data/test5.bed"
    >>> bed_get_region_id_scores(test_bed)
    {'CLIP2': 2.57, 'CLIP1': 1.58, 'CLIP3': 3.11}

    """
    id2sc_dic = {}
    # Open input .bed file.
    with open(in_bed) as f:
        for line in f:
            cols = line.strip().split("\t")
            site_id = cols[3]
            site_sc = float(cols[4])
            if no_float:
                site_sc = cols[4]
            id2sc_dic[site_id] = site_sc
    f.closed
    assert id2sc_dic, "nothing read in for in_bed \"%s\"" %(in_bed)
    return id2sc_dic


################################################################################

def bed_process_bed_file(in_bed_file, out_bed_file,
                         new2oldid_dic=None,
                         score_thr=None,
                         min_len=False,
                         max_len=False,
                         generate_unique_ids=False,
                         center_sites=False,
                         ext_lr=False,
                         seq_len_dic=False,
                         siteids2keep_dic=False,
                         seqids2keep_dic=False,
                         siteseqids2keep_dic=False,
                         zero_scores=False,
                         int_whole_nr=True,
                         rev_filter=False,
                         id_prefix="CLIP"):
    """
    Process .bed file in various ways:
    - Filter by region length (min_len, max_len) or region score (column 5)
      (score_thr). By default no score or length filtering is applied.
    - Option to reverse-filter scores (the lower score the better)
    - Center regions (center_sites=True)
    - Extend sites up- downstream (ext_lr=value)
    - Generate new region IDs (column 4, generate_unique_ids=True),
      optionally providing an id_prefix
    - Filter by given dictionary of region IDs (ids2keep_dic)
    - Print "0" scores to column 5 (zero_scores)

    Output processed .bed file (in_bed_file) to new bed file (out_bed_file)

    >>> in_bed = "test_data/test1.bed"
    >>> out_bed = "test_data/out.tmp.bed"
    >>> bed_process_bed_file(in_bed, out_bed, score_thr=1)
    >>> count_file_rows(out_bed)
    5
    >>> bed_process_bed_file(in_bed, out_bed, rev_filter=True, score_thr=1)
    >>> count_file_rows(out_bed)
    4
    >>> in_bed = "test_data/test5.bed"
    >>> out_bed = "test_data/out.tmp.bed"
    >>> out_bed_exp = "test_data/test5.centered_zero_sc.bed"
    >>> bed_process_bed_file(in_bed, out_bed, zero_scores=True, center_sites=True)
    >>> diff_two_files_identical(out_bed, out_bed_exp)
    True

    """

    # Output .bed file.
    OUTBED = open(out_bed_file,"w")

    # New site IDs.
    site_id_pref = id_prefix
    c_sites = 0

    # Open input .bed file.
    with open(in_bed_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            seq_id = cols[0]
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_sc = float(cols[4])
            site_pol = cols[5]
            site_l = site_e - site_s
            # Sanity checking .bed file.
            assert site_s < site_e, "invalid region coordinates in .bed file \"%s\" (start >= end: %i >= %i)" % (in_bed_file, site_s, site_e)
            assert site_s >= 0 and site_e >= 1, "invalid region coordinates in .bed file \"%s\" (start < 0 or end < 1)" % (in_bed_file)
            # Filter by IDs to keep dictionary.
            if siteids2keep_dic:
                if not site_id in siteids2keep_dic:
                    continue
            if seqids2keep_dic:
                if not seq_id in seqids2keep_dic:
                    continue
            if siteseqids2keep_dic:
                if site_id in siteseqids2keep_dic:
                    if not seq_id == siteseqids2keep_dic[site_id]:
                        continue
            # Filter by score.
            if score_thr is not None:
                if rev_filter:
                    if site_sc > score_thr:
                        continue
                else:
                    if site_sc < score_thr:
                        continue
            # Check whether score is whole number.
            if int_whole_nr:
                if not site_sc % 1:
                    site_sc = int(site_sc)
            # Filter by minimum site length.
            if min_len:
                if min_len > site_l:
                    continue
            # Filter by maximum site length.
            if max_len:
                if max_len < site_l:
                    continue
            # Update start + end positions.
            new_s = site_s
            new_e = site_e
            # Center sites (get center position).
            if center_sites:
                new_e = get_center_position(site_s, site_e)
                new_s = new_e - 1
            # If site extension is specified.
            if ext_lr:
                new_s = new_s - ext_lr
                new_e = new_e + ext_lr
                if new_s < 0:
                    new_s = 0
            # New site ID.
            if generate_unique_ids:
                c_sites += 1
                new_site_id = "%s_%i" % (site_id_pref, c_sites)
                # If new ID to old ID mapping should be generated.
                if new2oldid_dic is not None:
                    new2oldid_dic[new_site_id] = site_id
                    site_id = new_site_id
            new_sc = str(site_sc)
            if zero_scores:
                new_sc = "0"
            if seq_len_dic:
                assert seq_id in seq_len_dic, "sequence ID \"%s\" missing in given sequence lengths dictionary" %(seq_id)
                if new_e > seq_len_dic[seq_id]:
                    new_e = seq_len_dic[seq_id]
            # Output to new file.
            OUTBED.write("%s\t%i\t%i\t%s\t%s\t%s\n" % (seq_id,new_s,new_e,site_id,new_sc,site_pol))
    f.closed
    OUTBED.close()


################################################################################

def add_bed_output_ids_to_dic(output, ids_dic):
    """
    Add column 4 IDs from a BED output on command line (e.g. from
    cat test.bed) to ids_dic.

    >>> check_cmd = "cat test_data/test3.bed"
    >>> output = subprocess.getoutput(check_cmd)
    >>> ids_dic = {}
    >>> add_bed_output_ids_to_dic(output, ids_dic)
    >>> ids_dic
    {'CLIP1': 1, 'CLIP2': 1}

    """
    assert output, "no output given"
    ol = output.strip().split("\n")
    for r in ol:
        m = re.search('.+?\t\d+\t\d+\t(.+?)\t', r)
        ids_dic[m.group(1)] = 1


################################################################################

def check_random_negatives(in_bed, incl_bed, excl_bed, chr_lengths_file,
                           trouble_ids_dic=None,
                           report=False):
    """
    Check whether bedtools shuffle works as described.
    I.e., we demand the random negatives to fully overlap with incl_bed and with
    chromosomes / reference regions (chr_lengths_file). For regions where this is
    not the case, print a warning + output the regions.

    """
    # Return warning.
    warning = False

    # tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"

    # Read in chromosome lengths.
    chr_len_dic = {}
    with open(chr_lengths_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            chr_id = cols[0]
            chr_l = int(cols[1])
            chr_len_dic[chr_id] = chr_l
    f.closed
    assert chr_len_dic, "no chromsome lengths read in from chr_lengths_file"

    # Generate BED from chromosome lengths.
    bed_sequence_lengths_to_bed(chr_len_dic, tmp_bed)

    # Get sites that overlap partially or not at all.
    params = "-f 1 -v"
    # Check reference regions.
    check_cmd = "intersectBed -a " + in_bed + " -b " + tmp_bed + " " + params
    output = subprocess.getoutput(check_cmd)
    if output:
        if report:
            print("WARNING: random negative regions encountered that do not fully overlap with reference regions:\n%s" %(output))
        if trouble_ids_dic is not None:
            add_bed_output_ids_to_dic(output, trouble_ids_dic)
        warning = True
    # Check inclusion regions.
    params = "-s -f 1 -v"
    check_cmd = "intersectBed -a " + in_bed + " -b " + incl_bed + " " + params
    output = subprocess.getoutput(check_cmd)
    if output:
        if report:
            print("WARNING: random negative regions encountered that do not fully overlap with -incl regions:\n%s" %(output))
        if trouble_ids_dic is not None:
            add_bed_output_ids_to_dic(output, trouble_ids_dic)
        warning = True
    # Check exclusion regions, throw error here (not tolerable).
    params = "-s"
    check_cmd = "intersectBed -a " + in_bed + " -b " + excl_bed + " " + params
    output = subprocess.getoutput(check_cmd)
    if output:
        assert False, "random negative regions encountered that overlap with -excl regions!"
        #assert False, "ERROR: random negative regions encountered that overlap with -excl regions!\n%s\n%s" %(check_cmd, output)
    # Delete tmp files.
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)
    # Return True if warning occured.
    return warning


################################################################################

def bed_generate_random_negatives(in_bed, chr_sizes_file, out_bed,
                                  incl_bed=False,
                                  excl_bed=False,
                                  allow_overlaps=False):
    """
    Shuffle given in_bed, generating random negative regions. Optionally,
    the regions to extract negatives from can be controlled by incl_bed
    and excl_bed.

    in_bed:
        .bed file containing regions to shuffle, i.e., generate same number
        of random negatives (with same size distribution too)
    chr_sizes_file:
        File that stores chromosome IDs and their sizes
    out_bed:
        Output random negative regions in out_bed
    incl_bed:
        Regions from which to extract random negatives
    excl_bed:
        Regions from which no random negatives should be extracted
    allow_overlaps:
        Allow random negatives to overlap with each other

    Returns:
    Function returns True if no error occured.
    If loci error occured, function returns False.
    Any other error will throw an assertion error.
    If it is not possible to get the number of random negatives with the given
    restrictions, bedtools shuffle will throw the following error:
    Error, line 3: tried 1000 potential loci for entry, but could not avoid
    excluded regions.  Ignoring entry and moving on.
    This error will be thrown for every failed attempt to find a random
    negative for a certain positive instance.


    Tool:    bedtools shuffle (aka shuffleBed)
    Version: v2.29.0
    Summary: Randomly permute the locations of a feature file among a genome.

    Usage:   bedtools shuffle [OPTIONS] -i <bed/gff/vcf> -g <genome>

    Options:
    	-excl	A BED/GFF/VCF file of coordinates in which features in -i
    		should not be placed (e.g. gaps.bed).

    	-incl	Instead of randomly placing features in a genome, the -incl
    		options defines a BED/GFF/VCF file of coordinates in which
    		features in -i should be randomly placed (e.g. genes.bed).
    		Larger -incl intervals will contain more shuffled regions.
    		This method DISABLES -chromFirst.
    	-chrom	Keep features in -i on the same chromosome.
    		- By default, the chrom and position are randomly chosen.
    		- NOTE: Forces use of -chromFirst (see below).

    	-seed	Supply an integer seed for the shuffling.
    		- By default, the seed is chosen automatically.
    		- (INTEGER)

    	-f	Maximum overlap (as a fraction of the -i feature) with an -excl
    		feature that is tolerated before searching for a new,
    		randomized locus. For example, -f 0.10 allows up to 10%
    		of a randomized feature to overlap with a given feature
    		in the -excl file. **Cannot be used with -incl file.**
    		- Default is 1E-9 (i.e., 1bp).
    		- FLOAT (e.g. 0.50)

    	-chromFirst
    		Instead of choosing a position randomly among the entire
    		genome (the default), first choose a chrom randomly, and then
    		choose a random start coordinate on that chrom.  This leads
    		to features being ~uniformly distributed among the chroms,
    		as opposed to features being distribute as a function of chrom size.

    	-bedpe	Indicate that the A file is in BEDPE format.

    	-maxTries
    		Max. number of attempts to find a home for a shuffled interval
    		in the presence of -incl or -excl.
    		Default = 1000.
    	-noOverlapping
    		Don't allow shuffled intervals to overlap.
    	-allowBeyondChromEnd
    		Allow shuffled intervals to be relocated to a position
    		in which the entire original interval cannot fit w/o exceeding
    		the end of the chromosome.  In this case, the end coordinate of the
    		shuffled interval will be set to the chromosome's length.
    		By default, an interval's original length must be fully-contained
    		within the chromosome.

    """
    # Check for bedtools.
    assert is_tool("bedtools"), "bedtools not in PATH"
    # Construct call.
    check_cmd = "bedtools shuffle "
    if excl_bed:
        check_cmd = check_cmd + "-excl " + excl_bed + " "
    if incl_bed:
        check_cmd = check_cmd + "-incl " + incl_bed + " "
    if not allow_overlaps:
        check_cmd = check_cmd + "-noOverlapping "
    check_cmd = check_cmd + "-i " + in_bed + " -g " + chr_sizes_file + " > " + out_bed
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    # Look for "tried 1000 potential loci" error.
    if error:
        if re.search("potential loci", output):
            print("WARNING: number of extracted random negatives < requested number")
            return False
        else:
            assert False, "bedtools shuffle is complaining:\n%s\n%s" %(check_cmd, output)
    else:
        return True


################################################################################

def merge_files(files_list, out_file):
    """
    Merge list of files into one output file.

    """
    assert files_list, "given files_list is empty"
    # Delete out_file if exists.
    if os.path.exists(out_file):
        os.remove(out_file)
    for f in files_list:
        assert os.path.isfile(f), "list file \"%s\" not found" % (f)
        assert f != out_file, "cat does not like to cat file into same file (%s)" %(check_cmd)
        check_cmd = "cat " + f + " >> " + out_file
        output = subprocess.getoutput(check_cmd)
        error = False
        if output:
            error = True
        assert error == False, "cat did not like your input (in_file: %s, out_file: %s):\n%s" %(f, out_file, output)


################################################################################

def center_seq_make_context_lc(cl, seq):
    """
    Take the center region of a sequence of length cl, and make flanking
    sequence parts lowercase.

    >>> cl = 4
    >>> seq = "AAACGCGTTT"
    >>> center_seq_make_context_lc(cl, seq)
    'aaaCGCGttt'

    """
    assert cl > 0, "invalid cl given"
    assert seq, "invalid sequence given"
    assert cl < len(seq), "cl != len(seq), supply a sequence > cl"
    usl = int((len(seq) - cl) / 2)
    if re.search(".{%i}.{%i}.*" %(usl, cl), seq):
        m = re.search("(.{%i})(.{%i})(.*)" %(usl, cl), seq)
        us = m.group(1)
        c = m.group(2)
        ds = m.group(3)
        new_seq = us.lower() + c + ds.lower()
        assert len(new_seq) == len(seq), "differing lengths after processing"
        return new_seq
    else:
        assert False, "regex not matching for cl = %i and sequence \"%s\"" %(cl, seq)


################################################################################

def head_file_to_new(in_file, out_file, head_c):
    """
    Select top head_c rows from in_file and copy to out_file via head.

    >>> in_file = "test_data/test2.bed"
    >>> exp_file = "test_data/test2.exp.bed"
    >>> out_file = "test_data/test2.tmp.bed"
    >>> head_file_to_new(in_file, out_file, head_c=4)
    >>> diff_two_files_identical(out_file, exp_file)
    True

    """
    assert os.path.isfile(in_file), "in_file \"%s\" not found" % (in_file)
    assert head_c > 0, "# top rows to select should be > 0"
    check_cmd = "head -" + str(head_c) + " " + in_file + " > " + out_file
    assert in_file != out_file, "head does not like to head file into same file (%s)" %(check_cmd)
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "head did not like your input (in_file: %s, out_file: %s):\n%s" %(in_file, out_file, output)


################################################################################

def fasta_output_dic(fasta_dic, fasta_out,
                     split=False,
                     split_size=60):
    """
    Output FASTA sequences dictionary (sequence_id -> sequence) to fasta_out.

    split        Split FASTA sequence for output to file
    split_size   Split size

    >>> fasta_dic = {'seq1': 'ACGTACGTACGTAC', 'seq2': 'ACGT'}
    >>> split_size = 4
    >>> fasta_exp = "test_data/test5.exp.fa"
    >>> fasta_out = "test_data/test5.tmp.fa"
    >>> fasta_output_dic(fasta_dic, fasta_out, split=True, split_size=split_size)
    >>> diff_two_files_identical(fasta_exp, fasta_out)
    True

    """
    # Check.
    assert fasta_dic, "given dictionary fasta_dic empty"
    # Write sequences to FASTA file.
    OUTFA = open(fasta_out,"w")
    for seq_id in fasta_dic:
        seq = fasta_dic[seq_id]
        if split:
            OUTFA.write(">%s\n" %(seq_id))
            for i in range(0, len(seq), split_size):
                OUTFA.write("%s\n" %((seq[i:i+split_size])))
        else:
            OUTFA.write(">%s\n%s\n" %(seq_id, seq))
    OUTFA.close()


################################################################################

def dic_remove_entries(in_dic, filter_dic):
    """
    Remove entries from in_dic dictionary, given key values from filter_dic.

    >>> in_dic = {'id1': 10, 'id2': 15, 'id3':20}
    >>> filter_dic = {'id2' : 1}
    >>> dic_remove_entries(in_dic, filter_dic)
    {'id1': 10, 'id3': 20}
    """
    # Checks.
    assert in_dic, "given dictionary in_dic empty"
    assert filter_dic, "given dictionary filter_dic empty"
    # Filter.
    for filter_id in filter_dic:
        if filter_id in in_dic:
            del in_dic[filter_id]
    return in_dic


################################################################################

def gtf_extract_unique_exon_bed(in_gtf, out_bed,
                                use_ei_labels=False):
    """
    Given a .gtf file with exon features, extract exon unique (!) regions.
    Since the Ensembl exon_id regions are not unique regarding their genomic
    coordinates, create own IDs each representing one unique genomic region
    (unique start+end+strand info).

    Output .bed will look like this (column 4 ID == new exon ID):
    chr1	1000	2000	NEXT1	0	+
    chr1	3000	4000	NEXT2	0	+
    chr1	8000	9000	NEXT3	0	-
    chr1	6000	7000	NEXT4	0	-
    ...

    use_ei_labels:
        Instead of using exon ID, just print "E" in column 4.

    """

    # Store exon ID region data.
    reg_str_dic = {}

    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        if not feature == "exon":
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        # Store exon data.
        check_reg_str = "%s,%i,%i,%s" %(chr_id,feat_s,feat_e,feat_pol)
        reg_str_dic[check_reg_str] = 1

    f.close()

    # Output genomic exon regions.
    OUTBED = open(out_bed, "w")

    assert reg_str_dic, "no exon regions read in"

    c_ex = 0
    for reg_str in reg_str_dic:
        cols = reg_str.split(",")
        c_ex += 1
        ex_id = "NEXT" + str(c_ex)
        if use_ei_labels:
            ex_id = "E"
        OUTBED.write("%s\t%s\t%s\t%s\t0\t%s\n" % (cols[0], cols[1], cols[2], ex_id, cols[3]))
    OUTBED.close()


################################################################################

def gtf_extract_exon_bed(in_gtf, out_bed,
                         out_intron_bed=False,
                         use_ei_labels=False,
                         tr_ids_dic=False):
    """
    Given a .gtf file with exon features, extract exon regions and store in
    .bed file. Optionally, a dictionary of transcript IDs can be provided,
    meaning that only exon regions from the given transcripts will be extracted.
    If out_intron_bed is set, an intronic regions .bed file will also be
    extracted, based on the exonic regions .bed information.

    Output .bed will look like this (note column 4 ID format with transcript
    ID followed by _e+exon_number):
    chr1	1000	2000	ENST001_e1	0	+
    chr1	3000	4000	ENST001_e2	0	+
    chr1	8000	9000	ENST002_e1	0	-
    chr1	6000	7000	ENST002_e2	0	-
    ...

    use_ei_labels:
    Instead of using transcript ID + eX column 4 BED ID, just use "E" for
    exon region and "I" for intron region (if out_intron_bed) set.

    NOTE that function has been tested with .gtf files from Ensembl. .gtf files
    from different sources sometimes have a slightly different format, which
    could lead to incompatibilities / errors. See test files for format that
    works.

    Some tested Ensembl GTF files:
    Homo_sapiens.GRCh38.97.gtf.gz
    Mus_musculus.GRCm38.81.gtf.gz
    Mus_musculus.GRCm38.79.gtf.gz

    >>> in_gtf = "test_data/map_test_in.gtf"
    >>> exp_out_bed = "test_data/gtf_exon_out_exp.bed"
    >>> exp_out_intron_bed = "test_data/gtf_intron_out_exp.bed"
    >>> out_bed = "test_data/gtf_exon_out.bed"
    >>> out_intron_bed = "test_data/gtf_intron_out.bed"
    >>> gtf_extract_exon_bed(in_gtf, out_bed, out_intron_bed=out_intron_bed)
    >>> diff_two_files_identical(out_bed, exp_out_bed)
    True
    >>> diff_two_files_identical(out_intron_bed, exp_out_intron_bed)
    True

    """

    # Output genomic exon regions.
    OUTBED = open(out_bed, "w")

    # Read in exon features from GTF file.
    c_gtf_ex_feat = 0
    # Start end coordinates of exons.
    exon_e_dic = {}
    exon_s_dic = {}
    # Transcript stats.
    tr2pol_dic = {}
    tr2chr_dic = {}
    # dic for sanity checking exon number order.
    tr2exon_nr_dic = {}

    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        if not feature == "exon":
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)
        # Extract exon number.
        m = re.search('exon_number "(\d+?)"', infos)
        assert m, "exon_number entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        exon_nr = int(m.group(1))

        # Check if transcript ID is in transcript dic.
        if tr_ids_dic:
            if not transcript_id in tr_ids_dic:
                continue

        # Store transcript stats.
        tr2pol_dic[transcript_id] = feat_pol
        tr2chr_dic[transcript_id] = chr_id

        # Check whether exon numbers are incrementing for each transcript ID.
        if not transcript_id in tr2exon_nr_dic:
            tr2exon_nr_dic[transcript_id] = exon_nr
        else:
            assert tr2exon_nr_dic[transcript_id] < exon_nr, "transcript ID \"%s\" without increasing exon number order in GTF file \"%s\"" %(transcript_id, in_gtf)
            tr2exon_nr_dic[transcript_id] = exon_nr

        # Count exon entry.
        c_gtf_ex_feat += 1

        # Construct exon ID.
        exon_id = transcript_id + "_e" + str(exon_nr)
        # Store infos.
        exon_s_dic[exon_id] = feat_s
        exon_e_dic[exon_id] = feat_e
        if use_ei_labels:
            exon_id = "E"
        # Output genomic exon region.
        OUTBED.write("%s\t%i\t%i\t%s\t0\t%s\n" % (chr_id,feat_s,feat_e,exon_id,feat_pol))

    OUTBED.close()
    f.close()

    # Check for read-in features.
    assert c_gtf_ex_feat, "no exon features read in from \"%s\"" %(in_gtf)

    # Output intron .bed.
    if out_intron_bed:
        tr2intron_nr_dic = {}
        OUTBED = open(out_intron_bed, "w")
        for tr_id in tr2pol_dic:
            tr_pol = tr2pol_dic[tr_id]
            chr_id = tr2chr_dic[tr_id]
            tr_c = tr2exon_nr_dic[tr_id]
            intron_c = 0
            tr2intron_nr_dic[tr_id] = 0
            # 1-exon transcripts, no introns.
            if tr_c == 1:
                continue
            ex_list = []
            for i in range(tr_c):
                ex_nr = i + 1
                ex_id = tr_id + "_e" + str(ex_nr)
                ex_list.append(ex_id)
            for i in range(len(ex_list)):
                ex1i = i
                ex2i = i + 1
                # At last exon, no more introns to add.
                if ex2i == len(ex_list):
                    break
                ex1id = ex_list[ex1i]
                ex2id = ex_list[ex2i]
                ex1s = exon_s_dic[ex1id]
                ex2s = exon_s_dic[ex2id]
                ex1e = exon_e_dic[ex1id]
                ex2e = exon_e_dic[ex2id]
                # Plus case.
                intron_s = ex1e
                intron_e = ex2s
                if tr_pol == "-":
                    intron_s = ex2e
                    intron_e = ex1s
                intron_id = tr_id + "_i" + str(ex2i)
                intron_c += 1
                if use_ei_labels:
                    intron_id = "I"
                OUTBED.write("%s\t%i\t%i\t%s\t0\t%s\n" % (chr_id,intron_s,intron_e,intron_id,tr_pol))
            tr2intron_nr_dic[tr_id] = intron_c
        OUTBED.close()
        # Sanity check exon + intron numbers.
        for tr_id in tr2exon_nr_dic:
            exon_nr = tr2exon_nr_dic[tr_id]
            intron_nr = tr2intron_nr_dic[tr_id]
            assert (exon_nr-1) == intron_nr, "intron number != exon number - 1 for \"%s\" (%i != %i - 1)" %(tr_id, intron_nr, exon_nr)


################################################################################

def gtf_extract_gene_bed(in_gtf, out_bed,
                         gene_ids_dic=False):
    """
    Extract gene regions from in_gtf GTF file, and output to out_bed BED
    file.

    gene_ids_dic:
    Dictionary with gene IDs for filtering (keeping dic IDs).

    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> exp_out_bed = "test_data/gtf_gene_out.exp.bed"
    >>> tmp_out_bed = "test_data/gtf_gene_out.tmp.bed"
    >>> gtf_extract_gene_bed(in_gtf, tmp_out_bed)
    >>> diff_two_files_identical(tmp_out_bed, exp_out_bed)
    True

    """

    # Output gene regions.
    OUTBED = open(out_bed, "w")
    c_out = 0
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        if not feature == "gene":
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        # Extract gene ID and from infos.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)

        # Check if gene ID is in gene dic.
        if gene_ids_dic:
            if not gene_id in gene_ids_dic:
                continue

        # Output genomic exon region.
        c_out += 1
        OUTBED.write("%s\t%i\t%i\t%s\t0\t%s\n" % (chr_id,feat_s,feat_e,gene_id,feat_pol))

    OUTBED.close()
    f.close()

    assert c_out, "no regions output to out_bed. Invalid in_gtf or too restrictive gene_ids_dic filtering?"


################################################################################

def gtf_extract_tsl_gene_bed(in_gtf, out_bed,
                             strict=False,
                             basic=True,
                             min_tsl=1,
                             gene_ids_dic=False):
    """
    Extract gene regions from in_gtf GTF file, and output to out_bed BED
    file.

    gene_ids_dic:
    Dictionary with gene IDs for filtering (keeping dic IDs).
    strict:
    If True only output genes with TSL=1 transcripts.

    >>> in_gtf = "test_data/test_tsl_genes.gtf"
    >>> exp_out_bed = "test_data/test_tsl_genes.exp.bed"
    >>> tmp_out_bed = "test_data/test_tsl_genes.tmp.bed"
    >>> gtf_extract_tsl_gene_bed(in_gtf, tmp_out_bed)
    >>> diff_two_files_identical(tmp_out_bed, exp_out_bed)
    True

    """

    # Extract genes with TSL+basic transcripts from gtf.
    gene2chrse_dic = {}
    gene2pol_dic = {}
    gene2keep_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        if not feature == "gene" and not feature == "transcript":
            continue

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Extract gene ID and from infos.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)

        # Check if gene ID is in genes to select dic.
        if gene_ids_dic:
            if not gene_id in gene_ids_dic:
                continue

        # Store gene chrse ,pol.
        if feature == "gene":
            gene2chrse_dic[gene_id] = "%s\t%i\t%i" %(chr_id, feat_s, feat_e)
            gene2pol_dic[gene_id] = feat_pol
            continue

        """
        We are now in transcript row
        Extract transcript ID + transcript_support_level + tag info.
        Only output genes with transcript(s) featuring TSL + tag info.

        """

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)
        # Look for basic tag.
        m = re.search('tag "basic"', infos)
        if not m:
            if basic:
                continue

        # Get transcript support level (TSL).
        m = re.search('transcript_support_level "(.+?)"', infos)
        tsl_id = "NA"
        if m:
            tsl_id = m.group(1)
            if re.search("assigned to previous", tsl_id):
                m = re.search("(.+?) \(", tsl_id)
                tsl_id = m.group(1)
        else:
            continue

        # Strict filter.
        if strict:
            if min_tsl:
                if int(tsl_id) > min_tsl:
                    continue
            else:
                if not tsl_id != "1":
                    continue

        # Store gene ID with TSL + basic support.
        gene2keep_dic[gene_id] = 1
    f.close()

    assert gene2keep_dic, "no remaining genes to output"

    # Output gene regions.
    OUTBED = open(out_bed, "w")
    for gene_id in gene2keep_dic:
        OUTBED.write("%s\t%s\t0\t%s\n" % (gene2chrse_dic[gene_id],gene_id,gene2pol_dic[gene_id]))
    OUTBED.close()


################################################################################

def gtf_extract_transcript_bed(in_gtf, out_bed,
                               tr_ids_dic=False):
    """
    Extract transcript regions from in_gtf GTF file, and output to out_bed BED
    file.

    tr_ids_dic:
    Dictionary with transcript IDs for filtering (keeping dic IDs).

    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> exp_out_bed = "test_data/gtf_transcript_out.exp.bed"
    >>> tmp_out_bed = "test_data/gtf_transcript_out.tmp.bed"
    >>> gtf_extract_transcript_bed(in_gtf, tmp_out_bed)
    >>> diff_two_files_identical(tmp_out_bed, exp_out_bed)
    True

    """

    # Output transcript regions.
    OUTBED = open(out_bed, "w")
    c_out = 0
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)

        # Check if transcript ID is in transcript dic.
        if tr_ids_dic:
            if not transcript_id in tr_ids_dic:
                continue

        # Output genomic exon region.
        c_out += 1
        OUTBED.write("%s\t%i\t%i\t%s\t0\t%s\n" % (chr_id,feat_s,feat_e,transcript_id,feat_pol))

    OUTBED.close()
    f.close()

    assert c_out, "no regions output to out_bed. Invalid in_gtf or too restrictive tr_ids_dic filtering?"


################################################################################

def get_transcript_sequences_from_gtf(in_gtf, in_2bit,
                                      lc_repeats=False,
                                      tr_ids_dic=False):
    """
    Get spliced transcript sequences based on in_gtf annotations. For
    transcripts with > 1 exon, concatenate the exon sequences to build
    the transcript sequence. If one exon is missing / not extracted or
    if extracted lengths don't fit, the transcript sequence will be
    skipped / not output.
    Return dictionary with transcript_id -> sequence mapping.

    tr_ids_dic:
        Defines transcript IDs for which sequence should be extracted.

    """
    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"
    random_id = uuid.uuid1()
    tmp_fa = str(random_id) + ".tmp.fa"

    # Transcript sequences dic.
    tr_seqs_dic = {}

    # Extract transcript exon regions from GTF and store as BED.
    gtf_extract_exon_bed(in_gtf, tmp_bed, tr_ids_dic=tr_ids_dic)

    # Extract exon region sequences from .2bit.
    bed_extract_sequences_from_2bit(tmp_bed, tmp_fa, in_2bit,
                                    lc_repeats=lc_repeats)

    # Get transcript lengths from tmp_bed for comparison.
    tr_len_dic = bed_get_transcript_lengths_from_exon_regions(tmp_bed)
    # Get exon numbers for each transcript.
    tr_exc_dic = bed_get_transcript_exon_numbers(tmp_bed)

    # Read in sequences.
    exon_seqs_dic = read_fasta_into_dic(tmp_fa)

    # Concatenate exon region sequences.
    for tr_id in tr_exc_dic:
        ex_c = tr_exc_dic[tr_id]
        for i in range(ex_c):
            i += 1
            ex_id = tr_id + "_e" + str(i)
            if ex_id in exon_seqs_dic:
                ex_seq = exon_seqs_dic[ex_id]
                if tr_id not in tr_seqs_dic:
                    tr_seqs_dic[tr_id] = ex_seq
                else:
                    tr_seqs_dic[tr_id] += ex_seq
            else:
                print("WARNING: no sequence extracted for exon ID \"%s\". Skipping \"%s\" .. " %(ex_id, tr_id))
                if tr_id in tr_seqs_dic:
                    del tr_seqs_dic[tr_id]
                break
    # Checks.
    assert tr_seqs_dic, "tr_seqs_dic empty (no FASTA sequences extracted?)"
    for tr_id in tr_seqs_dic:
        tr_len = len(tr_seqs_dic[tr_id])
        exp_len = tr_len_dic[tr_id]
        assert tr_len == exp_len, "BED transcript length != FASTA transcript length for \"%s\"" %(tr_id)

    # Delete tmp files.
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)
    if os.path.exists(tmp_fa):
        os.remove(tmp_fa)

    # Return transcript sequences dic constructed from exon sequences.
    return tr_seqs_dic


################################################################################

def bed_get_transcript_lengths_from_exon_regions(in_bed):
    """
    Get spliced transcript lengths from in_bed BED file with transcript
    exon regions, with ID format:
    transcriptid_e1 (exon 1), transcriptid_e1 (exon 2)
    This is the output format from gtf_extract_exon_bed(), so both can
    be used in combination.

    >>> in_bed = "test_data/test6.bed"
    >>> bed_get_transcript_lengths_from_exon_regions(in_bed)
    {'ENST1': 4000, 'ENST2': 1500, 'ENST3': 2500}

    """
    tr_len_dic = {}
    # Open input .bed file.
    with open(in_bed) as f:
        for line in f:
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_len = site_e - site_s
            if re.search(".+_e\d", site_id):
                m = re.search("(.+)_e\d", site_id)
                tr_id = m.group(1)
                if tr_id not in tr_len_dic:
                    tr_len_dic[tr_id] = site_len
                else:
                    tr_len_dic[tr_id] += site_len
            else:
                assert False, "site ID \"%s\" missing added _e exon number" %(site_id)
    f.close()
    assert tr_len_dic, "nothing was read in (\"%s\" empty or malformatted?)" %(in_bed)
    return tr_len_dic


################################################################################

def bed_get_transcript_exon_numbers(in_bed):
    """
    Get number of exons for each transcript from in_bed BED file with
    transcript exon regions, with ID format:
    transcriptid_e1 (exon 1), transcriptid_e1 (exon 2)
    This is the output format from gtf_extract_exon_bed(), so both can
    be used in combination.

    >>> in_bed = "test_data/test6.bed"
    >>> bed_get_transcript_exon_numbers(in_bed)
    {'ENST1': 2, 'ENST2': 2, 'ENST3': 1}

    """
    tr_exc_dic = {}
    # Open input .bed file.
    with open(in_bed) as f:
        for line in f:
            cols = line.strip().split("\t")
            site_id = cols[3]
            if re.search(".+_e\d", site_id):
                m = re.search("(.+)_e\d", site_id)
                tr_id = m.group(1)
                if tr_id not in tr_exc_dic:
                    tr_exc_dic[tr_id] = 1
                else:
                    tr_exc_dic[tr_id] += 1
            else:
                assert False, "site ID \"%s\" missing added _e exon number" %(site_id)
    f.close()
    assert tr_exc_dic, "nothing was read in (\"%s\" empty or malformatted?)" %(in_bed)
    return tr_exc_dic


################################################################################

def bed_convert_transcript_to_genomic_sites(in_bed, in_gtf, out_bed,
                                            site2hitc_dic=None,
                                            out_folder=False):
    """
    Dependencies:
    bedtools (tested with 2.29.0)
    gzip

    Convert in_bed .bed file with transcript sites into genomic coordinates
    sites file. in_bed column 1 transcript IDs have to be present in
    in_gtf GTF file, from which genomic exon coordinates of the transcript
    will get extracted.

    site2hitc_dic:
        A site2hitc_dic can be given, where site ID to hit count will be
        stored for usage outside the function.

    Output:
    By default output to out_bed file, using id_p1, id_p2 IDs.
    If out_folder=True, use out_bed name as folder name.
    In this case, output these files to folder:
    exon_regions_genome.bed
    exon_regions_transcript.bed
    complete_hits.bed
    split_hits.bed
    all_hits.bed

    >>> test_gtf = "test_data/test_tr2gen.gtf"
    >>> test_in_bed = "test_data/test_tr2gen.bed"
    >>> test_out_exp_bed = "test_data/test_tr2gen.exp.bed"
    >>> test_out_tmp_bed = "test_data/test_tr2gen.tmp.bed"
    >>> bed_convert_transcript_to_genomic_sites(test_in_bed, test_gtf, test_out_tmp_bed)
    >>> diff_two_files_identical(test_out_exp_bed, test_out_tmp_bed)
    True
    >>> test_out = "test_data/tr2gen_tmp_out"
    >>> test_out_tmp_bed = "test_data/tr2gen_tmp_out/all_hits.bed"
    >>> bed_convert_transcript_to_genomic_sites(test_in_bed, test_gtf, test_out, out_folder=True)
    >>> diff_two_files_identical(test_out_exp_bed, test_out_tmp_bed)
    True

    """

    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"
    random_id = uuid.uuid1()
    tmp_out = str(random_id) + ".tmp.out"

    # Output files if output_folder=True.
    if out_folder:
        if not os.path.exists(out_bed):
            os.makedirs(out_bed)
    out_exon_regions_genome_bed = out_bed + "/" + "exon_regions_genome.bed"
    out_exon_regions_transcript_bed = out_bed + "/" + "exon_regions_transcript.bed"
    out_unique_hits_bed = out_bed + "/" + "unique_hits.bed"
    out_split_hits_bed = out_bed + "/" + "split_hits.bed"
    out_all_hits_bed = out_bed + "/" + "all_hits.bed"

    # Transcript IDs dic.
    tr_ids_dic = bed_get_chromosome_ids(in_bed)

    # Extract transcript exon regions from GTF and store as BED.
    gtf_extract_exon_bed(in_gtf, tmp_bed, tr_ids_dic=tr_ids_dic)
    if out_folder:
        make_file_copy(tmp_bed, out_exon_regions_transcript_bed)

    # Get exon region lengths.
    exid2len_dic = bed_get_region_lengths(tmp_bed)

    # Get exon numbers for each transcript.
    tr_exc_dic = bed_get_transcript_exon_numbers(tmp_bed)

    # Read in exon region stats.
    id2chr_dic = {}
    id2s_dic = {}
    id2e_dic = {}
    id2pol_dic = {}
    exid2trid_dic = {}
    with open(tmp_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_pol = cols[5]
            id2chr_dic[site_id] = chr_id
            id2s_dic[site_id] = site_s
            id2e_dic[site_id] = site_e
            id2pol_dic[site_id] = site_pol
            if re.search(".+_e\d", site_id):
                m = re.search("(.+)_e\d", site_id)
                tr_id = m.group(1)
                exid2trid_dic[site_id] = tr_id
            else:
                assert False, "site ID \"%s\" missing added _e exon number" %(site_id)
    f.close()

    # Output exon regions with transcript coordinates.
    OUTBED = open(tmp_bed, "w")
    for tr_id in tr_exc_dic:
        ex_c = tr_exc_dic[tr_id]
        new_s = 0
        for i in range(ex_c):
            i += 1
            ex_id = tr_id + "_e" + str(i)
            gen_s = id2s_dic[ex_id]
            gen_e = id2e_dic[ex_id]
            ex_len = gen_e - gen_s
            tr_s = new_s
            tr_e = new_s + ex_len
            OUTBED.write("%s\t%i\t%i\t%s\t0\t+\n" % (tr_id,tr_s,tr_e,ex_id))
            new_s = tr_e
    OUTBED.close()

    if out_folder:
        make_file_copy(tmp_bed, out_exon_regions_genome_bed)

    # Overlap in_bed with tmp_bed.
    params = "-wb"
    intersect_bed_files(in_bed, tmp_bed, params, tmp_out,
                        sorted_out=True)

    # Read in transcript site overlaps with transcript exon regions.
    site2c_dic = {}
    # Dictionaries for later outputting unique + split hits separately.
    siteid2pol_dic = {}
    siteid2sc_dic = {}
    partid2chrse_dic = {}
    with open(tmp_out) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            tr_id = cols[0]
            part_s = int(cols[1])
            part_e = int(cols[2])
            site_id = cols[3]
            site_sc = cols[4]
            ex_s = int(cols[7])
            ex_e = int(cols[8])
            ex_id = cols[9]
            ex_pol = id2pol_dic[ex_id]
            siteid2pol_dic[site_id] = ex_pol
            siteid2sc_dic[site_id] = site_sc
            if site_id in site2c_dic:
                site2c_dic[site_id] += 1
            else:
                site2c_dic[site_id] = 1
            # Hit part number.
            hit_c = site2c_dic[site_id]
            # Calculate genomic hit coordinates.
            # Plus strand case.
            gen_s = id2s_dic[ex_id] + part_s - ex_s
            gen_e = id2s_dic[ex_id] + part_e - ex_s
            # Minus strand case.
            if ex_pol == "-":
                gen_s = id2e_dic[ex_id] - part_e + ex_s
                gen_e = id2e_dic[ex_id] - part_s + ex_s
            # part ID.
            part_id = site_id + "_p" + str(hit_c)
            # Store chrse for each part ID.
            chrse = "%s\t%i\t%i" %(id2chr_dic[ex_id],gen_s,gen_e)
            partid2chrse_dic[part_id] = "%s\t%i\t%i" %(id2chr_dic[ex_id],gen_s,gen_e)

    # Produce seperate output files for unique + split hits.
    all_hits_bed = out_bed
    if out_folder:
        all_hits_bed = out_all_hits_bed
    ALLBED = open(all_hits_bed, "w")
    if out_folder:
        UNIBED = open(out_unique_hits_bed, "w")
        SPLBED = open(out_split_hits_bed, "w")
        for site_id in site2c_dic:
            hit_c = site2c_dic[site_id]
            if site2hitc_dic is not None:
                site2hitc_dic[site_id] = hit_c
            site_pol = siteid2pol_dic[site_id]
            site_sc = siteid2sc_dic[site_id]
            # For unique hit use site ID, for split hits use part IDs.
            if hit_c == 1:
                # Unique hits.
                part_id = site_id + "_p1"
                UNIBED.write("%s\t%s\t%s\t%s\n" %(partid2chrse_dic[part_id],site_id,site_sc,site_pol))
            else:
                # Split hits.
                for i in range(hit_c):
                    i += 1
                    part_id = site_id + "_p" + str(i)
                    SPLBED.write("%s\t%s\t%s\t%s\n" %(partid2chrse_dic[part_id],part_id,site_sc,site_pol))
    # Output all hits.
    for site_id in site2c_dic:
        hit_c = site2c_dic[site_id]
        if site2hitc_dic is not None:
            site2hitc_dic[site_id] = hit_c
        site_pol = siteid2pol_dic[site_id]
        site_sc = siteid2sc_dic[site_id]
        # For unique hit use site ID, for split hits use part IDs.
        if hit_c == 1:
            # Unique hits.
            part_id = site_id + "_p1"
            ALLBED.write("%s\t%s\t%s\t%s\n" %(partid2chrse_dic[part_id],site_id,site_sc,site_pol))
        else:
            # Split hits.
            for i in range(hit_c):
                i += 1
                part_id = site_id + "_p" + str(i)
                ALLBED.write("%s\t%s\t%s\t%s\n" %(partid2chrse_dic[part_id],part_id,site_sc,site_pol))

    # Delete tmp files.
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)
    if os.path.exists(tmp_out):
        os.remove(tmp_out)


################################################################################

def check_convert_chr_id(chr_id):
    """
    Check and convert chromosome IDs to format:
    chr1, chr2, chrX, ...
    If chromosome IDs like 1,2,X, .. given, convert to chr1, chr2, chrX ..
    Return False if given chr_id not standard and not convertable.

    Filter out scaffold IDs like:
    GL000009.2, KI270442.1, chr14_GL000009v2_random
    chrUn_KI270442v1 ...

    >>> chr_id = "chrX"
    >>> check_convert_chr_id(chr_id)
    'chrX'
    >>> chr_id = "4"
    >>> check_convert_chr_id(chr_id)
    'chr4'
    >>> chr_id = "MT"
    >>> check_convert_chr_id(chr_id)
    'chrM'
    >>> chr_id = "GL000009.2"
    >>> check_convert_chr_id(chr_id)
    False
    >>> chr_id = "chrUn_KI270442v1"
    >>> check_convert_chr_id(chr_id)
    False

    """
    assert chr_id, "given chr_id empty"

    if re.search("^chr", chr_id):
        if not re.search("^chr[\dMXY]+$", chr_id):
            chr_id = False
    else:
        # Convert to "chr" IDs.
        if chr_id == "MT":
            chr_id = "M"
        if re.search("^[\dMXY]+$", chr_id):
            chr_id = "chr" + chr_id
        else:
            chr_id = False
    return chr_id


################################################################################

def bed_get_chromosome_ids(bed_file,
                           std_chr_filter=False,
                           ids_dic=False):
    """
    Read in .bed file, return chromosome IDs (column 1 IDs).
    Return dic with chromosome ID -> count mapping.

    ids_dic:
        A non-empty ids_dic can be supplied, resulting in chromosome IDs
        to be added to the existing ids_dic dictionary.
    std_chr_filter:
        Filter / convert chromosome IDs with function check_convert_chr_id(),
        removing non-standard chromosomes, and convert IDs like 1,2,X,MT ..
        to chr1, chr2, chrX, chrM.

    >>> test_file = "test_data/test6.bed"
    >>> bed_get_chromosome_ids(test_file)
    {'chr1': 2, 'chr2': 2, 'chr3': 1}

    """
    if not ids_dic:
        ids_dic = {}
    with open(bed_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            # Check ID.
            if std_chr_filter:
                new_chr_id = check_convert_chr_id(chr_id)
                # If not standard chromosome ID or conversion failed, skip entry.
                if not new_chr_id:
                    continue
                else:
                    chr_id = new_chr_id
            if chr_id in ids_dic:
                ids_dic[chr_id] += 1
            else:
                ids_dic[chr_id] = 1
    f.closed
    assert ids_dic, "No chromosome IDs read into dictionary (input file \"%s\" empty or malformatted? Chromosome IDs filter activated?)" % (bed_file)
    return ids_dic


################################################################################

def bed_get_score_filtered_count(bed_file, sc_thr,
                                 rev_filter=False):

    """
    Read in BED file and count how many rows remain after filtering
    column 5 scores by sc_thr. By default, assume higher score == better
    score. Set rev_filter=True to reverse this.

    rev_filter:
    Set True to reverse filtering.

    >>> test_file = "test_data/test5.bed"
    >>> bed_get_score_filtered_count(test_file, 3)
    1
    >>> bed_get_score_filtered_count(test_file, 3, rev_filter=True)
    2
    >>> bed_get_score_filtered_count(test_file, 4)
    0

    """
    c_rem = 0
    with open(bed_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_sc = float(cols[4])
            if rev_filter:
                if site_sc > sc_thr:
                    continue
            else:
                if site_sc < sc_thr:
                    continue
            c_rem += 1
    f.closed
    return c_rem


################################################################################

def gtf_extract_exon_numbers(in_gtf,
                             tr_ids_dic=False):
    """
    Given a .gtf file with exon features, return dictionary with transcript
    ID and exon number.

    tr_ids_dic:
    Give tr_ids_dic dictionary with transcript IDs to keep.

    >>> in_gtf = "test_data/test_border_annot.gtf"
    >>> tr_ids_dic = {'ENST1': 1, 'ENST2': 1, 'ENST3': 1}
    >>> gtf_extract_exon_numbers(in_gtf, tr_ids_dic=tr_ids_dic)
    {'ENST1': 1, 'ENST2': 2, 'ENST3': 2}

    """

    # Transcript ID to exon count dic.
    tr2exc_dic = {}
    # dic for sanity checking exon number order.
    tr2exon_nr_dic = {}

    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        infos = cols[8]
        if not feature == "exon":
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)
        # Extract exon number.
        m = re.search('exon_number "(\d+?)"', infos)
        assert m, "exon_number entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        exon_nr = int(m.group(1))

        if tr_ids_dic:
            if transcript_id not in tr_ids_dic:
                continue

        # Count exon numbers.
        if not transcript_id in tr2exc_dic:
            tr2exc_dic[transcript_id] = 1
        else:
            tr2exc_dic[transcript_id] += 1

        # Check whether exon numbers are incrementing for each transcript ID.
        if not transcript_id in tr2exon_nr_dic:
            tr2exon_nr_dic[transcript_id] = exon_nr
        else:
            assert tr2exon_nr_dic[transcript_id] < exon_nr, "transcript ID \"%s\" without increasing exon number order in GTF file \"%s\"" %(transcript_id, in_gtf)
            tr2exon_nr_dic[transcript_id] = exon_nr
    f.close()

    # Check for read-in content.
    assert tr2exc_dic, "no exon features read in from \"%s\"" %(in_gtf)
    # Return to the castle.
    return tr2exc_dic


################################################################################

def bed_overlap_with_genomic_features(in_bed, feat_bed,
                                      out_file=False,
                                      int_whole_nr=True,
                                      use_feat_sc=False):
    """
    Overlap genomic regions in_bed BED with feature regions feat_bed BED.
    Return a dictionary of lists, with key = in_bed region ID and value
    a list of positions with length = region length, indicating for each
    position overlap (value = 1 or feat_bed region score) or not (value = 0).
    This means, each genomic position inside in_bed that overlaps with a
    region inside feat_bed will get either a value of 1 in the list, or
    the score of the overlapping region inside feat_bed (if use_feat_sc=True).
    Each genomic position inside in_bed not overlapping with feat_bed
    regions will get a 0 assigned. Note that the order of the list is
    the order of the sequence nucleotides (not the genomic position, which
    can be reversed for minus strand features).

    in_bed:
    Input BED regions file to add positional feature annotations to.
    feat_bed:
    Feature BED regions file from where to get positional annotations from.
    out_file:
    Output in_bed annotations to file.
    Format is:
    >region_id
    0
    1
    ...
    use_feat_sc:
    Use overlapping feature region scores instead of a value of 1
    for overlapping regions.

    >>> in_bed = "test_data/test7.in.bed"
    >>> feat_bed = "test_data/test7.feat.bed"
    >>> exp_out = "test_data/test7.exp.out"
    >>> tmp_out = "test_data/test7.tmp.out"
    >>> bed_overlap_with_genomic_features(in_bed, feat_bed, out_file=tmp_out, use_feat_sc=True)
    {'reg1': [0, 0, 5, 5, 5], 'reg2': [5, 0, 0, 0, 0]}
    >>> diff_two_files_identical(exp_out, tmp_out)
    True
    >>> bed_overlap_with_genomic_features(in_bed, feat_bed)
    {'reg1': [0, 0, 1, 1, 1], 'reg2': [1, 0, 0, 0, 0]}

    """

    # Check.
    assert is_tool("bedtools"), "bedtools not in PATH"
    assert os.path.isfile(in_bed), "cannot open in_bed BED file \"%s\"" % (in_bed)
    assert os.path.isfile(feat_bed), "cannot open feat_bed BED file \"%s\"" % (feat_bed)

    # Generate .tmp files.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"

    # Get polarity, start, end for each site ID.
    id2pol_dic = {}
    id2s_dic = {}
    id2e_dic = {}
    # Dictionary of lists, store position labels, init with 0.
    id2labels_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_pol = cols[5]
            id2pol_dic[site_id] = site_pol
            id2s_dic[site_id] = site_s
            id2e_dic[site_id] = site_e
            site_l = site_e - site_s
            assert site_l, "invalid site length for row \"%s\" in in_bed \"%s\"" %(row, in_bed)
            id2labels_dic[site_id] = [0]*site_l
    f.closed
    assert id2pol_dic, "no entries read into dictionary (input file \"%s\" empty or malformatted?)" % (in_bed)

    # Run overlap calculation to get overlapping feature regions.
    intersect_params = "-s -wb"
    intersect_bed_files(in_bed, feat_bed, intersect_params, tmp_bed)
    with open(tmp_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            s = int(cols[1]) + 1 # Make one-based.
            e = int(cols[2])
            site_id = cols[3]
            site_s = id2s_dic[site_id] + 1 # Make one-based.
            site_e = id2e_dic[site_id]
            site_pol = id2pol_dic[site_id]
            feat_sc = float(cols[10])
            # Check whether score is whole number.
            if int_whole_nr:
                if not feat_sc % 1:
                    feat_sc = int(feat_sc)
            # + case.
            if site_pol == "+":
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = i - site_s
                        if use_feat_sc:
                            id2labels_dic[site_id][li] = feat_sc
                        else:
                            id2labels_dic[site_id][li] = 1
            else:
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = site_e - i
                        if use_feat_sc:
                            id2labels_dic[site_id][li] = feat_sc
                        else:
                            id2labels_dic[site_id][li] = 1
    f.closed

    # It output to file enabled.
    if out_file:
        # Write labels to file.
        OUTLAB = open(out_file,"w")
        for site_id in id2labels_dic:
            OUTLAB.write(">%s\n" %(site_id))
            for label in id2labels_dic[site_id]:
                OUTLAB.write("%s\n" %(str(label)))
        OUTLAB.close()

    # Remove tmp files.
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)

    # Return dictionary of lists.
    return id2labels_dic


################################################################################

def get_uc_lc_sequence_segment(seq, cp,
                               vp_ext=20,
                               con_ext=0):
    """
    Given a sequence, a center position inside the sequence, a viewpoint
    extension value, and a context extension value: get the lowercase-
    uppercase-lowercase sequence segment.

    cp:
    1-based position that marks the center of the sequence.

    >>> seq = "ACGTacgtXYZ"
    >>> cp = 7
    >>> vp_ext = 3
    >>> con_ext = 2
    >>> get_uc_lc_sequence_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    'cgTACGTXYz'
    >>> cp = 7
    >>> vp_ext = 20
    >>> con_ext = 20
    >>> get_uc_lc_sequence_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    'ACGTACGTXYZ'
    >>> cp = 5
    >>> vp_ext = 0
    >>> con_ext = 0
    >>> get_uc_lc_sequence_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    'A'
    >>> cp = 5
    >>> vp_ext = 0
    >>> con_ext = 2
    >>> get_uc_lc_sequence_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    'gtAcg'
    >>> cp = 5
    >>> vp_ext = 2
    >>> con_ext = 0
    >>> get_uc_lc_sequence_segment(seq, cp, vp_ext=vp_ext, con_ext=con_ext)
    'GTACG'

    """
    # Checks.
    lseq = len(seq)
    assert lseq, "lseq evaluated to False"
    assert cp <= lseq, "given cp > lseq"
    assert cp > 0, "given cp < 1"

    # Upstream extensions.
    usucs = cp - vp_ext - 1
    usuce = cp - 1
    uslcs = cp - vp_ext - con_ext - 1
    uslce = usucs
    # Downstream extensions.
    dsucs = cp
    dsuce = cp + vp_ext
    dslcs = dsuce
    dslce = cp + vp_ext + con_ext

    # Extract segments.
    usucseg = seq[usucs:usuce].upper()
    uslcseg = seq[uslcs:uslce].lower()
    dsucseg = seq[dsucs:dsuce].upper()
    dslcseg = seq[dslcs:dslce].lower()
    cpseg = seq[cp-1:cp].upper()

    # Give it to me.
    final_seg = uslcseg + usucseg + cpseg + dsucseg + dslcseg
    return final_seg


################################################################################

def gtf_extract_most_prominent_transcripts(in_gtf, out_file,
                                           strict=False,
                                           min_len=False,
                                           report=False,
                                           return_ids_dic=None,
                                           set_ids_dic=False,
                                           add_infos=False):
    """
    Extract most prominent transcripts list from in_gtf.

    in_gtf:
    Genomic annotations (hg38) GTF file (.gtf or .gtf.gz)
    NOTE: tested with Ensembl GTF files, expects transcript
    support level (TSL) information.
    out_file:
        File to output transcript IDs (optionally with add_infos)
    min_len:
        Accept only transcripts with length >= --min-len
    strict:
        Accept only transcripts with transcript support level (TSL) 1-5
    return_ids_dic:
        If dictionary is given, return IDs in dictionary and do not output
        to file.
    set_ids_dic:
        Optionally provide transcript IDs which should be chosen as most
        prominent transcript for their respective gene.
    add_infos:
        Add additional information columns (gene ID, TSL, length) to out_list
        output file.

    >>> in_gtf = "test_data/test_most_prom_select.gtf"
    >>> out_file = "dummy"
    >>> ids_dic = {}
    >>> gtf_extract_most_prominent_transcripts(in_gtf, out_file,return_ids_dic=ids_dic)
    {'ENST02': 10000, 'ENST05': 8000}
    >>> ids_dic = {}
    >>> gtf_extract_most_prominent_transcripts(in_gtf, out_file,return_ids_dic=ids_dic,strict=True)
    {'ENST05': 8000}

    """

    # Comparison dictionary.
    id2sc = {}
    for i in range(5):
        pos = i + 1
        pos_str = "%i" %(pos)
        id2sc[pos_str] = pos
    id2sc["NA"] = 6

    if report:
        if strict:
            print("Strict transcript selection enabled ... ")
        if add_infos:
            print("Additional transcript infos in output file enabled ... ")

    # Read in transcript length (exonic regions).
    if report:
        print("Read in transcript lengths (exonic lengths) from GTF ... ")
    tr2exc_dic = {}
    tr2len_dic = gtf_get_transcript_lengths(in_gtf, tr2exc_dic=tr2exc_dic)
    assert tr2len_dic, "no transcript lengths read in from --gtf (invalid file format?)"
    if report:
        print("# transcripts read in:  %i" %(len(tr2len_dic)))

    # Store most prominent transcript.
    g2tr_id = {}
    g2tr_tsl = {}
    g2tr_len = {}
    g2tr_bt = {}
    g2gn = {}
    g2gbt = {}

    if report:
        print("Extract most prominent transcripts ... ")

    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Extract gene ID.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)
        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_id = m.group(1)
        # Extract gene name.
        m = re.search('gene_name "(.+?)"', infos)
        assert m, "gene_name entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_name = m.group(1)
        # Extract gene biotype.
        m = re.search('gene_biotype "(.+?)"', infos)
        assert m, "gene_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_biotype = m.group(1)
        # Extract transcript biotype.
        m = re.search('transcript_biotype "(.+?)"', infos)
        assert m, "transcript_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_biotype = m.group(1)

        # Transcript length.
        tr_len = tr2len_dic[tr_id]
        # Gene name.
        g2gn[gene_id] = gene_name
        # Gene biotype.
        g2gbt[gene_id] = gene_biotype

        # If dictionary with transcript IDs given that should be selected.
        if set_ids_dic:
            if tr_id in set_ids_dic:
                g2tr_id[gene_id] = tr_id
                g2tr_len[gene_id] = tr_len
                g2tr_tsl[gene_id] = "1"
                g2tr_bt[gene_id] = tr_biotype
                continue

        # Look for basic tag.
        m = re.search('tag "basic"', infos)
        if not m:
            continue
        # Get transcript support level (TSL).
        m = re.search('transcript_support_level "(.+?)"', infos)
        tsl_id = "NA"
        if m:
            tsl_id = m.group(1)
            if re.search("assigned to previous", tsl_id):
                m = re.search("(.+?) \(", tsl_id)
                tsl_id = m.group(1)

        # More filtering.
        if strict:
            if tsl_id == "NA":
                continue
        if min_len:
            if tr_len < min_len:
                continue

        # Update most prominent transcript.
        if not gene_id in g2tr_id:
            g2tr_id[gene_id] = tr_id
            g2tr_len[gene_id] = tr_len
            g2tr_tsl[gene_id] = tsl_id
            g2tr_bt[gene_id] = tr_biotype
        else:
            if id2sc[tsl_id] < id2sc[g2tr_tsl[gene_id]]:
                g2tr_id[gene_id] = tr_id
                g2tr_len[gene_id] = tr_len
                g2tr_tsl[gene_id] = tsl_id
                g2tr_bt[gene_id] = tr_biotype
            elif id2sc[tsl_id] == id2sc[g2tr_tsl[gene_id]]:
                if tr_len > g2tr_len[gene_id]:
                    g2tr_id[gene_id] = tr_id
                    g2tr_len[gene_id] = tr_len
                    g2tr_tsl[gene_id] = tsl_id
                    g2tr_bt[gene_id] = tr_biotype
    f.close()

    assert g2tr_id, "No IDs read into dictionary (input file \"%s\" empty or malformatted?)" % (in_gtf)
    c_prom_tr = len(g2tr_id)
    if report:
        print("Number of selected transcripts: %i" %(c_prom_tr))

    # If transcript IDs should be output to out_file.
    if return_ids_dic is None:
        # Output transcript IDs list.
        OUT = open(out_file, "w")
        if add_infos:
            OUT.write("gene_id\tgene_name\tgene_biotype\ttr_id\ttr_biotype\ttr_len\ttr_exc\ttsl\n")
        for gene_id in g2tr_id:
            tr_id = g2tr_id[gene_id]
            tr_len = g2tr_len[gene_id]
            tsl_id = g2tr_tsl[gene_id]
            tr_bt = g2tr_bt[gene_id]
            tr_exc = tr2exc_dic[tr_id]
            gene_name = g2gn[gene_id]
            gene_bt = g2gbt[gene_id]
            if add_infos:
                OUT.write("%s\t%s\t%s\t%s\t%s\t%i\t%i\t%s\n" % (gene_id,gene_name,gene_bt,tr_id,tr_bt,tr_len,tr_exc,tsl_id))
            else:
                OUT.write("%s\n" % (tr_id))
        OUT.close()
        if report:
            if add_infos:
                print("%i transcript IDs + additional infos written to:\n%s" %(c_prom_tr, out_file))
            else:
                print("%i transcript IDs written to:\n%s" %(c_prom_tr, out_file))
    else:
        for gene_id in g2tr_id:
            tr_id = g2tr_id[gene_id]
            tr_len = g2tr_len[gene_id]
            return_ids_dic[tr_id] = tr_len
        assert return_ids_dic, "no most prominent transcript IDs selected"
        return return_ids_dic


################################################################################

def gtf_get_transcript_lengths(in_gtf,
                               tr2exc_dic=None):
    """
    Get transcript lengths (= length of their exons, not unspliced length!)
    from GTF file.

    tr2exc_dic:
    Optionally provide a transcript ID to exon count dictionary for counting
    transcript exons.

    >>> in_gtf = "test_data/map_test_in.gtf"
    >>> gtf_get_transcript_lengths(in_gtf)
    {'ENST001': 2000, 'ENST002': 2000}

    """
    # Transcript ID to exonic length dictionary.
    tr2len_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        infos = cols[8]
        if not feature == "exon":
            continue
        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_id = m.group(1)
        # Sum up length.
        ex_len = feat_e - feat_s + 1
        if not tr_id in tr2len_dic:
            tr2len_dic[tr_id] = ex_len
        else:
            tr2len_dic[tr_id] += ex_len
        if tr2exc_dic is not None:
            if not tr_id in tr2exc_dic:
                tr2exc_dic[tr_id] = 1
            else:
                tr2exc_dic[tr_id] += 1
    f.close()
    assert tr2len_dic, "No IDs read into dictionary (--gtf file \"%s\" empty or malformatted?)" % (in_gtf)
    return tr2len_dic


################################################################################

def shuffle_sequences(seqs_dic,
                      new_ids=False,
                      id_prefix="CLIP",
                      di_shuffle=False):
    """
    Shuffle sequences given by seqs_dic (key = sequence ID, value = sequence).
    Return shuffled sequences in new dictionary, optionally with new IDs.

    new_ids:
    Assign new IDs to shuffled sequences.
    id_prefix:
    Use this ID prefix for the new sequence IDs (if new_ids=True).
    di_shuffle:
    Apply di-nucleotide shuffling to sequences, to preserve di-nucleotide
    frequencies.

    """

    new_seqs_dic = {}
    assert seqs_dic, "given seqs_dic dictionary empty?"
    c_ids = 0
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        c_ids += 1
        new_id = seq_id
        if new_ids:
            new_id = id_prefix + "_" + c_ids
        if len(seq) < 3:
            new_seqs_dic[new_id] = seq
        else:
            if di_shuffle:
                new_seq = shuffle_difreq(seq)
                new_seqs_dic[new_id] = new_seq
            else:
                seq_list = list(seq)
                random.shuffle(seq_list)
                new_seq = ''.join(seq_list)
                new_seqs_dic[new_id] = new_seq
    assert new_seqs_dic, "generated new_seqs_dic dictionary empty?"
    return new_seqs_dic


################################################################################

def shuffle_difreq(seq):
    """
    Does di-nucleotide shuffling of sequences, preserving the frequences.
    Code found here:
    https://www.biostars.org/p/66004/

    """
    from collections import Counter
    weighted_choice = lambda s : random.choice(sum(([v]*wt for v,wt in s),[]))

    # Get di-nucleotide frequencies.
    freqs = difreq(seq)

    # get the first base by the total frequency across the sequence
    shuff_seq = [None]
    while not shuff_seq[0] in freqs:
        shuff_seq = [weighted_choice(Counter(seq).items())]

    while len(shuff_seq) < len(seq):
        # each following base is based of the frequency of the previous base
        # and their co-occurrence in the original sequence.
        try:
            shuff_seq.append(weighted_choice(freqs[shuff_seq[-1]].items()))
        except KeyError:
            shuff_seq.pop()
    assert len(shuff_seq) == len(seq)
    return "".join(shuff_seq)


################################################################################

def difreq(seq):
    """
    Does di-nucleotide shuffling of sequences, preserving the frequences.
    Code found here:
    https://www.biostars.org/p/66004/

    """
    from collections import defaultdict
    counts = defaultdict(lambda: defaultdict(int))
    for a, b in zip(seq, seq[1:]):
        counts[a][b] += 1
    return dict((k, dict(v)) for k,v in counts.items())


################################################################################

def output_chromosome_lengths_file(len_dic, out_file,
                                   ids2print_dic=None):
    """
    Output chromosome lengths file with format:
    sequence_ID<tab>sequence_length
    """
    LOUT = open(out_file, "w")
    c_pr = 0
    for seq_id in len_dic:
        if ids2print_dic is not None:
            if seq_id in ids2print_dic:
                c_pr += 1
                LOUT.write("%s\t%i\n" %(seq_id, len_dic[seq_id]))
        else:
            c_pr += 1
            LOUT.write("%s\t%i\n" %(seq_id, len_dic[seq_id]))
    LOUT.close()
    assert c_pr, "nothing was printed out"


################################################################################

def bed_sequence_lengths_to_bed(len_dic, out_file,
                                ids_dic=None):
    """
    Given a dictionary of sequence lengths (sequence_id -> sequence_length),
    output the sequence regions to BED, with sequence ID as column 1 + 4,
    region start = 0, and region end = sequence_length.

    ids_dic:
    Dictionary with IDs to output (instead of all IDs in len_dic).

    >>> len_dic = {"tr1": 100, "tr2": 50}
    >>> out_exp_bed = "test_data/test_lengths_to_bed.exp.bed"
    >>> out_tmp_bed = "test_data/test_lengths_to_bed.tmp.bed"
    >>> bed_sequence_lengths_to_bed(len_dic, out_tmp_bed)
    >>> diff_two_files_identical(out_exp_bed, out_tmp_bed)
    True

    """
    assert len_dic, "given dictionary len_dic is empty"
    LOUT = open(out_file, "w")
    c_out = 0
    for seq_id in len_dic:
        seq_len = len_dic[seq_id]
        if ids_dic is not None:
            if seq_id in ids_dic:
                c_out += 1
                LOUT.write("%s\t0\t%i\t%s\t0\t+\n" %(seq_id, seq_len, seq_id))
        else:
            c_out += 1
            LOUT.write("%s\t0\t%i\t%s\t0\t+\n" %(seq_id, seq_len, seq_id))
    LOUT.close()
    assert c_out, "no sequence regions output to BED file"


################################################################################

def extract_transcript_sequences(bed_dic, seq_dic,
                                 ext_lr=False,
                                 revcom=False,
                                 full_hits_only=False):
    """
    Given a dictionary with bed regions (region ID -> BED row) and a
    sequence dictionary (Sequence ID -> sequence), extract the BED region
    sequences and return in new dictionary (region ID -> region sequence).

    ext_lr:
    Optionally, extend regions by ext_lr nt (up- and downstream).
    In case full extension is not possible, use maximum extension possible.

    revcom:
    if revcom=True and strand of bed_dic region is "-", return the reverse
    complement of the region sequence.

    full_hits_only:
    Set full_hits_only=True to only recover full hits.

    >>> seq_dic = {"T1" : "AAAACCCCGGGGTTTT", "T2" : "ATATACACAGAGCGCGCTCTGTGT"}
    >>> bed_dic = {"S1" : "T1\\t4\\t8\\tS1\\t0\\t+", "S2" : "T2\\t6\\t8\\tS2\\t0\\t+"}
    >>> extract_transcript_sequences(bed_dic, seq_dic, ext_lr=2)
    {'S1': 'AACCCCGG', 'S2': 'ACACAG'}
    >>> extract_transcript_sequences(bed_dic, seq_dic, ext_lr=5, full_hits_only=True)
    {'S2': 'TATACACAGAGC'}

    """
    id2seq_dic = {}
    # Process .bed regions.
    for reg_id in bed_dic:
        cols = bed_dic[reg_id].split("\t")
        seq_id = cols[0]
        reg_s = int(cols[1])
        reg_e = int(cols[2])
        reg_pol = cols[5]
        assert seq_id in seq_dic, "sequence ID \"%s\" not found in given sequence dictionary" %(seq_id)
        seq = seq_dic[seq_id]
        # Update region borders.
        new_s = reg_s
        new_e = reg_e
        exp_l = new_e - new_s
        # Adjust if given start or end is out of bounds.
        if new_s < 0:
            new_s = 0
        if new_e > len(seq):
            new_e = len(seq)
        # If region should be extended up- and downstream by ext_lr.
        if ext_lr:
            new_s = new_s - ext_lr
            new_e = reg_e + ext_lr
            exp_l = new_e - new_s
            # If start or end is out of bounds after extension.
            if new_s < 0:
                new_s = 0
            if new_e > len(seq):
                new_e = len(seq)
        reg_seq = seq[new_s:new_e]
        reg_l = len(reg_seq)
        if full_hits_only:
            if not reg_l == exp_l:
                continue
        if revcom:
            if reg_pol == "-":
                id2seq_dic[reg_id] = revcom_seq(reg_seq)
            else:
                id2seq_dic[reg_id] = reg_seq
        else:
            id2seq_dic[reg_id] = reg_seq
    assert id2seq_dic, "no sequences extracted"
    return id2seq_dic


################################################################################

def revcom_seq(seq,
               upper=False,
               convert_to_rna=False):
    """
    Return reverse complement to seq. By default, convert seq to uppercase
    and translate to DNA.

    # Convert to RNA.
    if convert_to_rna:
        new_seq_rna = new_seq.replace("T","U").replace("t","u")
        new_seq = new_seq_rna

    >>> seq = "AAACAGatt"
    >>> revcom_seq(seq)
    'aatCTGTTT'
    >>> revcom_seq(seq, upper=True)
    'AATCTGTTT'
    >>> revcom_seq(seq, convert_to_rna=True)
    'aauCUGUUU'

    """
    assert seq, "given sequence empty"
    # Make uppercase and convert to DNA.
    if upper:
        seq = seq[::-1].upper().replace("U","T")
    else:
        seq = seq[::-1].replace("U","T").replace("u","t")
    intab = "ACGTacgt"
    outtab = "TGCAtgca"
    # If RNA revcom should be output.
    if convert_to_rna:
        seq = seq.replace("T","U").replace("t","u")
        intab = "ACGUacgu"
        outtab = "UGCAugca"
    # Make revcom.
    transtab = str.maketrans(intab, outtab)
    rc_seq = seq.translate(transtab)
    return rc_seq


################################################################################

def random_order_dic_keys_into_list(in_dic):
    """
    Read in dictionary keys, and return random order list of IDs.

    """
    import random
    id_list = []
    for key in in_dic:
        id_list.append(key)
    random.shuffle(id_list)
    return id_list


################################################################################

def ushuffle_sequences(seqs_dic,
                       new_ids=False,
                       id2vpse_dic=False,
                       id_prefix="CLIP",
                       ushuffle_k=1):

    """
    Shuffle sequences given by seqs_dic (key = sequence ID, value = sequence).
    Return shuffled sequences in new dictionary, optionally with new IDs.
    This function uses uShuffle, available in Python here:
    https://github.com/guma44/ushuffle

    uShuffle can be installed inside conda environment with:
    pip install ushuffle

    Example code for Python 3 (input and output are byte objects, not strings):
    from ushuffle import shuffle, Shuffler
    seq = b"acgtgattagctagct"
    shuffler = Shuffler(seq, 2)
    for i in range(5):
        seqres = shuffler.shuffle()
        print("results:", seqres)
    print(shuffle(seq, 2))

    Output:
    results: b'agctacgatgttagct'
    results: b'atagctacgagttgct'
    results: b'atgctagcgagtactt'
    results: b'agcgctgatacttagt'
    results: b'agtgattagctacgct'
    b'agcgagctgttactat'

    new_ids:
        Assign new IDs to shuffled sequences.
    id_prefix:
        Use this ID prefix for the new sequence IDs (if new_ids=True).
    ushuffle_k:
        Supply ushuffle_k for k-nucleotide shuffling.
    id2vpse_dic:
        Dictionary with sequence ID -> [viewpoint_start, viewpoint_end].
        Use it to restore lowercase uppercase regions for each sequence,
        after shuffling.

    """
    from ushuffle import shuffle
    new_seqs_dic = {}
    assert seqs_dic, "given seqs_dic dictionary empty?"
    c_ids = 0
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        seq = seq.upper()
        if id2vpse_dic: # note: 1-based coords inside dic.
            assert seq_id in id2vpse_dic, "sequence ID %s not in id2vpse_dic" %(seq_id)
            vp_s = id2vpse_dic[seq_id][0]
            vp_e = id2vpse_dic[seq_id][1]
        c_ids += 1
        new_id = seq_id
        if new_ids:
            new_id = id_prefix + "_" + str(c_ids)
            if id2vpse_dic:
                id2vpse_dic[new_id] = [vp_s, vp_e]
        if len(seq) < 3:
            new_seqs_dic[new_id] = seq
        else:
            # String to byte object.
            seq_bo = seq.encode('ASCII')
            shuff_seq_bo = shuffle(seq_bo, ushuffle_k)
            shuff_seq = shuff_seq_bo.decode('ASCII')
            # Restore lowercase uppercase structure of original positive sequence.
            if id2vpse_dic:
                new_seqs_dic[new_id] = add_lowercase_context_to_sequences(shuff_seq, vp_s, vp_e)
            else:
                new_seqs_dic[new_id] = shuff_seq
    assert new_seqs_dic, "generated new_seqs_dic dictionary empty?"
    return new_seqs_dic


################################################################################

def add_lowercase_context_to_sequences(seq, uc_s, uc_e,
                                       convert_to_rna=False):
    """
    Given a sequence and uppercase middle region start (uc_s) and end (uc_e),
    make context region upstream + downstream lowercase.
    Two coordinates should be one-based.
    Return lowercase-uppercase-lowercase sequence.

    convert_to_rna:
    If True, convert new sequence to RNA.

    >>> seq = "AAAACCCCGGGGTTTT"
    >>> add_lowercase_context_to_sequences(seq, 5, 12, convert_to_rna=True)
    'aaaaCCCCGGGGuuuu'
    >>> add_lowercase_context_to_sequences(seq, 1, 8)
    'AAAACCCCggggtttt'
    >>> add_lowercase_context_to_sequences(seq, 15, 16)
    'aaaaccccggggttTT'

    """
    # Checks.
    seq_l = len(seq)
    assert uc_s < uc_e, "uc_s < uc_e not satisfied"
    assert uc_s > 0, "uc_s > 0 not satisfied"
    assert seq_l >= uc_e, "uppercase region end > sequence length"
    us_seq = seq[:uc_s-1].lower()
    center_seq = seq[uc_s-1:uc_e].upper()
    ds_seq = seq[uc_e:].lower()
    # New sequence.
    new_seq = us_seq + center_seq + ds_seq
    # Convert to RNA.
    if convert_to_rna:
        new_seq_rna = new_seq.replace("T","U").replace("t","u")
        new_seq = new_seq_rna
    return new_seq


################################################################################

def gtf_get_gene_ids_from_transcript_ids(tr_ids_dic, in_gtf,
                                         gene_ids_dic=False):
    """
    Get gene IDs for a dictionary of transcript IDs,
    returning dictionary with transcript ID (key) mapped to its gene ID (value).

    gene_ids_dic:
    If set, return dictionary with mapped gene IDs as keys only.

    >>> tr_ids_dic = {'ENST01': 1, 'ENST02': 1}
    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_gene_ids_from_transcript_ids(tr_ids_dic, in_gtf)
    {'ENST01': 'ENSG01', 'ENST02': 'ENSG02'}
    >>> gtf_get_gene_ids_from_transcript_ids(tr_ids_dic, in_gtf, gene_ids_dic=True)
    {'ENSG01': 1, 'ENSG02': 1}

    """
    # Checks.
    assert tr_ids_dic, "given dictionary tr_ids_dic empty"
    # Output dic.
    out_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)

        if not transcript_id in tr_ids_dic:
            continue

        # Extract gene ID.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)

        if gene_ids_dic:
            out_dic[gene_id] = 1
        else:
            out_dic[transcript_id] = gene_id
    f.close()
    # Check and return to barracks.
    assert out_dic, "out_dic empty, transcript features or gene IDs read in"
    return out_dic


################################################################################

def gtf_get_gene_biotypes_from_transcript_ids(tr_ids_dic, in_gtf,
                                              all_gbtc_dic=None):
    """
    Get gene IDs from dictionary of transcript IDs (tr_ids_dic), and based
    on these gene IDs create a dictionary of gene biotype counts.
    Return dictionary of gene biotype counts.

    all_gbtc_dic:
    If set, fill this dictionary with gene biotype counts for all genes.

    >>> tr_ids_dic = {'ENST01': 1, 'ENST02': 1}
    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_gene_biotypes_from_transcript_ids(tr_ids_dic, in_gtf)
    {'transcribed_unprocessed_pseudogene': 2}

    """
    # Checks.
    assert tr_ids_dic, "given dictionary tr_ids_dic empty"
    # transcript to gene ID dictionary.
    t2g_dic = {}
    # Gene to biotype dictionary.
    g2bt_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if feature == "gene":
            # Extract gene ID.
            m = re.search('gene_id "(.+?)"', infos)
            assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
            gene_id = m.group(1)
            # Extract gene biotype.
            m = re.search('gene_biotype "(.+?)"', infos)
            assert m, "gene_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
            gene_biotype = m.group(1)
            # Store infos.
            g2bt_dic[gene_id] = gene_biotype
            if all_gbtc_dic is not None:
                if gene_biotype in all_gbtc_dic:
                    all_gbtc_dic[gene_biotype] += 1
                else:
                    all_gbtc_dic[gene_biotype] = 1

        elif feature == "transcript":
            # Extract gene ID.
            m = re.search('gene_id "(.+?)"', infos)
            assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
            gene_id = m.group(1)
            # Extract transcript ID.
            m = re.search('transcript_id "(.+?)"', infos)
            assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
            transcript_id = m.group(1)
            if transcript_id in tr_ids_dic:
                t2g_dic[transcript_id] = gene_id
        else:
            continue
    f.close()
    # Create gene biotype counts dictionary for given transcript IDs.
    gbtc_dic = {}
    for tr_id in t2g_dic:
        gene_id = t2g_dic[tr_id]
        gbt = g2bt_dic[gene_id]
        if gbt in gbtc_dic:
            gbtc_dic[gbt] += 1
        else:
            gbtc_dic[gbt] = 1

    # Check and return to barracks.
    assert gbtc_dic, "gene biotype counts dictionary for given transcript IDs empty"
    return gbtc_dic


################################################################################

def gtf_get_transcript_infos(tr_ids_dic, in_gtf):
    """
    Get transcript infos (transcript biotype, gene ID, gene name, gene biotype)
    from dictionary of transcript IDs (tr_ids_dic).
    Return dictionary with:
    transcript ID -> [transcript biotype, gene ID, gene name, gene biotype]

    >>> tr_ids_dic = {'ENST01': 1}
    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_transcript_infos(tr_ids_dic, in_gtf)
    {'ENST01': ['lncRNA', 'ENSG01', 'ABC1', 'transcribed_unprocessed_pseudogene']}

    """
    # Checks.
    assert tr_ids_dic, "given dictionary tr_ids_dic empty"
    # Transcript to info dictionary.
    t2i_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)

        if not transcript_id in tr_ids_dic:
            continue

        # Extract gene ID.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)
        # Extract gene biotype.
        m = re.search('gene_biotype "(.+?)"', infos)
        assert m, "gene_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_biotype = m.group(1)
        # Extract transcript biotype.
        m = re.search('transcript_biotype "(.+?)"', infos)
        assert m, "transcript_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_biotype = m.group(1)
        m = re.search('gene_name "(.+?)"', infos)
        assert m, "gene_name entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_name = m.group(1)

        t2i_dic[transcript_id] = [tr_biotype, gene_id, gene_name, gene_biotype]

    f.close()

    # Check and return to joint.
    assert t2i_dic, "transcript infos dictionary for given transcript IDs empty"
    return t2i_dic


################################################################################

def gtf_get_gene_infos(gene_ids_dic, in_gtf):
    """
    Get gene infos (gene name, gene biotype) for gene IDs from dictionary
    of gene IDs (gene_ids_dic).
    Return dictionary with:
    gene ID -> [gene name, gene biotype]

    >>> gene_ids_dic = {'ENSG01': 1}
    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_gene_infos(gene_ids_dic, in_gtf)
    {'ENSG01': ['ABC1', 'transcribed_unprocessed_pseudogene']}

    """
    # Checks.
    assert gene_ids_dic, "given dictionary gene_ids_dic empty"
    # Gene to info dictionary.
    g2i_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "gene":
            continue

        # Extract gene ID.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)

        if not gene_id in gene_ids_dic:
            continue

        # Extract gene biotype.
        m = re.search('gene_biotype "(.+?)"', infos)
        assert m, "gene_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_biotype = m.group(1)
        m = re.search('gene_name "(.+?)"', infos)
        assert m, "gene_name entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_name = m.group(1)

        g2i_dic[gene_id] = [gene_name, gene_biotype]

    f.close()

    # Check and return to joint.
    assert g2i_dic, "gene infos dictionary for given gene IDs empty"
    return g2i_dic


################################################################################

def gtf_get_transcript_biotypes(tr_ids_dic, in_gtf):
    """
    Get transcript biotype labels + counts (return label -> count dic)
    for a set of transcript IDs (tr_ids_dic) and a given GTF file (in_gtf).

    >>> tr_ids_dic = {'ENST01': 1, 'ENST02': 1}
    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_transcript_biotypes(tr_ids_dic, in_gtf)
    {'lncRNA': 2}

    """
    # Checks.
    assert tr_ids_dic, "given dictionary tr_ids_dic empty"
    # Biotype to count dic.
    tbt2c_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)

        if not transcript_id in tr_ids_dic:
            continue

        # Extract transcript biotype.
        m = re.search('transcript_biotype "(.+?)"', infos)
        assert m, "transcript_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_biotype = m.group(1)

        # Store biotype info.
        if tr_biotype not in tbt2c_dic:
            tbt2c_dic[tr_biotype] = 1
        else:
            tbt2c_dic[tr_biotype] += 1
    f.close()
    # Check and return to barracks.
    assert tbt2c_dic, "no transcript biotype information read in"
    return tbt2c_dic


################################################################################

def gtf_get_transcript_ids(in_gtf):
    """
    Get transcript IDs from in_gtf GTF file.

    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_transcript_ids(in_gtf)
    {'ENST01': 1, 'ENST02': 1}

    """
    # Transcript IDs dictionary.
    tr_ids_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)

        # Store transcript ID.
        tr_ids_dic[transcript_id] = 1
    f.close()
    # Check and return to barracks.
    assert tr_ids_dic, "no transcript IDs read in"
    return tr_ids_dic


################################################################################

def gtf_get_gene_biotypes(gene_ids_dic, in_gtf,
                          all_gbtc_dic=None):
    """
    Get gene biotype labels + counts (return label -> count dic)
    for a set of gene IDs (gene_ids_dic) and a given GTF file (in_gtf).

    all_gbtc_dic:
    If all_gbtc_dic dictionary is given, fill up this dictionary with
    gene biotype labels and total counts for these in in_gtf GTF file.
    (not just counts for selected genes).

    >>> gene_ids_dic = {'ENSG01': 1, 'ENSG02': 1}
    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_get_gene_biotypes(gene_ids_dic, in_gtf)
    {'transcribed_unprocessed_pseudogene': 2}

    """
    # Checks.
    assert gene_ids_dic, "empty gene IDs dictionary given"
    # Biotype to count dic.
    gbtc_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "gene":
            continue

        # Extract gene ID.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)

        # Extract gene biotype.
        m = re.search('gene_biotype "(.+?)"', infos)
        assert m, "gene_biotype entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_biotype = m.group(1)

        if all_gbtc_dic is not None:
            if gene_biotype not in all_gbtc_dic:
                all_gbtc_dic[gene_biotype] = 1
            else:
                all_gbtc_dic[gene_biotype] += 1

        if not gene_id in gene_ids_dic:
            continue

        # Store biotype info.
        if gene_biotype not in gbtc_dic:
            gbtc_dic[gene_biotype] = 1
        else:
            gbtc_dic[gene_biotype] += 1
    f.close()
    # Check and return to shack.
    assert gbtc_dic, "no gene biotype information read in"
    return gbtc_dic


################################################################################

def gtf_count_isoforms_per_gene(in_gtf,
                                gene_ids_dic=False):
    """
    Count isoforms for each gene and return dictionary with:
    gene_id -> isoform_count.

    gene_ids_dic:
    Gene IDs for which to return isoform counts. Per default, retrun counts
    for all gene IDs.

    >>> in_gtf = "test_data/gene_test_in.gtf"
    >>> gtf_count_isoforms_per_gene(in_gtf)
    {'ENSG01': 1, 'ENSG02': 1}

    """
    # Gene ID to isoform count dic.
    id2c_dic = {}
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        feature = cols[2]
        infos = cols[8]
        if not feature == "transcript":
            continue

        # Extract gene ID.
        m = re.search('gene_id "(.+?)"', infos)
        assert m, "gene_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        gene_id = m.group(1)

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        transcript_id = m.group(1)

        if gene_ids_dic:
            if not gene_id in gene_ids_dic:
                continue

        if gene_id in id2c_dic:
            id2c_dic[gene_id] += 1
        else:
            id2c_dic[gene_id] = 1
    f.close()
    # Check and return to shotgun shack.
    assert id2c_dic, "no gene ID -> isoform count information read in"
    return id2c_dic


################################################################################

def bed_get_transcript_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, out_tra,
                                            stats_dic=None,
                                            id2ucr_dic=False,
                                            codon_annot=False,
                                            border_annot=False,
                                            merge_split_regions=True):
    """
    Get transcript region annotations for genomic BED file in_bed, given
    a GTF file with following annotations:
    five_prime_utr
    CDS
    three_prime_utr
    start_codon (optionally)
    stop_codon (optionally)
    Non of these (default if no overlap)

    tr_ids_dic:
        Transcript IDs dictionary, defines from which transcripts to use
        annotations. Be sure to not use overlapping transcripts, otherwise
        site annotations might be merged ones from different transcripts.
    stats_dic:
        If not None, extract statistics from transcript annotations and store
        in stats_dic.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        Set to define regions for which to extract transcript annotation
        stats, stored in stats_dic.
    codon_annot:
        Add start + stop codon region labels to regions overlapping with
        annotated start or stop codons (from in_gtf). S: start, E: stop
    border_annot:
        Add Transcript and exon border labels (from in_gtf).
        A: transcript start nt, Z: transcript end nt,  B: exon border nts
    merge_split_regions:
        If True, merge labels from IDs with format id1_p1, id1_p2 .. into one.
        Also works for _e1, _e2 .. labels.

    The transcript regions have to be mapped to genome first, where regions
    across exon borders can be split up, resulting in ids: id1_p1, id1_p2 ..
    Annotate these regions, then later merge _p1, _p2 before outputting to
    .tra file.
    in_bed can also be exon regions, with id1_e1, id1_e2 .. where IDs
    are transcript IDs.

    Use following labels:
    five_prime_utr -> F, CDS -> C, three_prime_utr -> T, none -> N
    start_codon -> S, stop_codon -> E

    Output .tra file with format:
    transcript_region_id<tab>FFFFSSSCCCC...

    >>> tr_ids_dic = {"ENST1": 1, "ENST2": 1}
    >>> in_bed = "test_data/test_tr_annot.bed"
    >>> in_gtf = "test_data/test_tr_annot.gtf"
    >>> tmp_tra = "test_data/test_tr_annot.tmp.tra"
    >>> tmp_codon_tra = "test_data/test_tr_annot_codons.tmp.tra"
    >>> exp_tra = "test_data/test_tr_annot.exp.tra"
    >>> exp_codon_tra = "test_data/test_tr_annot.codons.exp.tra"
    >>> bed_get_transcript_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, tmp_tra)
    >>> diff_two_files_identical(tmp_tra, exp_tra)
    True
    >>> bed_get_transcript_annotations_from_gtf(tr_ids_dic, in_bed, in_gtf, tmp_codon_tra, codon_annot=True)
    >>> diff_two_files_identical(tmp_codon_tra, exp_codon_tra)
    True

    """
    # Temp .bed file for storing genomic transcript annotations.
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"
    random_id = uuid.uuid1()
    tmp_out = str(random_id) + ".tmp.out"

    if stats_dic is not None:
        stats_dic["F"] = 0
        stats_dic["C"] = 0
        stats_dic["T"] = 0
        stats_dic["N"] = 0
        stats_dic["total_pos"] = 0
        # Count sites with these symbols.
        if codon_annot:
            stats_dic["S"] = 0
            stats_dic["E"] = 0
        if border_annot:
            stats_dic["A"] = 0
            stats_dic["Z"] = 0
            stats_dic["B"] = 0

    # Feature dictionary.
    feat_dic = {}
    feat_dic["five_prime_utr"] = "F"
    feat_dic["CDS"] = "C"
    feat_dic["three_prime_utr"] = "T"
    # Since CDS feature is separate from codons, make them part of CDS too.
    feat_dic["start_codon"] = "C"
    feat_dic["stop_codon"] = "C"
    # If codon labels should be added.
    if codon_annot:
        feat_dic["start_codon"] = "S"
        feat_dic["stop_codon"] = "E"
    # If border (transcript + exon) labels should be added.
    if border_annot:
        feat_dic["transcript"] = 1
        feat_dic["exon"] = 1

    # Read in in_bed, store start + end coordinates.
    id2s_dic = {}
    id2e_dic = {}
    id2parts_dic = {}
    # Store position labels list for each site in dic.
    id2labels_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            # Check if site ID is split site ID with _e or _p.
            if re.search('.+_[pe]\d+$', site_id):
                m = re.search('(.+)_[pe]\d+$', site_id)
                core_id = m.group(1)
                if core_id in id2parts_dic:
                    id2parts_dic[core_id] += 1
                else:
                    id2parts_dic[core_id] = 1
            else:
                assert site_id not in id2parts_dic, "non-unique site ID \"%s\" in in_bed" %(site_id)
                id2parts_dic[site_id] = 1
            id2s_dic[site_id] = site_s
            id2e_dic[site_id] = site_e
            site_l = site_e - site_s
            id2labels_dic[site_id] = ["N"]*site_l
    f.closed
    assert id2s_dic, "given in_bed \"%s\" empty?" %(in_bed)

    # Get transcript annotations from GTF and output them as BED regions.
    gtf_write_transcript_annotations_to_bed(tr_ids_dic, in_gtf, tmp_bed,
                                            set_feat_dic=feat_dic,
                                            border_annot=border_annot,
                                            codon_annot=codon_annot)

    # Preferred labels, i.e. do not overwrite these if present at position.
    pref_labels_dic = {}
    if codon_annot:
        pref_labels_dic["S"] = 1
        pref_labels_dic["E"] = 1
    if border_annot:
        pref_labels_dic["A"] = 1
        pref_labels_dic["Z"] = 1
        pref_labels_dic["B"] = 1

    # Run overlap calculation to get exon overlapping regions.
    intersect_params = "-s -wb"
    intersect_bed_files(in_bed, tmp_bed, intersect_params, tmp_out)

    """
    Example output:
    $ intersectBed -a sites.bed -b annot.bed -s -wb
    chr1	1000	1020	site1	0	+	chr1	980	1020	F	0	+
    chr1	1020	1023	site1	0	+	chr1	1020	1023	S	0	+
    chr1	1020	1050	site1	0	+	chr1	1020	1500	C	0	+
    """

    with open(tmp_out) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            s = int(cols[1]) + 1 # Make one-based.
            e = int(cols[2])
            site_id = cols[3]
            site_s = id2s_dic[site_id] + 1 # Make one-based.
            site_e = id2e_dic[site_id]
            site_pol = cols[5]
            label = cols[9]
            # + case.
            if site_pol == "+":
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = i - site_s
                        if id2labels_dic[site_id][li] not in pref_labels_dic:
                            id2labels_dic[site_id][li] = label
            else:
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = site_e - i
                        if id2labels_dic[site_id][li] not in pref_labels_dic:
                            id2labels_dic[site_id][li] = label
    f.closed

    # Output transcript region annotations to .tra file.
    OUTLAB = open(out_tra,"w")
    if merge_split_regions:
        # Merge split regions.
        for site_id in id2parts_dic:
            # Parts count.
            part_c = id2parts_dic[site_id]
            label_str = ""
            # For one-part regions, just output.
            if part_c == 1:
                # List to string.
                label_str = "".join(id2labels_dic[site_id])
            else:
                # For split regions, assemble parts and output.
                new_label_list = []
                for i in range(part_c):
                    i += 1
                    part_id = site_id + "_p%i" %(i)
                    if part_id not in id2labels_dic:
                        # Try exon ID.
                        part_id = site_id + "_e%i" %(i)
                        if part_id not in id2labels_dic:
                            assert False, "exon or part ID for site ID \"%i\" (part# %i) missing in id2labels_dic" %(site_id, i)
                    new_label_list += id2labels_dic[part_id]
                assert new_label_list, "merging split region label lists failed"
                # List to string.
                label_str = "".join(new_label_list)
            OUTLAB.write("%s\t%s\n" %(site_id, label_str))
    else:
        # Do not merge split regions, just output labels for each site.
        for site_id in id2labels_dic:
            # List to string.
            label_str = "".join(id2labels_dic[site_id])
            OUTLAB.write("%s\t%s\n" %(site_id, label_str))
    OUTLAB.close()

    if stats_dic:
        with open(out_tra) as f:
            for line in f:
                row = line.strip()
                cols = line.strip().split("\t")
                reg_id = cols[0]
                label_str = cols[1]
                if id2ucr_dic:
                    # If uppercase part only, prune label_str.
                    uc_s = id2ucr_dic[reg_id][0]
                    uc_e = id2ucr_dic[reg_id][1]
                    label_str = label_str[uc_s-1:uc_e]
                stats_dic["total_pos"] += len(label_str)
                # Count occurences (+1 for each site with label) for these labels.
                occ_labels = ["S", "E", "A", "Z", "B"]
                for ocl in occ_labels:
                    if re.search("%s" %(ocl), label_str):
                        stats_dic[ocl] += 1
                for i in range(len(label_str)):
                    l = label_str[i]
                    if l not in occ_labels:
                        stats_dic[l] += 1
        f.closed

    # Take out the trash.
    litter_street = True
    if litter_street:
        if os.path.exists(tmp_bed):
            os.remove(tmp_bed)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


################################################################################

def get_transcript_border_annotations(tr_ids_dic, in_gtf, out_bed,
                                      append=False):
    """

    Get transcript border annotations and write border positions to
    out_bed BED. Additional transcript annotations include:
    A : transcript start
    Z : transcript end
    B : Exon border position

    append:
    If True, append content to out_bed, instead of overwriting any
    existing out_bed.

    >>> tr_ids_dic = {'ENST1': 1, 'ENST2': 1, 'ENST3': 1}
    >>> in_gtf = "test_data/test_border_annot.gtf"
    >>> out_exp_bed = "test_data/test_border_annot.exp.bed"
    >>> out_tmp_bed = "test_data/test_border_annot.tmp.bed"
    >>> get_transcript_border_annotations(tr_ids_dic, in_gtf, out_tmp_bed)
    >>> diff_two_files_identical(out_tmp_bed, out_exp_bed)
    True

    """
    # Checker.
    assert tr_ids_dic, "given dictionary tr_ids_dic empty"

    # Get exon counts for transcripts from GTF.
    tr_exc_dic = gtf_extract_exon_numbers(in_gtf, tr_ids_dic=tr_ids_dic)

    # Features to look at.
    feat_dic = {'transcript': 1, 'exon': 1}

    # Extract transcript border annotations from in_gtf.
    if append:
        TBAOUT = open(out_bed, "a")
    else:
        TBAOUT = open(out_bed, "w")

    # Count processed features.
    c_out = 0

    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]

        # Feature check.
        if feature not in feat_dic:
            continue

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_id = m.group(1)

        # Only features from selected transcripts.
        if not tr_id in tr_ids_dic:
            continue

        # Get exon count for transcript.
        assert tr_id in tr_exc_dic, "transcript ID \"%s\" not in tr_exc_dic" %(tr_id)
        exc = tr_exc_dic[tr_id]

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        # Get transcript start and end positions.
        if feature == "transcript":
            # Start position.
            s_start = feat_s
            e_start = feat_s + 1
            # End position.
            s_end = feat_e - 1
            e_end = feat_e
            if feat_pol == "-":
                s_start = feat_e - 1
                e_start = feat_e
                s_end = feat_s
                e_end = feat_s + 1
            # Output positions to BED.
            TBAOUT.write("%s\t%i\t%i\tA\t0\t%s\n" %(chr_id, s_start, e_start, feat_pol))
            TBAOUT.write("%s\t%i\t%i\tZ\t0\t%s\n" %(chr_id, s_end, e_end, feat_pol))

        # Get exon border positions.
        if feature == "exon":
            # For single exon transcripts, no exon borders between A + Z.
            if exc == 1:
                continue
            m = re.search('exon_number "(\d+?)"', infos)
            assert m, "exon_number entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
            exon_nr = int(m.group(1))
            # Get infos based on exon number.
            if exon_nr == 1:
                # First exon.
                s_end = feat_e - 1
                e_end = feat_e
                if feat_pol == "-":
                    s_end = feat_s
                    e_end = feat_s + 1
                TBAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_end, e_end, feat_pol))
            elif exc == exon_nr:
                # Last exon.
                s_start = feat_s
                e_start = feat_s + 1
                if feat_pol == "-":
                    s_start = feat_e - 1
                    e_start = feat_e
                TBAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_start, e_start, feat_pol))
            else:
                # In-between exon.
                s_start = feat_s
                e_start = feat_s + 1
                s_end = feat_e - 1
                e_end = feat_e
                if feat_pol == "-":
                    s_start = feat_e - 1
                    e_start = feat_e
                    s_end = feat_s
                    e_end = feat_s + 1
                TBAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_start, e_start, feat_pol))
                TBAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_end, e_end, feat_pol))

        # Output labeled region.
        c_out += 1

    f.close()
    TBAOUT.close()
    assert c_out, "no transcript or exon features found or output"


################################################################################

def gtf_write_transcript_annotations_to_bed(tr_ids_dic, in_gtf, out_bed,
                                            set_feat_dic=False,
                                            border_annot=False,
                                            codon_annot=False):

    """
    Extract transcript region annotations from in_gtf GTF file and store
    annotations as BED regions in out_bed.


    Get transcript region annotations for genomic BED file in_bed, given
    a GTF file with following annotations:
    five_prime_utr
    CDS
    three_prime_utr
    start_codon (optionally)
    stop_codon (optionally)

    Use following labels:
    five_prime_utr -> F, CDS -> C, three_prime_utr -> T
    start_codon -> S, stop_codon -> E
    Transcript start -> A
    Transcript end -> Z
    Exon border -> B

    tr_ids_dic:
        Transcript IDs dictionary, defines from which transcripts to use
        annotations. Be sure to not use overlapping transcripts, otherwise
        site annotations might be merged ones from different transcripts.
    set_feat_dic:
        Overwrite features dictionary defined inside function, using the
        supplied one set_feat_dic.
    border_annot:
        Also output transcript + exon border positions to out_bed.
        Exon border labels (B) are added both to exon start + end, unless
        it is the first or last or the only exon (3 distinctions).
    codon_annot:
        Also output start_codon and stop_codon features.

    out_bed example output (notice labels as column 4 IDs):
    chr1	980	1020	F	0	+
    chr1	1020	1023	S	0	+
    chr1	1020	1500	C	0	+
    ...

    >>> in_gtf = "test_data/test_tr_annot.gtf"
    >>> tr_ids_dic = {"ENST1": 1, "ENST2": 1}
    >>> out_bed = "test_data/test_tr_annot_gtf.tmp.bed"
    >>> exp_bed = "test_data/test_tr_annot_gtf.exp.bed"
    >>> gtf_write_transcript_annotations_to_bed(tr_ids_dic, in_gtf, out_bed,codon_annot=True, border_annot=True)
    >>> diff_two_files_identical(out_bed, exp_bed)
    True

    """

    # Feature dictionary.
    feat_dic = {}
    feat_dic["five_prime_utr"] = "F"
    feat_dic["CDS"] = "C"
    feat_dic["three_prime_utr"] = "T"
    # Since CDS feature is separate from codons, make them part of CDS too.
    feat_dic["start_codon"] = "C"
    feat_dic["stop_codon"] = "C"
    # If start / stop codon annotations should be added too.
    if codon_annot:
        feat_dic["start_codon"] = "S"
        feat_dic["stop_codon"] = "E"
    if border_annot:
        feat_dic["transcript"] = 1
        feat_dic["exon"] = 1
    # Overwrite feat_dic if given to function.
    if set_feat_dic:
        feat_dic = set_feat_dic

    # If border (exon + transcript) annotations should be added too.
    tr_exc_dic = {}
    if border_annot:
        # Get exon counts for transcripts from GTF.
        tr_exc_dic = gtf_extract_exon_numbers(in_gtf, tr_ids_dic=tr_ids_dic)

    # Extract transcript annotations from in_gtf.
    TRAOUT = open(out_bed, "w")
    c_out = 0
    # Open GTF either as .gz or as text file.
    if re.search(".+\.gz$", in_gtf):
        f = gzip.open(in_gtf, 'rt')
    else:
        f = open(in_gtf, "r")
    for line in f:
        # Skip header.
        if re.search("^#", line):
            continue
        cols = line.strip().split("\t")
        chr_id = cols[0]
        feature = cols[2]
        feat_s = int(cols[3])
        feat_e = int(cols[4])
        feat_pol = cols[6]
        infos = cols[8]
        # Extract only features in feat_dic.
        if feature not in feat_dic:
            continue
        label = feat_dic[feature]

        # Extract transcript ID.
        m = re.search('transcript_id "(.+?)"', infos)
        assert m, "transcript_id entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
        tr_id = m.group(1)

        # Only features from selected transcripts.
        if tr_id not in tr_ids_dic:
            continue

        # Restrict to standard chromosomes.
        new_chr_id = check_convert_chr_id(chr_id)
        if not new_chr_id:
            continue
        else:
            chr_id = new_chr_id

        # Make start coordinate 0-base (BED standard).
        feat_s = feat_s - 1

        if feature == "transcript" or feature == "exon":
            if not border_annot:
                continue
            # Get exon count for transcript.
            assert tr_id in tr_exc_dic, "transcript ID \"%s\" not in tr_exc_dic" %(tr_id)
            exc = tr_exc_dic[tr_id]

            # Get transcript start and end positions.
            if feature == "transcript":
                # Start position.
                s_start = feat_s
                e_start = feat_s + 1
                # End position.
                s_end = feat_e - 1
                e_end = feat_e
                if feat_pol == "-":
                    s_start = feat_e - 1
                    e_start = feat_e
                    s_end = feat_s
                    e_end = feat_s + 1
                # Output positions to BED.
                TRAOUT.write("%s\t%i\t%i\tA\t0\t%s\n" %(chr_id, s_start, e_start, feat_pol))
                TRAOUT.write("%s\t%i\t%i\tZ\t0\t%s\n" %(chr_id, s_end, e_end, feat_pol))

            # Get exon border positions.
            if feature == "exon":
                # For single exon transcripts, no exon borders between A + Z.
                if exc == 1:
                    continue
                m = re.search('exon_number "(\d+?)"', infos)
                assert m, "exon_number entry missing in GTF file \"%s\", line \"%s\"" %(in_gtf, line)
                exon_nr = int(m.group(1))
                # Get infos based on exon number.
                if exon_nr == 1:
                    # First exon.
                    s_end = feat_e - 1
                    e_end = feat_e
                    if feat_pol == "-":
                        s_end = feat_s
                        e_end = feat_s + 1
                    TRAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_end, e_end, feat_pol))
                elif exc == exon_nr:
                    # Last exon.
                    s_start = feat_s
                    e_start = feat_s + 1
                    if feat_pol == "-":
                        s_start = feat_e - 1
                        e_start = feat_e
                    TRAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_start, e_start, feat_pol))
                else:
                    # In-between exon.
                    s_start = feat_s
                    e_start = feat_s + 1
                    s_end = feat_e - 1
                    e_end = feat_e
                    if feat_pol == "-":
                        s_start = feat_e - 1
                        e_start = feat_e
                        s_end = feat_s
                        e_end = feat_s + 1
                    TRAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_start, e_start, feat_pol))
                    TRAOUT.write("%s\t%i\t%i\tB\t0\t%s\n" %(chr_id, s_end, e_end, feat_pol))
            c_out += 1
            # Skip rest.
            continue

        c_out += 1
        # Output labeled region.
        TRAOUT.write("%s\t%i\t%i\t%s\t0\t%s\n" %(chr_id, feat_s, feat_e, label, feat_pol))
    f.close()
    TRAOUT.close()
    assert c_out, "no transcript annotation regions output"


################################################################################

def get_seq_len_list_from_dic(seqs_dic):
    """
    Given a dictinary with sequences, return a list of sequence lengths.

    >>> seqs_dic = {'seq1':'ACGT', 'seq2': 'ACGTACGT'}
    >>> get_seq_len_list_from_dic(seqs_dic)
    [4, 8]

    """
    assert seqs_dic, "sequences dictionary seqs_dic empty"
    len_list = []
    for seq_id in seqs_dic:
        len_list.append(len(seqs_dic[seq_id]))
    assert len_list, "sequence lengths list len_list empty"
    return len_list


################################################################################

def calc_seq_entropy(seq_l, ntc_dic):
    """
    Given a dictionary of nucleotide counts for a sequence ntc_dic and
    the length of the sequence seq_l, compute the Shannon entropy of
    the sequence.

    Formula (see CE formula) taken from:
    https://www.ncbi.nlm.nih.gov/pubmed/15215465

    >>> seq_l = 8
    >>> ntc_dic = {'A': 8, 'C': 0, 'G': 0, 'U': 0}
    >>> calc_seq_entropy(seq_l, ntc_dic)
    0
    >>> ntc_dic = {'A': 4, 'C': 4, 'G': 0, 'U': 0}
    >>> calc_seq_entropy(seq_l, ntc_dic)
    0.5
    >>> ntc_dic = {'A': 2, 'C': 2, 'G': 2, 'U': 2}
    >>> calc_seq_entropy(seq_l, ntc_dic)
    1.0

    """
    # For DNA or RNA, k = 4.
    k = 4
    # Shannon entropy.
    ce = 0
    for nt in ntc_dic:
        c = ntc_dic[nt]
        if c != 0:
            ce += (c/seq_l) * log((c/seq_l), k)
    if ce == 0:
        return 0
    else:
        return -1*ce


################################################################################

def dic_sum_up_lengths(in_dic):
    """
    Given a dictionary with strings or numbers, sum up the numbers /
    string lengths and return the total length.
    Currently works for integer numbers and strings.

    >>> in_dic = {'e1': 5, 'e2': 10}
    >>> dic_sum_up_lengths(in_dic)
    15
    >>> in_dic = {'e1': 'ACGT', 'e2': 'ACGTACGT'}
    >>> dic_sum_up_lengths(in_dic)
    12

    """
    assert in_dic, "given dictionary in_dic empty"
    sum = 0
    for e in in_dic:
        v = in_dic[e]
        if isinstance(v, str):
            sum += len(v)
        elif isinstance(v, int):
            sum += v
        else:
            assert False, "non-string or non-integer dictionary value given"
    return sum


################################################################################

def seqs_dic_count_nt_freqs(seqs_dic,
                            rna=False,
                            convert_to_uc=False,
                            count_dic=False):
    """
    Given a dictionary with sequences seqs_dic, count how many times each
    nucleotide is found in all sequences (== get nt frequencies).
    Return nucleotide frequencies count dictionary.

    By default, a DNA dictionary (A,C,G,T) is used, counting only these
    characters (note they are uppercase!).

    rna:
    Instead of DNA dictionary, use RNA dictionary (A,C,G,U) for counting.

    convert_to_uc:
    Convert sequences to uppercase before counting.

    count_dic:
    Supply a custom dictionary for counting only characters in
    this dictionary + adding counts to this dictionary.

    >>> seqs_dic = {'s1': 'AAAA', 's2': 'CCCGGT'}
    >>> seqs_dic_count_nt_freqs(seqs_dic)
    {'A': 4, 'C': 3, 'G': 2, 'T': 1}
    >>> seqs_dic_count_nt_freqs(seqs_dic, rna=True)
    {'A': 4, 'C': 3, 'G': 2, 'U': 0}

    """
    assert seqs_dic, "given dictionary seqs_dic empty"
    if not count_dic:
        count_dic = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        if rna:
            count_dic = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        if convert_to_uc:
            seq = seq.upper()
        seq_count_nt_freqs(seq, rna=rna, count_dic=count_dic)
    return count_dic

################################################################################

def seqs_dic_count_chars(seqs_dic):
    """
    Given a dictionary with sequences, count how many times each character
    appears.

    >>> seqs_dic = {'s1': 'ABCC', 's2': 'ABCD'}
    >>> seqs_dic_count_chars(seqs_dic)
    {'A': 2, 'B': 2, 'C': 3, 'D': 1}

    """
    assert seqs_dic, "given seqs_dic empty"
    cc_dic = {}
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        for c in seq:
            if c in cc_dic:
                cc_dic[c] += 1
            else:
                cc_dic[c] = 1
    assert cc_dic, "cc_dic empty"
    return cc_dic

################################################################################

def seq_count_nt_freqs(seq,
                       rna=False,
                       count_dic=False):
    """
    Count nucleotide (character) frequencies in given sequence seq.
    Return count_dic with frequencies.
    If count_dic is given, add count to count_dic.

    rna:
    Instead of DNA dictionary, use RNA dictionary (A,C,G,U) for counting.

    count_dic:
    Supply a custom dictionary for counting only characters in
    this dictionary + adding counts to this dictionary.

    >>> seq = 'AAAACCCGGT'
    >>> seq_count_nt_freqs(seq)
    {'A': 4, 'C': 3, 'G': 2, 'T': 1}
    >>> seq = 'acgtacgt'
    >>> seq_count_nt_freqs(seq)
    {'A': 0, 'C': 0, 'G': 0, 'T': 0}

    """

    assert seq, "given sequence string seq empty"
    if not count_dic:
        count_dic = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        if rna:
            count_dic = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
    # Conver to list.
    seq_list = list(seq)
    for nt in seq_list:
        if nt in count_dic:
            count_dic[nt] += 1
    return count_dic


################################################################################

def fasta_get_repeat_region_annotations(seqs_dic, out_rra,
                                        stats_dic=None,
                                        id2ucr_dic=False):
    """
    Get repeat region annotations for genomic and transcript regions,
    given a dictionary of sequences with lower- and uppercase sequences.
    When extracting sequences from .2bit using twoBitToFa, not enabling
    -noMask results in mixed lower- and uppercase regions returned.
    Lowercase: region annotated as repeat by RepeatMasker and
    Tandem Repeats Finder (with period of 12 or less).
    Uppercase: non-repeat region.
    Thus, add position-wise R (repeat) or N (no repeat) annotations
    to label repeat and non-repeat region sequences and output to
    rra_out.

    stats_dic:
        If not None, extract statistics from repeat annotations and store
        in stats_dic.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        Set to define regions for which to extract repeat annotation
        stats, stored in stats_dic.

    Output format example:
    seq1<tab>NNRRRRNN
    seq2<tab>NNNNNNN
    seq3<tab>RRRRRRRRR

    >>> seqs_dic = {'seq1': 'ACacgtAC', 'seq2': 'ACGUACG', 'seq3': 'acguacguu'}
    >>> out_exp_rra = "test_data/test8.exp.rra"
    >>> out_tmp_rra = "test_data/test8.tmp.rra"
    >>> fasta_get_repeat_region_annotations(seqs_dic, out_tmp_rra)
    >>> diff_two_files_identical(out_tmp_rra, out_exp_rra)
    True

    """
    assert seqs_dic, "given dictionary seqs_dic empty"

    if stats_dic is not None:
        stats_dic["R"] = 0
        stats_dic["N"] = 0
        stats_dic["total_pos"] = 0

    OUTRRA = open(out_rra,"w")
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        rra_str = ""
        seq_list = list(seq)
        for nt in seq_list:
            if nt.islower():
                rra_str += "R"
            else:
                rra_str += "N"
        OUTRRA.write("%s\t%s\n" %(seq_id, rra_str))
        if stats_dic:
            if id2ucr_dic:
                # If uppercase part only, prune rra_str.
                uc_s = id2ucr_dic[seq_id][0]
                uc_e = id2ucr_dic[seq_id][1]
                rra_str = rra_str[uc_s-1:uc_e]
            for l in rra_str:
                stats_dic["total_pos"] += 1
                stats_dic[l] += 1

    OUTRRA.close()


################################################################################

def seqs_dic_calc_entropies(seqs_dic,
                            rna=True,
                            uc_part_only=True):
    """
    Given a dictionary of sequences, calculate entropies for each sequence
    and return list of entropy values.

    seqs_dic:
    Dictionary with sequences.

    rna:
    Use RNA alphabet for counting (uppercase chars only)

    uc_part_only:
    Calculate entropy only for uppercase part of sequence

    >>> seqs_dic = {'seq1': 'AAAAAAAA', 'seq2': 'AAAACCCC', 'seq3': 'AACCGGUU'}
    >>> seqs_dic_calc_entropies(seqs_dic)
    [0, 0.5, 1.0]

    """
    assert seqs_dic, "given dictionary seqs_dic empty"
    entr_list = []
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        seq_l = len(seq)
        new_seq = seq
        # If only uppercase part should be used.
        if uc_part_only:
            m = re.search("[acgtu]*([ACGTU]+)[acgtu]*", seq)
            assert m, "uppercase sequence part extraction failed for sequence ID \"%s\" and sequence \"%s\"" %(seq_id, seq)
            new_seq = m.group(1)
            seq_l = len(new_seq)
        # Make uppercase (otherwise seq_l not correct).
        new_seq = new_seq.upper()
        # Get nt count dic.
        count_dic = seq_count_nt_freqs(new_seq, rna=rna)
        # Calculate sequence entropy.
        seq_entr = calc_seq_entropy(seq_l, count_dic)
        #if seq_entr > 0.5:
        #    print("Entropy: %.2f" %(seq_entr))
        #    print("%s: %s" %(seq_id, seq))
        entr_list.append(seq_entr)
    return entr_list


################################################################################

def ntc_dic_to_ratio_dic(ntc_dic,
                         perc=False):
    """
    Given a dictionary of nucleotide counts, return dictionary of nucleotide
    ratios (count / total nucleotide number).

    perc:
    If True, make percentages out of ratios (*100).

    >>> ntc_dic = {'A': 5, 'C': 2, 'G': 2, 'T': 1}
    >>> ntc_dic_to_ratio_dic(ntc_dic)
    {'A': 0.5, 'C': 0.2, 'G': 0.2, 'T': 0.1}

    """
    assert ntc_dic, "given dictionary ntc_dic empty"
    # Get total number.
    total_n = 0
    for nt in ntc_dic:
        total_n += ntc_dic[nt]
    ntr_dic = {}
    for nt in ntc_dic:
        ntc = ntc_dic[nt]
        ntr = ntc / total_n
        if perc:
            ntr = ntr*100
        ntr_dic[nt] = ntr
    return ntr_dic


################################################################################

def create_set_lengths_box_plot(pos_len_list, neg_len_list, out_plot,
                                disable_title=False,
                                theme=2,
                                scale_zero_max=False):
    """
    Create a box plot, to compare sequence lengths found in positive
    and negative set.
    Given two lists of lengths (positives, negatives), create a dataframe
    using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    """
    # Checker.
    assert pos_len_list, "given list pos_len_list empty"
    assert neg_len_list, "given list neg_len_list empty"
    if scale_zero_max:
        # Get maximum length for scaling.
        pos_max = max(pos_len_list)
        neg_max = max(neg_len_list)
        max_l = pos_max
        if pos_max < neg_max:
            max_l = neg_max
        # Get next highest number % 10.
        max_y = max_l
        while max_y % 10:
             max_y += 1
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'length': []}
    pos_c = len(pos_len_list)
    neg_c = len(neg_len_list)
    data['set'] += pos_c*[pos_label] + neg_c*[neg_label]
    data['length'] += pos_len_list + neg_len_list
    df = pd.DataFrame (data, columns = ['set','length'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="length", data=df, palette=['cyan','cyan'],
                    width=0.7, linewidth = 1.5, boxprops=dict(alpha=.7))
        # Modify.
        ax.set_ylabel("Length (nt)",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        if scale_zero_max:
            ax.set_ylim([0,max_y])
        if not disable_title:
            ax.axes.set_title("Site length distribution", fontsize=20)
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight')

    elif theme == 2:

        """
        Midnight Blue theme.

        ffffff : white
        190250 : midnight blue
        fcc826 : yellowish
        fd3b9d : pinkish
        2f19f3 : dash blue

        """
        # Theme colors.
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"

        # Custom flier (outlier) edge and face colors.
        flierprops = dict(markersize=5, markerfacecolor=box_color, markeredgecolor=text_color)
        boxprops = dict(color=box_color, edgecolor=text_color)
        medianprops = dict(color=text_color)
        meanprops = dict(color=text_color)
        whiskerprops = dict(color=text_color)
        capprops = dict(color=text_color)

        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="length", data=df,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    meanprops=meanprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    width=0.7, linewidth = 1.5)

        # Modify.
        ax.set_ylabel("Length (nt)",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        if scale_zero_max:
            ax.set_ylim([0,max_y])
        if not disable_title:
            ax.axes.set_title("Site length distribution", fontsize=20)
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight', transparent=True)


################################################################################

def create_entropy_box_plot(pos_entr_list, neg_entr_list, out_plot,
                            theme=2,
                            disable_title=False):
    """
    Create a box plot, to compare sequence entropies found in positive
    and negative set.
    Given two lists of entropies (positives, negatives), create a dataframe
    using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    theme:
    Choose between two themes for plotting, 1: default, 2: midnight blue

    fig, ax = plt.subplots(figsize=(15,8))
    sns.set(style="whitegrid", font_scale=1)
fig, ax = plt.subplots()
fig.savefig(

    # Make plot.
    sns.set(style="whitegrid")
    ent_plot = sns.boxplot(x="set", y="entropy", data=df, palette=['lightgrey','lightgrey'],
                           width=0.7, linewidth = 1.5, boxprops=dict(alpha=.7))
    # Modify.
    ent_plot.set_ylabel("Sequence complexity",fontsize=18)
    ent_plot.tick_params(axis='x', labelsize=18)
    ent_plot.tick_params(axis='y', labelsize=12)
    ent_plot.axes.set_title("Sequence complexity distribution", fontsize=20)
    ent_plot.set(xlabel=None)
    # Store plot.
    ent_plot.figure.savefig(out_plot, dpi=125, bbox_inches='tight')

sns.set(style="darkgrid")
seaborn.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})

seaborn.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})

fig, ax = plt.subplots()

"figure.facecolor": "white",
"axes.labelcolor": dark_gray,
"text.color": dark_gray,
                "axes.facecolor": "#EAEAF2",
                "axes.edgecolor": "white",
                "grid.color": "white",

    Midnight blue theme:
    fig.savefig(out_plot, dpi=125, bbox_inches='tight', transparent=True)

    Midnight Blue theme:
    ====================

    HTML Hex colors:
    ffffff : white
    190250 : midnight blue
    fcc826 : yellowish
    fd3b9d : pinkish
    2f19f3 : dash blue

    bgcolor="#190250"
    text="#ffffff"
    link="#fd3b9d"
    vlink="#fd3b9d"
    alink="#fd3b9d"




"text.color": "white",
    """
    # Checker.
    assert pos_entr_list, "given list pos_entr_list empty"
    assert neg_entr_list, "given list neg_entr_list empty"
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'entropy': []}
    pos_c = len(pos_entr_list)
    neg_c = len(neg_entr_list)
    data['set'] += pos_c*[pos_label] + neg_c*[neg_label]
    data['entropy'] += pos_entr_list + neg_entr_list
    df = pd.DataFrame (data, columns = ['set','entropy'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="entropy", data=df, palette=['cyan','cyan'],
                    width=0.7, linewidth = 1.5, boxprops=dict(alpha=.7))
        # Modify.
        ax.set_ylabel("Sequence complexity",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        if not disable_title:
            ax.axes.set_title("Sequence complexity distribution", fontsize=20)
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight')

    elif theme == 2:

        """
        Midnight Blue theme:
        ====================

        HTML Hex colors:
        ffffff : white
        190250 : midnight blue
        fcc826 : yellowish
        fd3b9d : pinkish
        2f19f3 : dash blue

        bgcolor="#190250"
        text="#ffffff"
        link="#fd3b9d"
        vlink="#fd3b9d"
        alink="#fd3b9d"

        Editing matplotlib boxplot element props:
        (from matplotlib.axes.Axes.boxplot)
        boxprops
        whiskerprops
        flierprops
        medianprops
        meanprops

        """
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Custom flier (outlier) edge and face colors.
        flierprops = dict(markersize=5, markerfacecolor=box_color, markeredgecolor=text_color)
        boxprops = dict(color=box_color, edgecolor=text_color)
        medianprops = dict(color=text_color)
        meanprops = dict(color=text_color)
        whiskerprops = dict(color=text_color)
        capprops = dict(color=text_color)
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="entropy", data=df,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    meanprops=meanprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    width=0.7, linewidth = 1.5)
        # Modify.
        ax.set_ylabel("Sequence complexity",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        # Boxplot whisker colors.
        #plt.setp(ax.lines, color=text_color)
        # Boxplot box edge + face colors.
        #plt.setp(ax.artists,  edgecolor=text_color, facecolor=box_color)
        if not disable_title:
            ax.axes.set_title("Sequence complexity distribution", fontsize=20)
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight', transparent=True)


################################################################################

def create_dint_ratios_grouped_bar_plot(pos_dintr_dic, neg_dintr_dic, out_plot,
                                        disable_title=False,
                                        theme=1):
    """
    Create a grouped bar plot, showing the di-nucleotide ratios (16 classes)
    in the positive and negative set.
    Input ratio dictionaries for positives and negatives, with key being
    di-nucleotide and value the ratio.
    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    MV colors:
    #69e9f6, #f154b2

    """

    # Checker.
    assert pos_dintr_dic, "given dictionary pos_dintr_dic empty"
    assert neg_dintr_dic, "given dictionary neg_dintr_dic empty"
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'dint': [], 'perc': []}
    for dint in pos_dintr_dic:
        data['set'].append(pos_label)
        data['dint'].append(dint)
        data['perc'].append(pos_dintr_dic[dint])
    for dint in neg_dintr_dic:
        data['set'].append(neg_label)
        data['dint'].append(dint)
        data['perc'].append(neg_dintr_dic[dint])
    df = pd.DataFrame (data, columns = ['set','dint', 'perc'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        g = sns.catplot(x="dint", y="perc", hue="set", data=df, height=6,
                        kind="bar", palette=["#69e9f6", "#f154b2"],
                        edgecolor="lightgrey",
                        legend=False)
        g.fig.set_figwidth(16)
        g.fig.set_figheight(4)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_ylabel("Percentage (%)",fontsize=20)
        ax[0,0].set(xlabel=None)
        ax[0,0].tick_params(axis='x', labelsize=20)
        ax[0,0].tick_params(axis='y', labelsize=16)
        if not disable_title:
            ax[0,0].axes.set_title("Di-nucleotide distribution", fontsize=22)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=16)
        g.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})

        g = sns.catplot(x="dint", y="perc", hue="set", data=df, height=6,
                        kind="bar", palette=["blue", "darkblue"],
                        edgecolor="#fcc826",
                        legend=False)
        g.fig.set_figwidth(16)
        g.fig.set_figheight(4)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_ylabel("Percentage (%)",fontsize=20)
        ax[0,0].set(xlabel=None)
        ax[0,0].tick_params(axis='x', labelsize=20)
        ax[0,0].tick_params(axis='y', labelsize=16)
        if not disable_title:
            ax[0,0].axes.set_title("Di-nucleotide distribution", fontsize=22)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=16, framealpha=0)
        g.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def create_str_elem_grouped_bar_plot(pos_str_stats_dic, neg_str_stats_dic, out_plot,
                                     disable_title=False,
                                     theme=1):
    """
    Create a grouped bar plot, showing average probabilities of secondary
    structure elements (U, E, H, I, M, S) in the positive and negative set.
    pos_str_stats_dic and neg_str_stats_dic contain the statistics for
    the the positive and negative set (mean + stdev values).
    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    Stats dictionary content.
    stats_dic["U"] = [pu_mean, pu_stdev]
    stats_dic["S"] = [ps_mean, ps_stdev]
    stats_dic["E"] = [pe_mean, pe_stdev]
    stats_dic["H"] = [ph_mean, ph_stdev]
    stats_dic["I"] = [pi_mean, pi_stdev]
    stats_dic["M"] = [pm_mean, pm_stdev]

    """
    # Checker.
    assert pos_str_stats_dic, "given dictionary pos_str_stats_dic empty"
    assert neg_str_stats_dic, "given dictionary neg_str_stats_dic empty"
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'elem': [], 'mean_p': [], 'stdev_p': []}
    for el in pos_str_stats_dic:
        if not re.search("^[U|S|E|H|I|M]$", el):
            continue
        data['set'].append(pos_label)
        data['elem'].append(el)
        data['mean_p'].append(pos_str_stats_dic[el][0])
        data['stdev_p'].append(pos_str_stats_dic[el][1])
    for el in neg_str_stats_dic:
        if not re.search("^[U|S|E|H|I|M]$", el):
            continue
        data['set'].append(neg_label)
        data['elem'].append(el)
        data['mean_p'].append(neg_str_stats_dic[el][0])
        data['stdev_p'].append(neg_str_stats_dic[el][1])
    df = pd.DataFrame (data, columns = ['set','elem', 'mean_p', 'stdev_p'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        g = sns.catplot(x="elem", y="mean_p", hue="set", data=df, height=6,
                        kind="bar", palette=["#69e9f6", "#f154b2"],
                        edgecolor="lightgrey",
                        legend=False)
        g.fig.set_figwidth(10)
        g.fig.set_figheight(4)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_ylabel("Mean probability",fontsize=22)
        ax[0,0].set(xlabel=None)
        ax[0,0].tick_params(axis='x', labelsize=22)
        ax[0,0].tick_params(axis='y', labelsize=17)
        if not disable_title:
            ax[0,0].axes.set_title("Structural elements distribution", fontsize=24)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=17)
        g.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        g = sns.catplot(x="elem", y="mean_p", hue="set", data=df, height=6,
                        kind="bar", palette=["blue", "darkblue"],
                        edgecolor="#fcc826",
                        legend=False)
        g.fig.set_figwidth(10)
        g.fig.set_figheight(4)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_ylabel("Mean probability",fontsize=22)
        ax[0,0].set(xlabel=None)
        ax[0,0].tick_params(axis='x', labelsize=22)
        ax[0,0].tick_params(axis='y', labelsize=17)
        if not disable_title:
            ax[0,0].axes.set_title("Structural elements distribution", fontsize=24)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=17, framealpha=0)
        g.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def create_eval_model_comp_scatter_plot(model1_scores, model2_scores, out_plot,
                                        x_label="Score model 1",
                                        y_label="Score model 2",
                                        theme=1):
    """
    Create graphprot2 eval scatter plot, to compare scores produced by
    two models on same dataset. Also calculates and plots R2 (coefficient
    of determination) value for two datasets.

    """
    assert model1_scores, "model1_scores empty"
    assert model2_scores, "model2_scores empty"
    set1_c = len(model1_scores)
    set2_c = len(model2_scores)
    assert set1_c == set2_c, "differing set sizes for set1_c and set2_c (%i != %i)" %(set1_c, set2_c)
    data = {'m1_score': [], 'm2_score': []}
    for i,sc in enumerate(model1_scores):
        data['m1_score'].append(sc)
        data['m2_score'].append(model2_scores[i])
    df = pd.DataFrame (data, columns = ['m1_score','m2_score'])

    # Calculate R2.
    correlation_matrix = np.corrcoef(model1_scores, model2_scores)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    r2str = "R2 = %.6f" %(r_squared)
    # R2 text coordinates.
    max_x = max(model1_scores)
    min_y = min(model2_scores)

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.scatterplot(x="m1_score", y="m2_score", data=df, color='#69e9f6', s=3)
        plt.text(max_x , min_y, r2str, color='black', horizontalalignment='right', size=10)
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=x_label)
        ax.set_ylabel(y_label)
        #ax.tick_params(axis='x', labelsize=18)
        #ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=150, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.scatterplot(x="m1_score", y="m2_score", data=df, color='blue', s=3)
        plt.text(max_x , min_y, r2str, color='blue', horizontalalignment='right', size=10)
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=x_label)
        ax.set_ylabel(y_label)
        #ax.tick_params(axis='x', labelsize=18)
        #ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=150, bbox_inches='tight', transparent=True)


################################################################################

def get_jaccard_index(list1, list2):
    """
    Given two lists of string/numbers, calculate Jaccard index (similarity)
    between the two sets.
    J(A,B) = intersection(A,B) / union(A,B)
    0 <= J(A,B) <= 1
    1 if sets are identical
    0 if intersection = 0

    >>> list1 = [1,1,2,3]
    >>> list2 = [2,3,4]
    >>> get_jaccard_index(list1,list2)
    0.5
    >>> list2 = [1,2,3]
    >>> get_jaccard_index(list1,list2)
    1.0
    >>> list2 = [4]
    >>> get_jaccard_index(list1,list2)
    0.0

    """
    assert list1, "list1 empty"
    assert list2, "list2 empty"
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


################################################################################

def create_eval_kmer_score_kde_plot(set_scores, out_plot,
                                    set_label="Positives",
                                    x_label="k-mer score",
                                    y_label="Density",
                                    theme=1):
    """
    Create graphprot2 eval kdeplot, plotting density for set of k-mer scores.

    """
    assert set_scores, "set_scores empty"
    data = {'score': []}
    data['score'] += set_scores
    df = pd.DataFrame (data, columns = ['score'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.kdeplot(x="score", data=df, color='#69e9f6')
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=x_label)
        ax.set_ylabel(y_label)
        #ax.tick_params(axis='x', labelsize=18)
        #ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=150, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.kdeplot(x="score", data=df, color='blue')
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=x_label)
        ax.set_ylabel(y_label)
        #ax.tick_params(axis='x', labelsize=18)
        #ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=150, bbox_inches='tight', transparent=True)


################################################################################

def create_eval_kde_plot(set1_scores, set2_scores, out_plot,
                         set1_label="Positives",
                         set2_label="Negatives",
                         x_label="Whole-site score",
                         y_label="Density",
                         theme=1):
    """
    Create graphprot2 eval kdeplot, plotting densities for two sets of
    scores.

    """
    assert set1_scores, "set1_scores empty"
    assert set2_scores, "set2_scores empty"
    set1_c = len(set1_scores)
    set2_c = len(set2_scores)
    # assert set1_c == set2_c, "differing set sizes for set1_c and set2_c (%i != %i)" %(set1_c, set2_c)
    data = {'set': [], 'score': []}
    data['set'] += set1_c*[set1_label] + set2_c*[set2_label]
    data['score'] += set1_scores + set2_scores
    df = pd.DataFrame (data, columns = ['set','score'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.kdeplot(x="score", data=df, hue="set", palette=["#69e9f6", "#f154b2"])
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=x_label)
        ax.set_ylabel(y_label)
        #ax.tick_params(axis='x', labelsize=18)
        #ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=150, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        # aqua, deepskyblue
        sns.kdeplot(x="score", data=df, hue="set", palette=["blue", "deepskyblue"])
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=x_label)
        ax.set_ylabel(y_label)
        #ax.tick_params(axis='x', labelsize=18)
        #ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=150, bbox_inches='tight', transparent=True)


################################################################################

def gp2_eval_generate_html_report(ws_scores, neg_ws_scores,
                                  out_folder, gp2lib_path,
                                  html_report_out="report.graphprot2_eval.html",
                                  kmer2rank_dic=False,
                                  kmer2sc_dic=False,
                                  kmer2c_dic=False,
                                  kmer2scstdev_dic=False,
                                  kmer2bestsc_dic=False,
                                  kmer2scrank_dic=False,
                                  kmer2avgscrank_dic=False,
                                  kmer2mm_dic=False,
                                  kmer_stdev_dic=False,
                                  kmer2bestmm_dic=False,
                                  ch_info_dic=False,
                                  kmer_size=5,
                                  min_kmer_score=0.1,
                                  min_jacc_sc=0.1,
                                  top_motif_file_dic=False,
                                  bottom_motif_file_dic=False,
                                  kmer_top_n=25,
                                  onlyseq=True,
                                  add_ws_scores=False,
                                  theme=1,
                                  lookup_kmer=False,
                                  jacc_scores_dic=False,
                                  jacc_stats_dic=False,
                                  plots_subfolder="html_plots"):
    """
    Generate HTML report for graphprot2 eval, showing stats and plots regarding
    whole site scores and k-mers.

    For onlyseq:
        - m1m2 scores scatter plot
        - whole site scores density plot (+ vs -)
        - kmer scores density plot
        - top scoring kmer stats table
        - jaccard index analysis (co-occuring positive top scoring kmers)
    For additional features:
        - above and:
        - extended stats with plots (avg scores vs top scores for each kmer)
        - avg vs top scores correlation plot (like scatter plot)
        - jaccard index using top scores for ranking

    """
    # Checks.
    assert ws_scores, "ws_scores empty"
    assert neg_ws_scores, "neg_ws_scores empty"
    assert os.path.exists(out_folder), "out_folder does not exist"
    assert os.path.exists(gp2lib_path), "gp2lib_path does not exist"
    assert kmer2rank_dic, "kmer2rank_dic needed"
    assert kmer2sc_dic, "kmer2sc_dic needed"
    assert kmer2c_dic, "kmer2c_dic needed"
    assert kmer2scrank_dic, "kmer2scrank_dic needed"
    assert jacc_scores_dic, "jacc_scores_dic needed"
    assert jacc_stats_dic, "jacc_stats_dic needed"
    if not onlyseq:
        assert kmer2bestsc_dic, "kmer2bestsc_dic needed in case of additional features"
        assert kmer2scstdev_dic, "kmer2scstdev_dic needed in case of additional features"
        assert kmer2mm_dic, "kmer2mm_dic needed in case of additional features"
        assert kmer_stdev_dic, "kmer_stdev_dic needed in case of additional features"
        assert kmer2bestmm_dic, "kmer2bestmm_dic needed in case of additional features"
        assert ch_info_dic, "ch_info_dic needed in case of additional features"
        assert kmer2avgscrank_dic, "kmer2avgscrank_dic needed in case of additional features"

    # Import markdown to generate report.
    from markdown import markdown

    # Output subfolder for plots.
    plots_folder = plots_subfolder
    plots_out_folder = out_folder + "/" + plots_folder
    if not os.path.exists(plots_out_folder):
        os.makedirs(plots_out_folder)
    # Output files.
    html_out = out_folder + "/" + "report.graphprot2_eval.html"
    if html_report_out:
        html_out = html_report_out
    # Plot files.
    ws_sc_plot = "whole_site_scores_kde_plot.png"
    kmer_sc_plot = "kmer_scores_kde_plot.png"
    avg_best_kmer_kde_plot = "avg_best_kmer_scores_kde_plot.png"
    avg_best_kmer_scatter_plot = "avg_best_kmer_scores_scatter_plot.png"
    model_comp_plot = "model_comparison_plot.png"
    ws_sc_plot_out = plots_out_folder + "/" + ws_sc_plot
    kmer_sc_plot_out = plots_out_folder + "/" + kmer_sc_plot
    model_comp_plot_out = plots_out_folder + "/" + model_comp_plot
    avg_best_kmer_kde_plot_out = plots_out_folder + "/" + avg_best_kmer_kde_plot
    avg_best_kmer_scatter_plot_out = plots_out_folder + "/" + avg_best_kmer_scatter_plot

    # Logo paths.
    logo1_path = gp2lib_path + "/content/logo1.png"
    logo2_path = gp2lib_path + "/content/logo2.png"
    logo3_path = gp2lib_path + "/content/logo3.png"
    sorttable_js_path = gp2lib_path + "/content/sorttable.js"

    # Create theme-specific HTML header.
    if theme == 1:
        mdtext = """
<head>
<title>GraphProt2 - Model Evaluation Report</title>
<script src="%s" type="text/javascript"></script>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="600" />

""" %(sorttable_js_path, logo1_path)
    elif theme == 2:
        mdtext = """
<head>
<title>GraphProt2 - Model Evaluation Report</title>
<script src="%s" type="text/javascript"></script>
<style>
h1 {color:#fd3b9d;}
h2 {color:#fd3b9d;}
h3 {color:#fd3b9d;}
</style>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="500" />

<body style="font-family:sans-serif" bgcolor="#190250" text="#fcc826" link="#fd3b9d" vlink="#fd3b9d" alink="#fd3b9d">

""" %(sorttable_js_path, logo2_path)
    elif theme == 3:
        mdtext = """
<head>
<title>GraphProt2 - Model Evaluation Report</title>
<script src="%s" type="text/javascript"></script>
<style>
h1 {color:#1fcc2c;}
h2 {color:#1fcc2c;}
h3 {color:#1fcc2c;}
</style>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="400" />

<body style="font-family:monospace" bgcolor="#1d271e" text="#1fcc2c" link="#1fcc2c" vlink="#1fcc2c" alink="#1fcc2c">

""" %(sorttable_js_path, logo3_path)
    else:
        assert False, "invalid theme ID given"

    # Add first section markdown.
    mdtext += """

# Model Evaluation Report

List of available model evaluation statistics generated
by GraphProt2 (graphprot2 eval):

- [Whole-site score distribution](#ws-scores-plot)"""
    if onlyseq:
        mdtext += "\n"
        mdtext += "- [k-mer score distribution](#kmer-scores-plot)"
    else:
        mdtext += "\n"
        mdtext += "- [k-mer score distributions](#kmer-scores-plots)"
    mdtext += "\n"
    mdtext += "- [k-mer statistics](#kmer-stats)"
    mdtext += "\n"
    mdtext += "- [k-mer co-occurrence statistics](#kmer-cooc-stats)"
    if lookup_kmer:
        mdtext += "\n"
        mdtext += "- [Lookup k-mer statistics](#lookup-kmer-stats)\n"
        mdtext += "- [Lookup k-mer co-occurrence statistics](#lookup-kmer-cooc-stats)"
    if add_ws_scores:
        mdtext += "\n"
        mdtext += "- [Model comparison](#model-comp-plot)"
    mdtext += "\n&nbsp;\n"

    """
    Whole-site score distributions for positives and negatives.

    """
    print("Generate whole-site scores plot .. ")
    # Make whole-site score distributions for positives and negatives.
    create_eval_kde_plot(ws_scores, neg_ws_scores, ws_sc_plot_out,
                         set1_label="Positives",
                         set2_label="Negatives",
                         x_label="Whole-site score",
                         y_label="Density",
                         theme=theme)
    plot_path = plots_folder + "/" + ws_sc_plot

    mdtext += """
## Whole-site score distribution ### {#ws-scores-plot}

Whole-site score distributions for the positive and negative sequence
set, scored by the trained model. Since the model was trained on these
two sequence sets, we expect on average higher scores for the positive
sequences (given a sufficient model performance).

"""
    mdtext += '<img src="' + plot_path + '" alt="Whole-site score distributions"' + "\n"
    mdtext += 'title="Whole-site score distributions" width="500" />' + "\n"
    mdtext += """

**Figure:** Whole-site score distributions for the positive (Positives) and
negative (Negatives) sequence set, scored by the trained model.

&nbsp;

"""

    """
    k-mer score distribution(s)

    - If onlyseq, one distribution plot (k-mer scores)
    - If additional features, plot best scores + average scores,
      first distribution plot and then scatter plot.

    """
    kmer_sc_list = []
    for kmer in kmer2sc_dic:
        kmer_sc_list.append(kmer2sc_dic[kmer])
    c_kmers = len(kmer_sc_list)
    # Get best k-mer scores list.
    best_kmer_sc_list = []
    if not onlyseq:
        for kmer in kmer2bestsc_dic:
            best_kmer_sc_list.append(kmer2bestsc_dic[kmer])

    if onlyseq:

        print("Generate sequence k-mer stats and plots ... ")

        set_label = "Positive %i-mers" %(kmer_size)
        x_label = "%i-mer score" %(kmer_size)
        create_eval_kmer_score_kde_plot(kmer_sc_list, kmer_sc_plot_out,
                                        set_label=set_label,
                                        x_label=x_label,
                                        y_label="Density",
                                        theme=theme)
        plot_path = plots_folder + "/" + kmer_sc_plot

        mdtext += """
## k-mer score distribution ### {#kmer-scores-plot}

Score distribution of sequence k-mers (k = %i) found in the positive training set.
k-mers are scored by the model, using the subgraph encompassing the k-mer.

""" %(kmer_size)
        mdtext += '<img src="' + plot_path + '" alt="k-mer score distribution"' + "\n"
        mdtext += 'title="k-mer score distribution" width="500" />' + "\n"
        mdtext += """

**Figure:** Score distribution of k-mers found in the positive training sequence set.

&nbsp;

"""

        # k-mer stats table for onlyseq.
        mdtext += """
## k-mer statistics ### {#kmer-stats}

**Table:** Sequence k-mer statistics (score (sc) rank, k-mer score, k-mer count,
count rank) for the top %i scoring sequence %i-mers (ranked by k-mer score).

""" %(kmer_top_n, kmer_size)

        mdtext += "| sc rank | &nbsp; k-mer &nbsp; | &nbsp; k-mer sc &nbsp; | k-mer count | k-mer count rank | \n"
        mdtext += "| :-: | :-: | :-: | :-: | :-: |\n"
        sc_rank = 0
        for kmer, sc in sorted(kmer2sc_dic.items(), key=lambda item: item[1], reverse=True):
            sc_rank += 1
            if sc_rank > kmer_top_n:
                break
            kmer_count_rank = kmer2rank_dic[kmer]
            kmer_count = kmer2c_dic[kmer]
            mdtext += "| %i | %s | %.6f | %i | %i |\n" %(sc_rank, kmer, sc, kmer_count, kmer_count_rank)
        mdtext += "\n&nbsp;\n&nbsp;\n"

        # k-mer co-occurrence statistics table for onlyseq.
        mdtext += """
## k-mer co-occurrence statistics ### {#kmer-cooc-stats}

**Table:** sequence k-mer co-occurrence statistics (Jaccard index (JI) rank, k-mer 1,
k-mer 2, Jaccard index, k-mer 1 score (sc), k-mer 2 score, mean minimum distance
of k-mers on sequences containing both k-mers with standard deviation, number
of intersections (sequences containing both k-mers), size of union of sequences
containing the two k-mers).
Entries are sorted by the Jaccard index of the two k-mers (where set is defined
as the set of sequences containing the k-mer). Only the top %i k-mer pairs
are shown, with a minimum Jaccard index of %s, a minimum k-mer score of
%s, and a minimum # of intersections of 10, and a minimum mean minimum distance
(yes!) of %i. An empty table means that no pairs have met the filtering criteria.

""" %(kmer_top_n, str(min_jacc_sc), str(min_kmer_score), kmer_size)

        mdtext += "| JI rank | &nbsp; k-mer 1 &nbsp; |  &nbsp; k-mer 2 &nbsp; | JI | k-mer 1 sc | k-mer 2 sc | mean min dist (+- stdev) | # intersect | # union | \n"
        mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | \n"
        jacc_rank = 0
        kmers2sumsc_dic = {}
        for kmers, jacc_idx in sorted(jacc_scores_dic.items(), key=lambda item: item[1], reverse=True):
            min_dist_mean = jacc_stats_dic[kmers][5]
            c_intersects = jacc_stats_dic[kmers][7]
            if min_dist_mean < kmer_size:
                continue
            if c_intersects < 10:
                continue
            jacc_rank += 1
            if jacc_rank > kmer_top_n:
                break
            kmer1 = jacc_stats_dic[kmers][0]
            kmer2 = jacc_stats_dic[kmers][1]
            kmer1_sc = jacc_stats_dic[kmers][2]
            kmer2_sc = jacc_stats_dic[kmers][3]
            sum_sc = kmer1_sc + kmer2_sc
            kmers2sumsc_dic[kmers] = sum_sc
            min_dist_stdev = jacc_stats_dic[kmers][6]
            c_union = jacc_stats_dic[kmers][8]
            mdtext += "| %i | %s | %s | %.6f | %.6f | %.6f | %.6f (+- %.6f) | %i | %i |\n" %(jacc_rank, kmer1, kmer2, jacc_idx, kmer1_sc, kmer2_sc, min_dist_mean, min_dist_stdev, c_intersects, c_union)
        mdtext += "\n&nbsp;\n&nbsp;\n"

        if lookup_kmer:
            # Lookup k-mer stats table for onlyseq.
            print("Generate --lookup-kmer stats and plots ... ")
            mdtext += """
## Lookup k-mer statistics ### {#lookup-kmer-stats}

**Table:** lookup k-mer statistics (score (sc) rank, k-mer score, k-mer count,
count rank) for lookup k-mer %s.

""" %(lookup_kmer)

            mdtext += "| sc rank | &nbsp; k-mer &nbsp; | &nbsp; k-mer sc &nbsp; | k-mer count | k-mer count rank | \n"
            mdtext += "| :-: | :-: | :-: | :-: | :-: |\n"
            lk_sc_rank = kmer2scrank_dic[lookup_kmer]
            lk_sc = kmer2sc_dic[lookup_kmer]
            lk_count_rank = kmer2rank_dic[lookup_kmer]
            lk_count = kmer2c_dic[lookup_kmer]
            mdtext += "| %i | %s | %.6f | %i | %i |\n" %(lk_sc_rank, lookup_kmer, lk_sc, lk_count, lk_count_rank)
            mdtext += "\n&nbsp;\n&nbsp;\n"

            # Lookup k-mer co-occurrence statistics table for onlyseq.
            mdtext += """
## Lookup k-mer co-occurrence statistics ### {#lookup-kmer-cooc-stats}

**Table:** Lookup sequence k-mer co-occurrence statistics.
Entries are sorted by the Jaccard index (JI) of the two k-mers.
Only the top %i k-mer pairs are shown, with a minimum Jaccard index
of %s, a minimum k-mer score (sc) of %s, and a minimum # of intersections of 10.

""" %(kmer_top_n, str(min_jacc_sc), str(min_kmer_score))

            mdtext += "| JI rank | &nbsp; k-mer 1 &nbsp; |  &nbsp; k-mer 2 &nbsp; | JI | k-mer 1 sc | k-mer 2 sc | mean min dist (+- stdev) | # intersect | # union | \n"
            mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | \n"
            jacc_rank = 0
            kmers2sumsc_dic = {}
            for kmers, jacc_idx in sorted(jacc_scores_dic.items(), key=lambda item: item[1], reverse=True):
                jacc_rank += 1
                kmer1 = jacc_stats_dic[kmers][0]
                kmer2 = jacc_stats_dic[kmers][1]
                if kmer1 != lookup_kmer and kmer2 != lookup_kmer:
                    continue
                kmer1_sc = jacc_stats_dic[kmers][2]
                kmer2_sc = jacc_stats_dic[kmers][3]
                sum_sc = kmer1_sc + kmer2_sc
                kmers2sumsc_dic[kmers] = sum_sc
                min_dist_mean = jacc_stats_dic[kmers][5]
                min_dist_stdev = jacc_stats_dic[kmers][6]
                c_intersects = jacc_stats_dic[kmers][7]
                c_union = jacc_stats_dic[kmers][8]
                mdtext += "| %i | %s | %s | %.6f | %.6f | %.6f | %.6f (+- %.6f) | %i | %i |\n" %(jacc_rank, kmer1, kmer2, jacc_idx, kmer1_sc, kmer2_sc, min_dist_mean, min_dist_stdev, c_intersects, c_union)
            mdtext += "\n&nbsp;\n&nbsp;\n"

    else:

        print("Generate additional feature k-mer stats and plots ... ")

        set1_label = "Average %i-mer scores" %(kmer_size)
        set2_label = "Best %i-mer scores" %(kmer_size)
        x_label = "%i-mer score" %(kmer_size)
        create_eval_kde_plot(kmer_sc_list, best_kmer_sc_list, avg_best_kmer_kde_plot_out,
                             set1_label=set1_label,
                             set2_label=set2_label,
                             x_label=x_label,
                             y_label="Density",
                             theme=theme)
        plot_path1 = plots_folder + "/" + avg_best_kmer_kde_plot

        x_label = "Average %i-mer score" %(kmer_size)
        y_label = "Best %i-mer score" %(kmer_size)
        create_eval_model_comp_scatter_plot(kmer_sc_list, best_kmer_sc_list,
                                            avg_best_kmer_scatter_plot_out,
                                            x_label=x_label,
                                            y_label=y_label,
                                            theme=theme)
        plot_path2 = plots_folder + "/" + avg_best_kmer_scatter_plot

        mdtext += """
## k-mer score distributions ### {#kmer-scores-plots}

Average and best score distribution of all k-mers (k = %i) found in the
positive training set (with additional features).
As additional features vary, the same sequence k-mer can have different
scores, depending on the underlying additional feature values at a certain
sequence position. It is therefore possible to select the best score or
calculate the average score for each sequence k-mer from all k-mer
occurences in a positive set with additional features.

""" %(kmer_size)
        mdtext += '<img src="' + plot_path1 + '" alt="average vs best k-mer score distribution"' + "\n"
        mdtext += 'title="average vs best k-mer score distribution" width="500" />' + "\n"
        mdtext += """

**Figure:** Average and best score distribution of all k-mers found in the
positive training set (with additional features).
&nbsp;

"""

        mdtext += '<img src="' + plot_path2 + '" alt="average vs best k-mer scores scatter plot"' + "\n"
        mdtext += 'title="average vs best k-mer scores scatter plot" width="500" />' + "\n"
        mdtext += """

**Figure:** Average vs. best k-mer scores scatter plot for all k-mers found in the
positive training set (with additional features). The more the best score of each k-mer
deviates from its average score, the more scattered the points should be above the
diagonal line (y = x) in the upper-left area. No points should be located below
the diagonal (lower right), since the best score is always >= the average score.
Points on the diagonal are likely k-mers with an occurence = 1, i.e., best score ==
average score. Points near the diagonal are k-mers that feature similar annotated
additional feature values across the positive dataset.

&nbsp;

"""

        # k-mer stats table for additional features.
        mdtext += """
## k-mer statistics ### {#kmer-stats}

**Table:** Sequence k-mer statistics with additional features (best score (sc) rank,
best k-mer score, average (avg) k-mer score + standard deviation, average k-mer rank,
total k-mer count, and total count rank) for the top %i scoring %i-mers
(ranked by best k-mer score). Best scoring motif + average scoring motif are
also shown.

""" %(kmer_top_n, kmer_size)

        # Motif plots output folders.
        avg_motif_plots_folder = plots_out_folder + "/" + "avg_motif_plots"
        best_motif_plots_folder = plots_out_folder + "/" + "best_motif_plots"
        if not os.path.exists(avg_motif_plots_folder):
            os.makedirs(avg_motif_plots_folder)
        if not os.path.exists(best_motif_plots_folder):
            os.makedirs(best_motif_plots_folder)

        kmer_i = 10 ** len(str(c_kmers))
        sc_rank = 0

        mdtext += "| best sc rank | &nbsp; k-mer &nbsp; | best k-mer sc | best sc motif | avg k-mer rank | avg k-mer sc | avg sc stdev | avg sc motif | avg + best sc | diff(avg sc, best sc) | k-mer count | k-mer count rank |\n"
        mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |\n"

        for kmer, best_sc in sorted(kmer2bestsc_dic.items(), key=lambda item: item[1], reverse=True):
            sc_rank += 1
            kmer_i += 1
            if sc_rank > kmer_top_n:
                break
            kmer_count_rank = kmer2rank_dic[kmer]
            kmer_count = kmer2c_dic[kmer]
            avg_sc = kmer2sc_dic[kmer]
            avg_sc_stdev = kmer2scstdev_dic[kmer]
            avg_sc_rank = kmer2avgscrank_dic[kmer]
            sc_sum = best_sc + avg_sc
            sc_diff = abs(best_sc-avg_sc)
            # Generate average score motif.
            avg_sc_plot_file = avg_motif_plots_folder + "/" + str(kmer_i)[1:] + "_" + kmer + ".avg_sc.png"
            make_motif_plot(kmer2mm_dic[kmer], ch_info_dic, avg_sc_plot_file,
                            fid2stdev_dic=kmer_stdev_dic[kmer])
            pp1 = plots_folder + "/avg_motif_plots/" + str(kmer_i)[1:] + "_" + kmer + ".avg_sc.png"
            # Generate best score motif.
            best_sc_plot_file = best_motif_plots_folder + "/" + str(kmer_i)[1:] + "_" + kmer + ".best_sc.png"
            make_motif_plot(kmer2bestmm_dic[kmer], ch_info_dic, best_sc_plot_file,
                            fid2stdev_dic=False)
            pp2 = plots_folder + "/best_motif_plots/" + str(kmer_i)[1:] + "_" + kmer + ".best_sc.png"
            mdtext += '| %i | %s | %.6f | <image src = "%s" width="150px"></image> | %i | %.6f | %.6f | <image src = "%s" width="150px"></image> | %.6f | %.6f | %i | %i |\n' %(sc_rank, kmer, best_sc, pp2, avg_sc_rank, avg_sc, avg_sc_stdev, pp1, sc_sum, sc_diff, kmer_count, kmer_count_rank)
        mdtext += "\n&nbsp;\n&nbsp;\n"

        # k-mer co-occurrence statistics table for additional features.
        mdtext += """
## k-mer co-occurrence statistics ### {#kmer-cooc-stats}

**Table:** sequence k-mer co-occurrence statistics with additional features
(Jaccard index (JI) rank, k-mer 1, k-mer 2, Jaccard index, best k-mer 1 score (sc),
best k-mer 2 score, mean minimum distance of k-mers on sequences containing both
k-mers with standard deviation, number of intersections (sequences containing
both k-mers), size of union of sequences containing the two k-mers).
Entries are sorted by the Jaccard index of the two k-mers (where set is defined
as the set of sequences containing the k-mer). Only the top %i k-mer pairs
are shown, with a minimum Jaccard index of %s a minimum k-mer score of
%s, a minimum # of intersections of 10, , and a minimum mean minimum distance
(yes!) of %i. An empty table means that no pairs have met the filtering criteria.

""" %(kmer_top_n, str(min_jacc_sc), str(min_kmer_score), kmer_size)

        mdtext += "| JI rank | &nbsp; k-mer 1 &nbsp; |  &nbsp; k-mer 2 &nbsp; | JI | k-mer 1 sc | k-mer 2 sc | mean min dist | dist stdev | # intersect | # union | \n"
        mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | \n"
        jacc_rank = 0
        kmers2sumsc_dic = {}
        for kmers, jacc_idx in sorted(jacc_scores_dic.items(), key=lambda item: item[1], reverse=True):
            min_dist_mean = jacc_stats_dic[kmers][5]
            c_intersects = jacc_stats_dic[kmers][7]
            if c_intersects < 10:
                continue
            if min_dist_mean < kmer_size:
                continue
            jacc_rank += 1
            if jacc_rank > kmer_top_n:
                break
            kmer1 = jacc_stats_dic[kmers][0]
            kmer2 = jacc_stats_dic[kmers][1]
            kmer1_sc = jacc_stats_dic[kmers][2]
            kmer2_sc = jacc_stats_dic[kmers][3]
            sum_sc = kmer1_sc + kmer2_sc
            kmers2sumsc_dic[kmers] = sum_sc
            min_dist_stdev = jacc_stats_dic[kmers][6]
            c_union = jacc_stats_dic[kmers][8]
            mdtext += "| %i | %s | %s | %.6f | %.6f | %.6f | %.3f | %.3f | %i | %i |\n" %(jacc_rank, kmer1, kmer2, jacc_idx, kmer1_sc, kmer2_sc, min_dist_mean, min_dist_stdev, c_intersects, c_union)
        mdtext += "\n&nbsp;\n&nbsp;\n"

        if lookup_kmer:
            # Lookup k-mer stats table for additional features.
            print("Generate --lookup-kmer stats and plots ... ")
            mdtext += """
## Lookup k-mer statistics ### {#lookup-kmer-stats}

**Table:** lookup k-mer statistics (best score (sc) rank, average (avg) score rank,
k-mer score, k-mer count,
count rank) for lookup k-mer %s and training data with additional features.

""" %(lookup_kmer)

            mdtext += "| best sc rank | avg sc rank | &nbsp; k-mer &nbsp; | best k-mer sc | avg k-mer sc | avg sc stdev | best sc motif | avg sc motif | k-mer count | k-mer count rank | \n"
            mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | \n"
            lk_best_sc_rank = kmer2scrank_dic[lookup_kmer]
            lk_avg_sc_rank = kmer2avgscrank_dic[lookup_kmer]
            lk_best_sc = kmer2bestsc_dic[lookup_kmer]
            lk_avg_sc = kmer2sc_dic[lookup_kmer]
            lk_avg_sc_stdev = kmer2scstdev_dic[lookup_kmer]
            lk_count_rank = kmer2rank_dic[lookup_kmer]
            lk_count = kmer2c_dic[lookup_kmer]
            lk_avg_plot_file = plots_out_folder + "/lookup_kmer_%s.avg_sc.png" %(lookup_kmer)
            make_motif_plot(kmer2mm_dic[lookup_kmer], ch_info_dic, lk_avg_plot_file,
                            fid2stdev_dic=kmer_stdev_dic[lookup_kmer])
            pp1 = plots_folder + "/lookup_kmer_%s.avg_sc.png" %(lookup_kmer)

            lk_best_plot_file = plots_out_folder + "/lookup_kmer_%s.best_sc.png" %(lookup_kmer)
            make_motif_plot(kmer2bestmm_dic[lookup_kmer], ch_info_dic, lk_best_plot_file,
                            fid2stdev_dic=False)
            pp2 = plots_folder + "/lookup_kmer_%s.best_sc.png" %(lookup_kmer)
            mdtext += '| %i | %i | %s | %.6f | %.6f | %.6f | <image src = "%s" width="150px"></image> | <image src = "%s" width="150px"></image> | %i | %i |\n' %(lk_best_sc_rank, lk_avg_sc_rank, kmer, lk_best_sc, lk_avg_sc, lk_avg_sc_stdev, pp2, pp1, kmer_count, kmer_count_rank)
            mdtext += "\n&nbsp;\n&nbsp;\n"

            # Lookup k-mer co-occurrence statistics table for additional features.
            mdtext += """
## Lookup k-mer co-occurrence statistics ### {#lookup-kmer-cooc-stats}

**Table:** Lookup k-mer co-occurrence statistics for for lookup k-mer %s and
training data with additional features.
Entries are sorted by the Jaccard index (JI) of the two k-mers. The Jaccard index of
two k-mers is calculated based on the two sequence sets that contain the
two k-mers, with each sequence ID being a set member. Scores are best k-mer scores.
Only the top %i k-mer pairs including %s are shown, with a minimum Jaccard index
of %s and a minimum k-mer score of %s.

""" %(lookup_kmer, kmer_top_n, lookup_kmer, str(min_jacc_sc), str(min_kmer_score))

            mdtext += "| JI rank | &nbsp; k-mer 1 &nbsp; |  &nbsp; k-mer 2 &nbsp; | JI | k-mer 1 score | k-mer 2 score | mean min dist (+- stdev) | # intersect | # union | \n"
            mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | \n"
            jacc_rank = 0
            kmers2sumsc_dic = {}
            for kmers, jacc_idx in sorted(jacc_scores_dic.items(), key=lambda item: item[1], reverse=True):
                kmer1 = jacc_stats_dic[kmers][0]
                kmer2 = jacc_stats_dic[kmers][1]
                if kmer1 != lookup_kmer and kmer2 != lookup_kmer:
                    continue
                jacc_rank += 1
                if jacc_rank > kmer_top_n:
                    break
                kmer1_sc = jacc_stats_dic[kmers][2]
                kmer2_sc = jacc_stats_dic[kmers][3]
                sum_sc = kmer1_sc + kmer2_sc
                kmers2sumsc_dic[kmers] = sum_sc
                min_dist_mean = jacc_stats_dic[kmers][5]
                min_dist_stdev = jacc_stats_dic[kmers][6]
                c_intersects = jacc_stats_dic[kmers][7]
                c_union = jacc_stats_dic[kmers][8]
                mdtext += "| %i | %s | %s | %.6f | %.6f | %.6f | %.6f (+- %.6f) | %i | %i |\n" %(jacc_rank, kmer1, kmer2, jacc_idx, kmer1_sc, kmer2_sc, min_dist_mean, min_dist_stdev, c_intersects, c_union)
            mdtext += "\n&nbsp;\n&nbsp;\n"

    """
    Model comparison plot.

    """

    if add_ws_scores:

        x_label = "Model 1 score"
        y_label = "Model 2 score"
        print("Generate --train-in vs --add-train-in model comparison plot ... ")
        create_eval_model_comp_scatter_plot(ws_scores, add_ws_scores,
                                            model_comp_plot_out,
                                            x_label=x_label,
                                            y_label=y_label,
                                            theme=theme)
        plot_path = plots_folder + "/" + model_comp_plot

        mdtext += """
## Model comparison ### {#model-comp-plot}

To compare two models, the postive training set is scored with two models
and the two model scores are displayed as a scatter plot. More similar
models should show higher correlation, resulting in a higher R2 score
(coeffient of determination).

"""
        mdtext += '<img src="' + plot_path + '" alt="model comparison plot"' + "\n"
        mdtext += 'title="model comparison plot" width="500" />' + "\n"
        mdtext += """

**Figure:** Model comparison scatter plot, comparing whole-site model scores
on the positive training set for the two input models. Model 1: model from
--train-in folder. Model 2: model from --add-train-in.
&nbsp;

"""

    print("Generate HTML report ... ")

    # Convert mdtext to html.
    md2html = markdown(mdtext, extensions=['attr_list', 'tables'])

    #OUTMD = open(md_out,"w")
    #OUTMD.write("%s\n" %(mdtext))
    #OUTMD.close()
    OUTHTML = open(html_out,"w")
    OUTHTML.write("%s\n" %(md2html))
    OUTHTML.close()

    # change <table> to sortable.
    check_cmd = "sed -i 's/<table>/<table class=" + '"sortable"' + ">/g' " + html_out
    output = subprocess.getoutput(check_cmd)
    error = False
    if output:
        error = True
    assert error == False, "sed command returned error:\n%s" %(output)


################################################################################

def gp2_gt_generate_html_report(pos_seqs_dic, neg_seqs_dic, out_folder,
                                dataset_type, gp2lib_path,
                                html_report_out=False,
                                plots_subfolder=False,
                                id2ucr_dic=False,
                                pos_str_stats_dic=False,
                                neg_str_stats_dic=False,
                                pos_phastcons_stats_dic=False,
                                neg_phastcons_stats_dic=False,
                                pos_phylop_stats_dic=False,
                                neg_phylop_stats_dic=False,
                                pos_eia_stats_dic=False,
                                neg_eia_stats_dic=False,
                                pos_tra_stats_dic=False,
                                neg_tra_stats_dic=False,
                                pos_rra_stats_dic=False,
                                neg_rra_stats_dic=False,
                                add_feat_dic_list=False,
                                target_gbtc_dic=False,
                                all_gbtc_dic=False,
                                t2hc_dic=False,
                                t2i_dic=False,
                                theme=1,
                                kmer_top=10,
                                target_top=10,
                                rna=True,
                                uc_entropy=True,
                                ):
    """
    Generate HTML report for graphprot2 gt, comparing extracted positive
    with negative set.

    pos_seqs_dic:
        Positive set sequences dictionary.
    neg_seqs_dic:
        Negative set sequences dictionary.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> "uppercase_start-uppercase_end"
        where both positions are 1-based.
        If given, use only subsequences defined by this dictionary for
        generating sequence stats.
    pos_str_stats_dic:
        Positive set structure statistics dictionary
    neg_str_stats_dic:
        Negative set structure statistics dictionary
    pos_phastcons_stats_dic:
        Positive set phastcons scores statistics dictionary
    neg_phastcons_stats_dic:
        Negative set phastcons scores statistics dictionary
    add_feat_dic_list:
        List of dictionaries with additional BED feature statistics,
        where positive and corresponding negative set are stored together,
        so indices 1,2 3,4 5,6 ... belong together (positive stats dic first).
    out_folder:
        graphprot2 gt results output folder, to store report in.
    rna:
        Set True if input sequences are RNA.
    uc_entropy:
        Calculate sequence entropies only for uppercase sequence parts,
        ignoring the lowercase context sequence parts.
    html_report_out:
        HTML report output file.
    target_gbtc_dic:
        Gene biotype counts for target set dictionary.
    all_gbtc_dic:
        Gene biotype counts for all genes dictionary (gene biotype -> count)
    t2hc_dic:
        Transcript ID to hit count (# sites on transcript) dictionary.
    t2i_dic:
        Transcript ID to info list dictionary.

    More to add:
    References
    GraphProt2 version
    Command line call

    """
    # Checks.
    ds_types = {'s':1, 't':1, 'g':1}
    assert dataset_type in ds_types, "invalid dataset type given (expected g, s, or t)"
    # Import markdown to generate report.
    from markdown import markdown

    # Checks.
    if add_feat_dic_list:
        if len(add_feat_dic_list) % 2:
            assert False, "even number of dictionaries expected for given add_feat_dic_list"

    # Output subfolder for plots.
    plots_folder = plots_subfolder
    plots_out_folder = out_folder + "/" + plots_folder
    if not os.path.exists(plots_out_folder):
        os.makedirs(plots_out_folder)
    # Output files.
    html_out = out_folder + "/" + "report.graphprot2_gt.html"
    if html_report_out:
        html_out = html_report_out
    #md_out = out_folder + "/" + "report.graphprot2_gt.md"
    # Plot files.
    lengths_plot = "set_lengths_plot.png"
    entropy_plot = "sequence_complexity_plot.png"
    dint_plot = "dint_percentages_plot.png"
    str_elem_plot = "str_elem_plot.png"
    phastcons_plot = "phastcons_plot.png"
    phylop_plot = "phylop_plot.png"
    eia_plot = "exon_intron_region_plot.png"
    tra_plot = "transcript_region_plot.png"
    rra_plot = "repeat_region_plot.png"
    lengths_plot_out = plots_out_folder + "/" + lengths_plot
    entropy_plot_out = plots_out_folder + "/" + entropy_plot
    dint_plot_out = plots_out_folder + "/" + dint_plot
    str_elem_plot_out = plots_out_folder + "/" + str_elem_plot
    phastcons_plot_out = plots_out_folder + "/" + phastcons_plot
    phylop_plot_out = plots_out_folder + "/" + phylop_plot
    eia_plot_out = plots_out_folder + "/" + eia_plot
    tra_plot_out = plots_out_folder + "/" + tra_plot
    rra_plot_out = plots_out_folder + "/" + rra_plot

    print("Generate statistics for HTML report ... ")

    """
    If only uppercase part of sequences should be used for stats,
    prune the sequence dictionaries based on uppercase region start
    and end info stored in id2ucr_dic.
    """
    if id2ucr_dic:
        for pos_id in pos_seqs_dic:
            seq = pos_seqs_dic[pos_id]
            uc_s = id2ucr_dic[pos_id][0]
            uc_e = id2ucr_dic[pos_id][1]
            pos_seqs_dic[pos_id] = seq[uc_s-1:uc_e]
        for neg_id in neg_seqs_dic:
            seq = neg_seqs_dic[neg_id]
            uc_s = id2ucr_dic[neg_id][0]
            uc_e = id2ucr_dic[neg_id][1]
            neg_seqs_dic[neg_id] = seq[uc_s-1:uc_e]

    # Site numbers.
    c_pos_out = len(pos_seqs_dic)
    c_neg_out = len(neg_seqs_dic)
    # Site lengths.
    pos_len_list = get_seq_len_list_from_dic(pos_seqs_dic)
    neg_len_list = get_seq_len_list_from_dic(neg_seqs_dic)
    # Get entropy scores for sequences.
    pos_entr_list = seqs_dic_calc_entropies(pos_seqs_dic, rna=rna,
                                            uc_part_only=uc_entropy)
    neg_entr_list = seqs_dic_calc_entropies(neg_seqs_dic, rna=rna,
                                            uc_part_only=uc_entropy)

    # Get set nucleotide frequencies.
    pos_ntc_dic = seqs_dic_count_nt_freqs(pos_seqs_dic, rna=rna,
                                          convert_to_uc=True)
    neg_ntc_dic = seqs_dic_count_nt_freqs(neg_seqs_dic, rna=rna,
                                          convert_to_uc=True)
    # Get nucleotide ratios.
    pos_ntr_dic = ntc_dic_to_ratio_dic(pos_ntc_dic, perc=True)
    neg_ntr_dic = ntc_dic_to_ratio_dic(neg_ntc_dic, perc=True)

    # Get dinucleotide percentages.
    pos_dintr_dic = seqs_dic_count_kmer_freqs(pos_seqs_dic, 2, rna=rna,
                                              return_ratios=True,
                                              perc=True,
                                              report_key_error=True,
                                              convert_to_uc=True)
    neg_dintr_dic = seqs_dic_count_kmer_freqs(neg_seqs_dic, 2, rna=rna,
                                              return_ratios=True,
                                              perc=True,
                                              report_key_error=True,
                                              convert_to_uc=True)
    # Get 3-mer percentages.
    pos_3mer_dic = seqs_dic_count_kmer_freqs(pos_seqs_dic, 3, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    neg_3mer_dic = seqs_dic_count_kmer_freqs(neg_seqs_dic, 3, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    # Get 4-mer percentages.
    pos_4mer_dic = seqs_dic_count_kmer_freqs(pos_seqs_dic, 4, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    neg_4mer_dic = seqs_dic_count_kmer_freqs(neg_seqs_dic, 4, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    # Get 5-mer percentages.
    pos_5mer_dic = seqs_dic_count_kmer_freqs(pos_seqs_dic, 5, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    neg_5mer_dic = seqs_dic_count_kmer_freqs(neg_seqs_dic, 5, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)

    # Logo paths.
    logo1_path = gp2lib_path + "/content/logo1.png"
    logo2_path = gp2lib_path + "/content/logo2.png"
    logo3_path = gp2lib_path + "/content/logo3.png"

    # Create theme-specific HTML header.
    if theme == 1:
        mdtext = """
<head>
<title>GraphProt2 - Training Set Generation Report</title>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="600" />

""" %(logo1_path)
    elif theme == 2:
        mdtext = """
<head>
<title>GraphProt2 - Training Set Generation Report</title>
<style>
h1 {color:#fd3b9d;}
h2 {color:#fd3b9d;}
h3 {color:#fd3b9d;}
</style>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="500" />

<body style="font-family:sans-serif" bgcolor="#190250" text="#fcc826" link="#fd3b9d" vlink="#fd3b9d" alink="#fd3b9d">

""" %(logo2_path)
    elif theme == 3:
        mdtext = """
<head>
<title>GraphProt2 - Training Set Generation Report</title>
<style>
h1 {color:#1fcc2c;}
h2 {color:#1fcc2c;}
h3 {color:#1fcc2c;}
</style>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="400" />

<body style="font-family:monospace" bgcolor="#1d271e" text="#1fcc2c" link="#1fcc2c" vlink="#1fcc2c" alink="#1fcc2c">

""" %(logo3_path)
    else:
        assert False, "invalid theme ID given"

    # Add first section markdown.
    mdtext += """

# Training set generation report

List of available statistics for the training dataset generated
by GraphProt2 (graphprot2 gt):

- [Training dataset statistics](#set-stats)
- [Site length distribution](#len-plot)
- [Sequence complexity distribution](#ent-plot)
- [Di-nucleotide distribution](#dint-plot)
- [Top k-mer statistics](#kmer-stats)"""



    if pos_str_stats_dic and neg_str_stats_dic:
        mdtext += "\n"
        mdtext += "- [Structural elements distribution](#str-elem-plot)\n"
        mdtext += "- [Secondary structure statistics](#bp-stats)"
    if pos_phastcons_stats_dic or pos_phylop_stats_dic:
        mdtext += "\n"
        mdtext += "- [Conservation scores distribution](#con-plot)\n"
        mdtext += "- [Conservation scores statistics](#con-stats)"
    if pos_eia_stats_dic and neg_eia_stats_dic:
        mdtext += "\n"
        mdtext += "- [Exon-intron region distribution](#eia-plot)\n"
        mdtext += "- [Exon-intron region statistics](#eia-stats)"
    if pos_tra_stats_dic and neg_tra_stats_dic:
        mdtext += "\n"
        mdtext += "- [Transcript region distribution](#tra-plot)\n"
        mdtext += "- [Transcript region statistics](#tra-stats)"
    if pos_rra_stats_dic and neg_rra_stats_dic:
        mdtext += "\n"
        mdtext += "- [Repeat region distribution](#rra-plot)\n"
        mdtext += "- [Repeat region statistics](#rra-stats)"
    if target_gbtc_dic and all_gbtc_dic:
        mdtext += "\n"
        mdtext += "- [Target gene biotype statistics](#gbt-stats)"
    if t2hc_dic and t2i_dic:
        mdtext += "\n"
        mdtext += "- [Target region overlap statistics](#tro-stats)"
    if add_feat_dic_list:
        mdtext += "\n"
        mdtext += "- [BED feature statistics](#bed-stats)\n"
        mdtext += "- [BED feature coverage distribution](#bed-plot)"
    mdtext += "\n&nbsp;\n"

    # Make general stats table.
    mdtext += """
## Training dataset statistics ### {#set-stats}

**Table:** Training dataset statistics regarding sequence lengths
(min, max, mean, and median length) in nucleotides (nt),
sequence complexity (mean Shannon entropy over all sequences in the set)
and nucleotide contents (A, C, G, U).

"""
    mdtext += "| Attribute | &nbsp; Positives &nbsp; | &nbsp; Negatives &nbsp; | \n"
    mdtext += "| :-: | :-: | :-: |\n"
    mdtext += "| # sites | %i | %i |\n" %(c_pos_out, c_neg_out)
    mdtext += "| min site length | %i | %i |\n" %(min(pos_len_list), min(neg_len_list))
    mdtext += "| max site length | %i | %i |\n" %(max(pos_len_list), max(neg_len_list))
    mdtext += "| mean site length | %.1f | %.1f |\n" %(statistics.mean(pos_len_list), statistics.mean(neg_len_list))
    mdtext += "| median site length | %i | %i |\n" %(statistics.median(pos_len_list), statistics.median(neg_len_list))
    mdtext += "| mean complexity | %.3f | %.3f |\n" %(statistics.mean(pos_entr_list), statistics.mean(neg_entr_list))
    mdtext += '| %A |' + " %.2f | %.2f |\n" %(pos_ntr_dic["A"], neg_ntr_dic["A"])
    mdtext += '| %C |' + " %.2f | %.2f |\n" %(pos_ntr_dic["C"], neg_ntr_dic["C"])
    mdtext += '| %G |' + " %.2f | %.2f |\n" %(pos_ntr_dic["G"], neg_ntr_dic["G"])
    mdtext += '| %U |' + " %.2f | %.2f |\n" %(pos_ntr_dic["U"], neg_ntr_dic["U"])
    mdtext += "\n&nbsp;\n&nbsp;\n"

    # Make site length distribution box plot.
    create_set_lengths_box_plot(pos_len_list, neg_len_list, lengths_plot_out,
                                theme=theme,
                                disable_title=True)
    lengths_plot_path = plots_folder + "/" + lengths_plot

    mdtext += """
## Site length distribution ### {#len-plot}

Lengths differences in the training dataset can arise in two cases:

- FASTA sequences of various lengths are given as input (--in)
- BED sites of various lengths are given as input (--in) and --mode 2 is set

Otherwise, all sequences (positives and negatives) are expected to have
more or less the same length. This is because --mode 1 or --mode 3 both
reduce the sites to a length of 1 before uniform extension is applied
(controlled by --seq-ext and --con-ext).
Some length differences can still occur though, e.g. if transcript
sequences are extracted close to transcript ends, or as negatives
are sampled randomly from a larger pool of initial negatives.
Note that the lowercase context sequence parts which can be added by
--con-ext are excluded from the HTML report statistics. Only the
uppercase sequence parts of each site (== whole site if --con-ext False)
contribute to the statistics in this report.
Lowercase context sequence parts (if --con-ext is set) are only
used for base pair calculation as well as during model training
(graphprot train) for profile prediction and motif generation.

"""
    mdtext += '<img src="' + lengths_plot_path + '" alt="Site length distribution"' + "\n"
    mdtext += 'title="Site length distribution" width="500" />' + "\n"
    mdtext += """

**Figure:** Site length distributions for the positive and negative dataset.

&nbsp;

"""
    # Make sequence complexity box plot.
    create_entropy_box_plot(pos_entr_list, neg_entr_list, entropy_plot_out,
                            theme=theme,
                            disable_title=True)
    entropy_plot_path = plots_folder + "/" + entropy_plot

    mdtext += """
## Sequence complexity distribution ### {#ent-plot}

The Shannon entropy is calculated for each sequence to measure
its information content (i.e., its complexity). A sequence with
equal amounts of all four nucleotides has an entropy value of 1.0
(highest possible). A sequence with equal amounts of two nucleotides
has an entropy value of 0.5. Finally, the lowest possible entropy is
achieved by a sequence which contains only one type of nucleotide.
Find the formula used to compute Shannon's entropy
[here](https://www.ncbi.nlm.nih.gov/pubmed/15215465) (see CE formula).


"""
    mdtext += '<img src="' + entropy_plot_path + '" alt="Sequence complexity distribution"' + "\n"
    mdtext += 'title="Sequence complexity distribution" width="500" />' + "\n"
    mdtext += """

**Figure:** Sequence complexity (Shannon entropy
computed for each sequence) distributions for the positive and
negative dataset.

&nbsp;

"""
    # Make di-nucleotide grouped bar plot.
    create_dint_ratios_grouped_bar_plot(pos_dintr_dic, neg_dintr_dic, dint_plot_out,
                                        theme=theme,
                                        disable_title=True)
    dint_plot_path = plots_folder + "/" + dint_plot

    mdtext += """
## Di-nucleotide distribution ### {#dint-plot}

Di-nucleotide percentages are shown for both the positive and negative dataset.

"""
    mdtext += '<img src="' + dint_plot_path + '" alt="Di-nucleotide distribution"' + "\n"
    mdtext += 'title="Di-nucleotide distribution" width="1000" />' + "\n"
    mdtext += """

**Figure:** Di-nucleotide percentages for the positive and negative dataset.

&nbsp;

"""
    # Make the k-mer tables.
    top3mertab = generate_top_kmer_md_table(pos_3mer_dic, neg_3mer_dic,
                                            top=kmer_top,
                                            val_type="p")
    top4mertab = generate_top_kmer_md_table(pos_4mer_dic, neg_4mer_dic,
                                            top=kmer_top,
                                            val_type="p")
    top5mertab = generate_top_kmer_md_table(pos_5mer_dic, neg_5mer_dic,
                                            top=kmer_top,
                                            val_type="p")
    mdtext += """
## Top k-mer statistics ### {#kmer-stats}

**Table:** Top %i 3-mers for the positive and negative set and their percentages in the respective sequence set. In case of uniform distribution with all 3-mers present, each 3-mer would have a percentage = 1.5625.

""" %(kmer_top)
    mdtext += top3mertab
    mdtext += "\n&nbsp;\n"

    mdtext += """
**Table:** Top %i 4-mers for the positive and negative set and their percentages in the respective sequence set. In case of uniform distribution with all 4-mers present, each 4-mer would have a percentage = 0.390625.

""" %(kmer_top)
    mdtext += top4mertab
    mdtext += "\n&nbsp;\n"

    mdtext += """
**Table:** Top %i 5-mers for the positive and negative set and their percentages in the respective sequence set. In case of uniform distribution with all 5-mers present, each 5-mer would have a percentage = 0.09765625.

""" %(kmer_top)
    mdtext += top5mertab
    mdtext += "\n&nbsp;\n&nbsp;\n"

    if pos_str_stats_dic and neg_str_stats_dic:
        # Checks.
        assert pos_str_stats_dic['seqlen_sum'], "unexpected total sequence length of 0 encountered"
        assert neg_str_stats_dic['seqlen_sum'], "unexpected total sequence length of 0 encountered"

        # Make structural elements bar plot.
        create_str_elem_grouped_bar_plot(pos_str_stats_dic, neg_str_stats_dic,
                                         str_elem_plot_out,
                                         theme=theme,
                                         disable_title=True)
        str_elem_plot_path = plots_folder + "/" + str_elem_plot

        mdtext += """
## Structural elements distribution ### {#str-elem-plot}

Mean position-wise probabilities of the different loop context structural elements are shown
for both the positive and negative dataset. U: unpaired, E: external loop, H: hairpin loop,
I: internal loop, M: multi-loop, S: paired.

"""
        mdtext += '<img src="' + str_elem_plot_path + '" alt="Structural elements distribution"' + "\n"
        mdtext += 'title="Structural elements distribution" width="650" />' + "\n"
        mdtext += """

**Figure:** Mean position-wise probabilities of different loop context structural elements for
the positive and negative dataset. U: unpaired, E: external loop, H: hairpin loop,
I: internal loop, M: multi-loop, S: paired.

&nbsp;

"""
        # Make base pair stats table.
        pos_bps_per_100nt = pos_str_stats_dic['bp_c'] / (pos_str_stats_dic['seqlen_sum'] / 100)
        neg_bps_per_100nt = neg_str_stats_dic['bp_c'] / (neg_str_stats_dic['seqlen_sum'] / 100)
        pos_mean_bp_p = pos_str_stats_dic['bp_p'][0]
        neg_mean_bp_p = neg_str_stats_dic['bp_p'][0]
        pos_mean_bp_stdev = pos_str_stats_dic['bp_p'][1]
        neg_mean_bp_stdev = neg_str_stats_dic['bp_p'][1]

        mdtext += """
## Secondary structure statistics ### {#bp-stats}

**Table:** Secondary structure (base pair + structural elements) statistics of
the generated training set. Mean probabilities p() are given together with standard deviations (+- ...).

"""
        mdtext += "| &nbsp; &nbsp; &nbsp; Attribute &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; Positives &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; Negatives &nbsp; &nbsp; &nbsp; &nbsp; | \n"
        mdtext += "| :-: | :-: | :-: |\n"
        mdtext += "| total sequence length | %i | %i |\n" %(pos_str_stats_dic['seqlen_sum'], neg_str_stats_dic['seqlen_sum'])
        mdtext += "| # base pairs | %i | %i |\n" %(pos_str_stats_dic['bp_c'], neg_str_stats_dic['bp_c'])
        mdtext += "| base pairs per 100 nt | %.1f | %.1f |\n" %(pos_bps_per_100nt, neg_bps_per_100nt)
        mdtext += "| # no-base-pair sites | %i | %i |\n" %(pos_str_stats_dic['nobpsites_c'], neg_str_stats_dic['nobpsites_c'])
        mdtext += "| mean p(base pair) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_mean_bp_p, pos_mean_bp_stdev, neg_mean_bp_p, neg_mean_bp_stdev)
        mdtext += "| mean p(paired) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_str_stats_dic['S'][0], pos_str_stats_dic['S'][1], neg_str_stats_dic['S'][0], neg_str_stats_dic['S'][1])
        mdtext += "| mean p(unpaired) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_str_stats_dic['U'][0], pos_str_stats_dic['U'][1], neg_str_stats_dic['U'][0], neg_str_stats_dic['U'][1])
        mdtext += "| mean p(external loop) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_str_stats_dic['E'][0], pos_str_stats_dic['E'][1], neg_str_stats_dic['E'][0], neg_str_stats_dic['E'][1])
        mdtext += "| mean p(hairpin loop) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_str_stats_dic['H'][0], pos_str_stats_dic['H'][1], neg_str_stats_dic['H'][0], neg_str_stats_dic['H'][1])
        mdtext += "| mean p(internal loop) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_str_stats_dic['I'][0], pos_str_stats_dic['I'][1], neg_str_stats_dic['I'][0], neg_str_stats_dic['I'][1])
        mdtext += "| mean p(multi loop) | %.4f (+-%.4f) | %.4f (+-%.4f) |\n" %(pos_str_stats_dic['M'][0], pos_str_stats_dic['M'][1], neg_str_stats_dic['M'][0], neg_str_stats_dic['M'][1])
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Conservation scores plots and stats.
    if pos_phastcons_stats_dic or pos_phylop_stats_dic:
        mdtext += """
## Conservation scores distribution ### {#con-plot}

Mean conservation scores with standard deviations are shown for the positive
and negative set.

"""
        # phastCons plot.
        if pos_phastcons_stats_dic and neg_phastcons_stats_dic:

            create_conservation_scores_bar_plot(pos_phastcons_stats_dic, neg_phastcons_stats_dic,
                                                phastcons_plot_out, "phastCons",
                                                disable_title=True,
                                                theme=theme)
            phastcons_plot_path = plots_folder + "/" + phastcons_plot
            mdtext += '<img src="' + phastcons_plot_path + '" alt="phastCons scores distribution"' + "\n"
            mdtext += 'title="phastCons scores distribution" width="400" />' + "\n"
            mdtext += """

**Figure:** Mean phastCons conservation score and standard deviation for the positive and negative dataset.

&nbsp;

"""
        # phyloP plot.
        if pos_phylop_stats_dic and neg_phylop_stats_dic:
            create_conservation_scores_bar_plot(pos_phylop_stats_dic, neg_phylop_stats_dic,
                                                phylop_plot_out, "phyloP",
                                                disable_title=True,
                                                theme=theme)
            phylop_plot_path = plots_folder + "/" + phylop_plot
            mdtext += '<img src="' + phylop_plot_path + '" alt="phyloP scores distribution"' + "\n"
            mdtext += 'title="phyloP scores distribution" width="400" />' + "\n"
            mdtext += """

**Figure:** Mean phyloP conservation score and standard deviation (before -1 .. 1 normalization) for the positive and negative dataset.

&nbsp;

"""

        mdtext += """
## Conservation scores statistics ### {#con-stats}

**Table:** Conservation scores statistics. Note that phyloP statistics are
calculated before normalization (normalizing values to -1 .. 1).

"""
        mdtext += "| &nbsp; &nbsp; Attribute &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; Positives &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; Negatives &nbsp; &nbsp; &nbsp; | \n"
        mdtext += "| :-: | :-: | :-: |\n"
        if pos_phastcons_stats_dic and neg_phastcons_stats_dic:
            pos_pc_zero_perc = "%.2f" % ((pos_phastcons_stats_dic["zero_pos"] / pos_phastcons_stats_dic["total_pos"]) * 100)
            neg_pc_zero_perc = "%.2f" % ((neg_phastcons_stats_dic["zero_pos"] / neg_phastcons_stats_dic["total_pos"]) * 100)
            mdtext += "| # phastCons scores | %i | %i |\n" %(pos_phastcons_stats_dic['total_pos'], neg_phastcons_stats_dic['total_pos'])
            mdtext += "| # zero scores | %i | %i |\n" %(pos_phastcons_stats_dic['zero_pos'], neg_phastcons_stats_dic['zero_pos'])
            mdtext += '| % zero scores |' + " %s | %s |\n" %(pos_pc_zero_perc, neg_pc_zero_perc)
            mdtext += "| min score | %s | %s |\n" %(str(pos_phastcons_stats_dic['min']), str(neg_phastcons_stats_dic['min']))
            mdtext += "| max score | %s | %s |\n" %(str(pos_phastcons_stats_dic['max']), str(neg_phastcons_stats_dic['max']))
            mdtext += "| mean score | %.3f (+-%.3f) | %.3f (+-%.3f) |\n" %(pos_phastcons_stats_dic['mean'], pos_phastcons_stats_dic['stdev'], neg_phastcons_stats_dic['mean'], neg_phastcons_stats_dic['stdev'])
        if pos_phylop_stats_dic and neg_phylop_stats_dic:
            pos_pp_zero_perc = "%.2f" % ((pos_phylop_stats_dic["zero_pos"] / pos_phylop_stats_dic["total_pos"]) * 100)
            neg_pp_zero_perc = "%.2f" % ((neg_phylop_stats_dic["zero_pos"] / neg_phylop_stats_dic["total_pos"]) * 100)
            mdtext += "| # phyloP scores | %i | %i |\n" %(pos_phylop_stats_dic['total_pos'], neg_phylop_stats_dic['total_pos'])
            mdtext += "| # zero scores | %i | %i |\n" %(pos_phylop_stats_dic['zero_pos'], neg_phylop_stats_dic['zero_pos'])
            mdtext += '| % zero scores |' + " %s | %s |\n" %(pos_pp_zero_perc, neg_pp_zero_perc)
            mdtext += "| min score | %s | %s |\n" %(str(pos_phylop_stats_dic['min']), str(neg_phylop_stats_dic['min']))
            mdtext += "| max score | %s | %s |\n" %(str(pos_phylop_stats_dic['max']), str(neg_phylop_stats_dic['max']))
            mdtext += "| mean score | %.3f (+-%.3f) | %.3f (+-%.3f) |\n" %(pos_phylop_stats_dic['mean'], pos_phylop_stats_dic['stdev'], neg_phylop_stats_dic['mean'], neg_phylop_stats_dic['stdev'])
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Exon-intron region plots and stats.
    if pos_eia_stats_dic and neg_eia_stats_dic:
        mdtext += """
## Exon-intron region distribution ### {#eia-plot}

Distribution of exon and intron regions for the positive and negative set.

"""
        # EIA plot.
        create_reg_annot_grouped_bar_plot(pos_eia_stats_dic, neg_eia_stats_dic, eia_plot_out,
                                          ["E", "I", "N"],
                                          perc=True, theme=theme)
        eia_plot_path = plots_folder + "/" + eia_plot
        mdtext += '<img src="' + eia_plot_path + '" alt="Exon-intron region distribution"' + "\n"
        mdtext += 'title="Exon-intron region distribution" width="550" />' + "\n"
        mdtext += """
**Figure:** Percentages of exon (E) and intron (I) regions for the positive and negative set.
If --eia-n is set, also include regions not covered by introns or exons (N).

&nbsp;

## Exon-intron region statistics ### {#eia-stats}

**Table:** Exon-intron region statistics for the positive and negative set.
If --eia-ib is set, also include statistics for sites containing intron
5' (F) and intron 3' (T) ends.

"""
        # EIA stats.
        if "F" in pos_eia_stats_dic:
            pos_perc_f_sites = "%.2f" % ((pos_eia_stats_dic['F'] / c_pos_out)*100) + " %"
            pos_perc_t_sites = "%.2f" % ((pos_eia_stats_dic['T'] / c_pos_out)*100) + " %"
            neg_perc_f_sites = "%.2f" % ((neg_eia_stats_dic['F'] / c_neg_out)*100) + " %"
            neg_perc_t_sites = "%.2f" % ((neg_eia_stats_dic['T'] / c_neg_out)*100) + " %"
        pos_perc_e = "%.2f" % ((pos_eia_stats_dic['E'] / pos_eia_stats_dic['total_pos'])*100)
        pos_perc_i = "%.2f" % ((pos_eia_stats_dic['I'] / pos_eia_stats_dic['total_pos'])*100)
        neg_perc_e = "%.2f" % ((neg_eia_stats_dic['E'] / neg_eia_stats_dic['total_pos'])*100)
        neg_perc_i = "%.2f" % ((neg_eia_stats_dic['I'] / neg_eia_stats_dic['total_pos'])*100)
        if "N" in pos_eia_stats_dic:
            pos_perc_n = "%.2f" % ((pos_eia_stats_dic['N'] / pos_eia_stats_dic['total_pos'])*100)
            neg_perc_n = "%.2f" % ((neg_eia_stats_dic['N'] / neg_eia_stats_dic['total_pos'])*100)
        mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Positives &nbsp; | &nbsp; Negatives &nbsp; | \n"
        mdtext += "| :-: | :-: | :-: |\n"
        mdtext += '| % E |' + " %s | %s |\n" %(pos_perc_e, neg_perc_e)
        mdtext += '| % I |' + " %s | %s |\n" %(pos_perc_i, neg_perc_i)
        if "N" in pos_eia_stats_dic:
            mdtext += '| % N |' + " %s | %s |\n" %(pos_perc_n, neg_perc_n)
        if "F" in pos_eia_stats_dic:
            mdtext += "| F sites | %i (%s) | %i (%s) |\n" %(pos_eia_stats_dic['F'], pos_perc_f_sites, neg_eia_stats_dic['F'], neg_perc_f_sites)
            mdtext += "| T sites | %i (%s) | %i (%s) |\n" %(pos_eia_stats_dic['T'], pos_perc_t_sites, neg_eia_stats_dic['T'], neg_perc_t_sites)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Transcript region plots and stats.
    if pos_tra_stats_dic and neg_tra_stats_dic:
        mdtext += """
## Transcript region distribution ### {#tra-plot}

Distribution of transcript regions for the positive and negative set.

"""
        # TRA plot.
        create_reg_annot_grouped_bar_plot(pos_tra_stats_dic, neg_tra_stats_dic, tra_plot_out,
                                          ["F", "C", "T", "N"],
                                          perc=True, theme=theme)
        tra_plot_path = plots_folder + "/" + tra_plot
        mdtext += '<img src="' + tra_plot_path + '" alt="Transcript region distribution"' + "\n"
        mdtext += 'title="Transcript region distribution" width="550" />' + "\n"
        mdtext += """
**Figure:** Percentages of 5'UTR (F), CDS (C), and 3'UTR (T) positions as well as
positions not covered by these transcript regions (N) for the positive and negative set.

&nbsp;

## Transcript region statistics ### {#tra-stats}

**Table:** Transcript region statistics for the positive and negative set.
Percentages of positions covered by 5'UTR (F), CDS (C), 3'UTR (T), or non
of these regions (N) are given.
If --tra-codons is set, also include statistics for start codons (S) and
stop codons (E) (sites which contain these).
If --tra-borders is set, also include statistics for transcript starts (A),
 transcript ends (Z), exon borders (B) (sites which contain these).

"""
        # TRA stats.
        pos_perc_f = "%.2f" % ((pos_tra_stats_dic['F'] / pos_tra_stats_dic['total_pos'])*100)
        pos_perc_c = "%.2f" % ((pos_tra_stats_dic['C'] / pos_tra_stats_dic['total_pos'])*100)
        pos_perc_t = "%.2f" % ((pos_tra_stats_dic['T'] / pos_tra_stats_dic['total_pos'])*100)
        pos_perc_n = "%.2f" % ((pos_tra_stats_dic['N'] / pos_tra_stats_dic['total_pos'])*100)
        neg_perc_f = "%.2f" % ((neg_tra_stats_dic['F'] / neg_tra_stats_dic['total_pos'])*100)
        neg_perc_c = "%.2f" % ((neg_tra_stats_dic['C'] / neg_tra_stats_dic['total_pos'])*100)
        neg_perc_t = "%.2f" % ((neg_tra_stats_dic['T'] / neg_tra_stats_dic['total_pos'])*100)
        neg_perc_n = "%.2f" % ((neg_tra_stats_dic['N'] / neg_tra_stats_dic['total_pos'])*100)
        mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Positives &nbsp; | &nbsp; Negatives &nbsp; | \n"
        mdtext += "| :-: | :-: | :-: |\n"
        mdtext += '| % F |' + " %s | %s |\n" %(pos_perc_f, neg_perc_f)
        mdtext += '| % C |' + " %s | %s |\n" %(pos_perc_c, neg_perc_c)
        mdtext += '| % T |' + " %s | %s |\n" %(pos_perc_t, neg_perc_t)
        mdtext += '| % N |' + " %s | %s |\n" %(pos_perc_n, neg_perc_n)
        # Start stop codon annotations.
        if "S" in pos_tra_stats_dic:
            pos_perc_s_sites = "%.2f" % ((pos_tra_stats_dic['S'] / c_pos_out)*100) + " %"
            pos_perc_e_sites = "%.2f" % ((pos_tra_stats_dic['E'] / c_pos_out)*100) + " %"
            neg_perc_s_sites = "%.2f" % ((neg_tra_stats_dic['S'] / c_neg_out)*100) + " %"
            neg_perc_e_sites = "%.2f" % ((neg_tra_stats_dic['E'] / c_neg_out)*100) + " %"
            mdtext += "| S sites | %i (%s) | %i (%s) |\n" %(pos_tra_stats_dic['S'], pos_perc_s_sites, neg_tra_stats_dic['S'], neg_perc_s_sites)
            mdtext += "| E sites | %i (%s) | %i (%s) |\n" %(pos_tra_stats_dic['E'], pos_perc_e_sites, neg_tra_stats_dic['E'], neg_perc_e_sites)
        # Border annotations.
        if "A" in pos_tra_stats_dic:
            pos_perc_a_sites = "%.2f" % ((pos_tra_stats_dic['A'] / c_pos_out)*100) + " %"
            pos_perc_b_sites = "%.2f" % ((pos_tra_stats_dic['B'] / c_pos_out)*100) + " %"
            pos_perc_z_sites = "%.2f" % ((pos_tra_stats_dic['Z'] / c_pos_out)*100) + " %"
            neg_perc_a_sites = "%.2f" % ((neg_tra_stats_dic['A'] / c_neg_out)*100) + " %"
            neg_perc_b_sites = "%.2f" % ((neg_tra_stats_dic['B'] / c_neg_out)*100) + " %"
            neg_perc_z_sites = "%.2f" % ((neg_tra_stats_dic['Z'] / c_neg_out)*100) + " %"
            mdtext += "| A sites | %i (%s) | %i (%s) |\n" %(pos_tra_stats_dic['A'], pos_perc_a_sites, neg_tra_stats_dic['A'], neg_perc_a_sites)
            mdtext += "| B sites | %i (%s) | %i (%s) |\n" %(pos_tra_stats_dic['B'], pos_perc_b_sites, neg_tra_stats_dic['B'], neg_perc_b_sites)
            mdtext += "| Z sites | %i (%s) | %i (%s) |\n" %(pos_tra_stats_dic['Z'], pos_perc_z_sites, neg_tra_stats_dic['Z'], neg_perc_z_sites)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Repeat region plots and stats.
    if pos_rra_stats_dic and neg_rra_stats_dic:
        mdtext += """
## Repeat region distribution ### {#rra-plot}

Distribution of repeat regions for the positive and negative set. Repeat
regions are annotated in the .2bit genomic sequences file as lowercase
sequences. These regions were identified by RepeatMasker and Tandem Repeats
Finder (with period of 12 or less).

"""
        # RRA plot.
        create_reg_annot_grouped_bar_plot(pos_rra_stats_dic, neg_rra_stats_dic, rra_plot_out,
                                          ["R", "N"],
                                          perc=True, theme=theme)
        rra_plot_path = plots_folder + "/" + rra_plot
        mdtext += '<img src="' + rra_plot_path + '" alt="Repeat region distribution"' + "\n"
        mdtext += 'title="Repeat region distribution" width="550" />' + "\n"
        mdtext += """
**Figure:** Percentages of repeat (R) and no-repeat (N) regions for the
positive and negative set.

&nbsp;

## Repeat region statistics ### {#rra-stats}

**Table:** Repeat region statistics for the positive and negative set.
Percentages of positive and negative regions covered by repeat (R)
 and non-repeat (N) regions are given.

"""
        # RRA stats.
        pos_perc_r = "%.2f" % ((pos_rra_stats_dic['R'] / pos_rra_stats_dic['total_pos'])*100)
        pos_perc_n = "%.2f" % ((pos_rra_stats_dic['N'] / pos_rra_stats_dic['total_pos'])*100)
        neg_perc_r = "%.2f" % ((neg_rra_stats_dic['R'] / neg_rra_stats_dic['total_pos'])*100)
        neg_perc_n = "%.2f" % ((neg_rra_stats_dic['N'] / neg_rra_stats_dic['total_pos'])*100)

        mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Positives &nbsp; | &nbsp; Negatives &nbsp; |\n"
        mdtext += "| :-: | :-: | :-: |\n"
        mdtext += '| % R |' + " %s | %s |\n" %(pos_perc_r, neg_perc_r)
        mdtext += '| % N |' + " %s | %s |\n" %(pos_perc_n, neg_perc_n)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Target gene biotype count stats.
    if target_gbtc_dic and all_gbtc_dic:
        mdtext += """
## Target gene biotype statistics ### {#gbt-stats}

**Table:** Target gene biotype counts for the positive set and their percentages
(count normalized by total count for the respective gene biotype).

"""
        mdtext += "| &nbsp; Gene biotype &nbsp; | &nbsp; Target count &nbsp; | &nbsp; Total count &nbsp; | &nbsp; Percentage &nbsp; | \n"
        mdtext += "| :-: | :-: | :-: | :-: |\n"
        unit = " %"
        for bt, target_c in sorted(target_gbtc_dic.items(), key=lambda item: item[1], reverse=True):
            all_c = all_gbtc_dic[bt]
            perc_c = "%.2f" % ((target_c / all_c) * 100)
            mdtext += "| %s | %i | %i | %s%s |\n" %(bt, target_c, all_c, perc_c, unit)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    if t2hc_dic and t2i_dic:
        mdtext += """
## Target region overlap statistics ### {#tro-stats}

**Table:** Target region overlap statistics, showing the top %i targeted
regions (transcript or genes), with the # overlaps == # of positive sites
overlapping with the region.

""" %(target_top)

        if dataset_type == "t":
            mdtext += "| &nbsp; # overlaps &nbsp; | &nbsp; Transcript ID &nbsp; | &nbsp; &nbsp; Transcript biotype &nbsp; &nbsp; | &nbsp; Gene ID &nbsp; | &nbsp; Gene name &nbsp; | &nbsp; &nbsp; Gene biotype &nbsp; &nbsp; | \n"
            mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: |\n"
            i = 0
            for tr_id, ol_c in sorted(t2hc_dic.items(), key=lambda item: item[1], reverse=True):
                i += 1
                if i > target_top:
                    break
                tr_bt = t2i_dic[tr_id][0]
                gene_id = t2i_dic[tr_id][1]
                gene_name = t2i_dic[tr_id][2]
                gene_bt = t2i_dic[tr_id][3]
                mdtext += "| %i | %s | %s |  %s | %s | %s |\n" %(ol_c, tr_id, tr_bt, gene_id, gene_name, gene_bt)
            mdtext += "| ... | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |\n"
            mdtext += "\n&nbsp;\n&nbsp;\n"

        elif dataset_type == "g":
            mdtext += "| &nbsp; # overlaps &nbsp; | &nbsp; Gene ID &nbsp; | &nbsp; Gene name &nbsp; | &nbsp; &nbsp; Gene biotype &nbsp; &nbsp; | \n"
            mdtext += "| :-: | :-: | :-: | :-: |\n"
            i = 0
            for gene_id, ol_c in sorted(t2hc_dic.items(), key=lambda item: item[1], reverse=True):
                i += 1
                if i > target_top:
                    break
                gene_name = t2i_dic[gene_id][0]
                gene_bt = t2i_dic[gene_id][1]
                mdtext += "| %i | %s | %s |  %s |\n" %(ol_c, gene_id, gene_name, gene_bt)
            mdtext += "| ... | &nbsp; | &nbsp; |  &nbsp; |\n"
            mdtext += "\n&nbsp;\n&nbsp;\n"

    # Additional BED annotations.
    if add_feat_dic_list:
        mdtext += """
## BED feature statistics ### {#bed-stats}

Additional BED annotation feature statistics (from --feat-in table) for the
positive and negative dataset.

"""
        pos_cov_dic = {}
        neg_cov_dic = {}
        for i in range(0, len(add_feat_dic_list) - 1, 2):
            pos_stats_dic = add_feat_dic_list[i]
            neg_stats_dic = add_feat_dic_list[i+1]
            feat_id = pos_stats_dic["feat_id"]
            feat_type = pos_stats_dic["feat_type"]
            pos_total_pos = pos_stats_dic["total_pos"]
            neg_total_pos = neg_stats_dic["total_pos"]
            pos_perc_zero_sites = "%.2f" % ((pos_stats_dic['zero_sites'] / pos_stats_dic['total_sites'])*100) + " %"
            neg_perc_zero_sites = "%.2f" % ((neg_stats_dic['zero_sites'] / neg_stats_dic['total_sites'])*100) + " %"
            if feat_type == "C":
                pos_c_0 = pos_stats_dic["0"]
                pos_c_1 = pos_stats_dic["1"]
                neg_c_0 = neg_stats_dic["0"]
                neg_c_1 = neg_stats_dic["1"]
                pos_perc_0 = "%.2f" % ((pos_c_0 / pos_total_pos)*100) + " %"
                pos_perc_1 = "%.2f" % ((pos_c_1 / pos_total_pos)*100) + " %"
                neg_perc_0 = "%.2f" % ((neg_c_0 / neg_total_pos)*100) + " %"
                neg_perc_1 = "%.2f" % ((neg_c_1 / neg_total_pos)*100) + " %"
            else:
                pos_mean = pos_stats_dic["mean"]
                pos_stdev = pos_stats_dic["stdev"]
                neg_mean = neg_stats_dic["mean"]
                neg_stdev = neg_stats_dic["stdev"]
                pos_c_0 = pos_stats_dic["zero_pos"]
                neg_c_0 = neg_stats_dic["zero_pos"]
                pos_c_1 = pos_total_pos - pos_c_0
                neg_c_1 = neg_total_pos - neg_c_0
                pos_perc_0 = "%.2f" % ((pos_c_0 / pos_total_pos)*100) + " %"
                pos_perc_1 = "%.2f" % ((pos_c_1 / pos_total_pos)*100) + " %"
                neg_perc_0 = "%.2f" % ((neg_c_0 / neg_total_pos)*100) + " %"
                neg_perc_1 = "%.2f" % ((neg_c_1 / neg_total_pos)*100) + " %"

            # Store feature coverage (percentage of positions overlapping).
            pos_feat_cov = (pos_c_1 / pos_total_pos) * 100
            neg_feat_cov = (neg_c_1 / neg_total_pos) * 100
            pos_cov_dic[feat_id] = pos_feat_cov
            neg_cov_dic[feat_id] = neg_feat_cov

            mdtext += """
### BED annotation file feature \"%s\" statistics

""" %(feat_id)

            if feat_type == "C":
                mdtext += """

**Table:** BED feature region length + score statistics for the
positive and negative set.
Feature type is one-hot encoding, i.e., every overlapping position
gets a 1 assigned, every not overlapping position a 0.

"""
            else:
                mdtext += """

**Table:** BED feature region length + score statistics for the
positive and negative set.
Feature type is numerical, i.e., every position gets the score of the
overlapping feature region assigned. In case of no feature region overlap,
the position gets a score of 0.

"""
            mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Positives &nbsp; | &nbsp; Negatives &nbsp; |\n"
            mdtext += "| :-: | :-: | :-: |\n"
            mdtext += "| mean length | %.2f (+-%.2f) | %.2f (+-%.2f) |\n" %(pos_stats_dic["mean_l"], pos_stats_dic["stdev_l"], neg_stats_dic["mean_l"], neg_stats_dic["stdev_l"])
            mdtext += "| median length | %i | %i |\n" %(pos_stats_dic["median_l"], neg_stats_dic["median_l"])
            mdtext += "| min length | %i | %i |\n" %(pos_stats_dic["min_l"], neg_stats_dic["min_l"])
            mdtext += "| max length | %i | %i |\n" %(pos_stats_dic["max_l"], neg_stats_dic["max_l"])
            if feat_type == "C":
                mdtext += "| # total positions | %i | %i |\n" %(pos_total_pos, neg_total_pos)
                mdtext += "| # 0 positions | %i (%s) | %i (%s) |\n" %(pos_c_0, pos_perc_0, neg_c_0, neg_perc_0)
                mdtext += "| # 1 positions | %i (%s) | %i (%s) |\n" %(pos_c_1, pos_perc_1, neg_c_1, neg_perc_1)
                mdtext += '| % all-zero sites |' + " %s | %s |\n" %(pos_perc_zero_sites, neg_perc_zero_sites)
            else:
                mdtext += "| # total positions | %i | %i |\n" %(pos_total_pos, neg_total_pos)
                mdtext += "| # 0 positions | %i (%s) | %i (%s) |\n" %(pos_c_0, pos_perc_0, neg_c_0, neg_perc_0)
                mdtext += "| # non-0 positions | %i (%s) | %i (%s) |\n" %(pos_c_1, pos_perc_1, neg_c_1, neg_perc_1)
                mdtext += '| % all-zero sites |' + " %s | %s |\n" %(pos_perc_zero_sites, neg_perc_zero_sites)
                mdtext += "| mean score | %.3f (+-%.3f) | %.3f (+-%.3f) |\n" %(pos_mean, pos_stdev, neg_mean, neg_stdev)
            mdtext += "\n&nbsp;\n&nbsp;\n"

        # Create additional BED features coverage plot.
        mdtext += """
## BED feature coverage distribution ### {#bed-plot}

Additional BED feature coverage distributions for the
positive and negative dataset.

"""
        create_train_set_bed_feat_cov_plot(pos_cov_dic, neg_cov_dic,
                                           bed_cov_plot_out,
                                           theme=args.theme)
        bed_cov_plot_path = plots_folder + "/" + bed_cov_plot
        mdtext += '<img src="' + bed_cov_plot_path + '" alt="BED feature coverage distribution"' + "\n"
        mdtext += 'title="BED feature coverage distribution" width="800" />' + "\n"
        mdtext += """
**Figure:** Additional BED feature coverage distributions for the
positive and negative dataset. Feature coverage means how much
percent of the positive or negative regions are covered by the
respective BED feature (i.e., overlap with it). The BED feature
IDs from --feat-in are given on the y-axis, their coverage on the
x-axis.

&nbsp;

"""

    print("Generate HTML report ... ")

    # Convert mdtext to html.
    md2html = markdown(mdtext, extensions=['attr_list', 'tables'])

    #OUTMD = open(md_out,"w")
    #OUTMD.write("%s\n" %(mdtext))
    #OUTMD.close()

    OUTHTML = open(html_out,"w")
    OUTHTML.write("%s\n" %(md2html))
    OUTHTML.close()


################################################################################

def create_train_set_bed_feat_cov_plot(pos_cov_dic, neg_cov_dic, out_plot,
                                       theme=1):
    """
    Create a grouped bar plot, showing the coverage for each BED feature
    from --feat-in over the positive and negative set. Coverage means
    how much percentage of the positive or negative regions are covered
    by the BED feature (== overlap with it).
    Input dictionaries for positives (pos_cov_dic) and negatives
    (neg_cov_dic) store for each feature ID (key) the coverage of
    the feature in percent (value).
    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    MV colors:
    #69e9f6, #f154b2

    """

    # Checker.
    assert pos_cov_dic, "given dictionary pos_cov_dic empty"
    assert neg_cov_dic, "given dictionary neg_cov_dic empty"
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'feat_id': [], 'perc': []}

    for feat_id in pos_cov_dic:
        data['set'].append(pos_label)
        data['feat_id'].append(feat_id)
        data['perc'].append(pos_cov_dic[feat_id])
    for feat_id in neg_cov_dic:
        data['set'].append(neg_label)
        data['feat_id'].append(feat_id)
        data['perc'].append(neg_cov_dic[feat_id])
    df = pd.DataFrame (data, columns = ['set','feat_id', 'perc'])

    # Scale height depending on # of features.
    c_ids = len(pos_cov_dic)
    fheight = 1.5 * c_ids

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        g = sns.catplot(x="perc", y="feat_id", hue="set", data=df,
                        kind="bar", palette=["#69e9f6", "#f154b2"],
                        edgecolor="lightgrey",
                        legend=False)
        g.fig.set_figwidth(15)
        g.fig.set_figheight(fheight)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_xlabel("Feature coverage (%)",fontsize=20)
        ax[0,0].set(ylabel=None)
        ax[0,0].tick_params(axis='x', labelsize=16)
        ax[0,0].tick_params(axis='y', labelsize=20)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=16)
        g.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})

        g = sns.catplot(x="perc", y="feat_id", hue="set", data=df,
                        kind="bar", palette=["blue", "darkblue"],
                        edgecolor="#fcc826",
                        legend=False)
        g.fig.set_figwidth(15)
        g.fig.set_figheight(fheight)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_xlabel("Percentage (%)",fontsize=20)
        ax[0,0].set(ylabel=None)
        ax[0,0].tick_params(axis='x', labelsize=16)
        ax[0,0].tick_params(axis='y', labelsize=20)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=16, framealpha=0)
        g.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def create_reg_annot_grouped_bar_plot(pos_ra_dic, neg_ra_dic, out_plot,
                                      plot_labels,
                                      perc=False,
                                      theme=1):
    """
    Create a bar plot for region labels, given plot_labels to define what
    counts to plot. If perc=True, look for "total_pos" dictionary entry
    to normalize counts and plot percentages.

    Input dictionary has following keys:
    labels, total_pos

    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    MV colors:
    #69e9f6, #f154b2

    """

    # Checker.
    assert pos_ra_dic, "given dictionary pos_ra_dic empty"
    assert neg_ra_dic, "given dictionary neg_ra_dic empty"
    assert plot_labels, "given labels to plot list empty"
    if perc:
        assert pos_ra_dic["total_pos"], "total_pos key missing in pos_ra_dic"
        assert neg_ra_dic["total_pos"], "total_pos key missing in neg_ra_dic"
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'label': [], 'count': []}
    for l in pos_ra_dic:
        if l in plot_labels:
            lc = pos_ra_dic[l]
            if perc:
                lc = (lc / pos_ra_dic["total_pos"]) * 100
            data['set'].append(pos_label)
            data['label'].append(l)
            data['count'].append(lc)
    for l in neg_ra_dic:
        if l in plot_labels:
            lc = neg_ra_dic[l]
            if perc:
                lc = (lc / neg_ra_dic["total_pos"]) * 100
            data['set'].append(neg_label)
            data['label'].append(l)
            data['count'].append(lc)

    df = pd.DataFrame (data, columns = ['set','count', 'label'])
    y_label = "# positions"
    if perc:
        y_label = "% positions"
    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        g = sns.catplot(x="label", y="count", hue="set", data=df,
                        kind="bar", palette=["#69e9f6", "#f154b2"],
                        edgecolor="lightgrey",
                        legend=False)
        g.fig.set_figwidth(8)
        g.fig.set_figheight(4)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_ylabel(y_label,fontsize=22)
        ax[0,0].set(xlabel=None)
        ax[0,0].tick_params(axis='x', labelsize=22)
        ax[0,0].tick_params(axis='y', labelsize=17)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=17)
        g.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        g = sns.catplot(x="label", y="count", hue="set", data=df,
                        kind="bar", palette=["blue", "darkblue"],
                        edgecolor="#fcc826",
                        legend=False)
        g.fig.set_figwidth(8)
        g.fig.set_figheight(4)
        # Modify axes.
        ax = g.axes
        ax[0,0].set_ylabel(y_label,fontsize=22)
        ax[0,0].set(xlabel=None)
        ax[0,0].tick_params(axis='x', labelsize=22)
        ax[0,0].tick_params(axis='y', labelsize=17)
        # Add legend at specific position.
        plt.legend(loc=(1.01, 0.4), fontsize=17, framealpha=0)
        g.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def create_conservation_scores_bar_plot(pos_con_dic, neg_con_dic, out_plot,
                                        con_type,
                                        disable_title=False,
                                        theme=1):
    """
    Create a bar plot, showing the mean conservation score with standard
    deviation error bar in the positive and negative set.

    Input dictionary has following keys:
    mean, stdev, zero_pos, total_pos

    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    MV colors:
    #69e9f6, #f154b2

    """

    # Checker.
    assert pos_con_dic, "given dictionary pos_con_dic empty"
    assert neg_con_dic, "given dictionary neg_con_dic empty"
    # Make pandas dataframe.
    pos_label = "Positives"
    neg_label = "Negatives"
    data = {'set': [], 'mean': [], 'stdev': []}
    data['set'].append(pos_label)
    data['mean'].append(pos_con_dic['mean'])
    data['stdev'].append(pos_con_dic['stdev'])
    data['set'].append(neg_label)
    data['mean'].append(neg_con_dic['mean'])
    data['stdev'].append(neg_con_dic['stdev'])
    df = pd.DataFrame (data, columns = ['set','mean', 'stdev'])
    y_label = "Mean " + con_type + " score"
    set_title = con_type + " scores distribution"
    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.barplot(x="set", y="mean", data=df, yerr=df['stdev'], ecolor="darkgrey",
                        palette=["#69e9f6", "#f154b2"],
                        edgecolor="lightgrey")
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=14)
        if not disable_title:
            ax.axes.set_title(set_title, fontsize=20)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.barplot(x="set", y="mean", data=df, yerr=df['stdev'], ecolor="#fcc826",
                        palette=["blue", "darkblue"],
                        edgecolor="#fcc826")
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=14)
        if not disable_title:
            ax.axes.set_title(set_title, fontsize=20)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def generate_top_kmer_md_table(pos_kmer_dic, neg_kmer_dic,
                               top=5,
                               val_type="c"):
    """
    Given k-mer count dictionaries for positive and negative set, generate
    a markdown table with top 5 k-mers (sorted by decending dictionary value).

    val_type:
    Specify type of stored dictionary value.
    c : count (count of k-mer)
    r : ratio (k-mer count / total k-mer count)
    p : percentage ( (k-mer count / total k-mer count) * 100)

    """
    assert pos_kmer_dic, "given dictionary pos_kmer_dic empty"
    assert neg_kmer_dic, "given dictionary neg_kmer_dic empty"
    assert re.search("^[c|p|r]$", val_type), "invalid val_type given"
    # Get size of k.
    k = 0
    for kmer in pos_kmer_dic:
        k = len(kmer)
        break
    # Expected kmer number.
    exp_kmer_nr = pow(4,k)
    pos_kmer_nr = 0
    neg_kmer_nr = 0
    for kmer in pos_kmer_dic:
        kc = pos_kmer_dic[kmer]
        if kc:
            pos_kmer_nr += 1
    for kmer in neg_kmer_dic:
        kc = neg_kmer_dic[kmer]
        if kc:
            neg_kmer_nr += 1
    pos_kmer_perc = "%.2f " %((pos_kmer_nr / exp_kmer_nr) * 100) + " %"
    neg_kmer_perc = "%.2f " %((neg_kmer_nr / exp_kmer_nr) * 100) + " %"
    # Adjust decimal places based on k-mer size.
    dc_p = 2
    dc_r = 4
    if k > 3:
        for i in range(k-3):
            dc_p += 1
            dc_r += 1
    dc_p_str = "%."+str(dc_p)+"f"
    dc_r_str = "%."+str(dc_r)+"f"
    add_ch = ""
    if val_type == "p":
        add_ch = " %"
        # Format percentage to two decimal places.
        for kmer in pos_kmer_dic:
            new_v = dc_p_str % pos_kmer_dic[kmer]
            pos_kmer_dic[kmer] = new_v
        for kmer in neg_kmer_dic:
            new_v = dc_p_str % neg_kmer_dic[kmer]
            neg_kmer_dic[kmer] = new_v
    elif val_type == "r":
        # Format percentage to four decimal places.
        for kmer in pos_kmer_dic:
            new_v = dc_r_str % pos_kmer_dic[kmer]
            pos_kmer_dic[kmer] = new_v
        for kmer in neg_kmer_dic:
            new_v = dc_r_str % neg_kmer_dic[kmer]
            neg_kmer_dic[kmer] = new_v

    # Get top j k-mers.
    i = 0
    pos_topk_list = []

    for kmer, v in sorted(pos_kmer_dic.items(), key=lambda item: item[1], reverse=True):
        i += 1
        if i > top:
            break
        pos_topk_list.append(kmer)
    i = 0
    neg_topk_list = []
    for kmer, v in sorted(neg_kmer_dic.items(), key=lambda item: item[1], reverse=True):
        i += 1
        if i > top:
            break
        neg_topk_list.append(kmer)

    # Generate markdown table.
    mdtable = "| Rank | &nbsp; &nbsp; Positives &nbsp; &nbsp; | &nbsp; &nbsp; Negatives &nbsp; &nbsp;|\n"
    mdtable += "| :-: | :-: | :-: |\n"
    for i in range(top):
        pos_kmer = pos_topk_list[i]
        neg_kmer = neg_topk_list[i]
        pos = i + 1
        mdtable += "| %i | %s (%s%s) | %s (%s%s) |\n" %(pos, pos_kmer, str(pos_kmer_dic[pos_kmer]), add_ch, neg_kmer, str(neg_kmer_dic[neg_kmer]), add_ch)

    mdtable += "| ... | &nbsp; | &nbsp; |\n"
    mdtable += "| # distinct k-mers | %i (%s) | %i (%s) |\n" %(pos_kmer_nr, pos_kmer_perc, neg_kmer_nr, neg_kmer_perc)

    # Return markdown table.
    return mdtable


################################################################################

def get_kmer_dic(k,
                 rna=False):
    """
    Return a dictionary of k-mers. By default, DNA alphabet is used (ACGT).
    Value for each k-mer key is set to 0.

    rna:
    Use RNA alphabet (ACGU).

    >>> get_kmer_dic(1)
    {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    >>> get_kmer_dic(2, rna=True)
    {'AA': 0, 'AC': 0, 'AG': 0, 'AU': 0, 'CA': 0, 'CC': 0, 'CG': 0, 'CU': 0, 'GA': 0, 'GC': 0, 'GG': 0, 'GU': 0, 'UA': 0, 'UC': 0, 'UG': 0, 'UU': 0}

    """
    # Check.
    assert k, "invalid k given"
    assert k > 0, "invalid k given"
    # Dictionary.
    mer2c_dic = {}
    # Alphabet.
    nts = ["A", "C", "G", "T"]
    if rna:
        nts = ["A", "C", "G", "U"]
    # Recursive k-mer dictionary creation.
    def fill(i, seq, mer2c_dic):
        if i:
            for nt in nts:
                fill(i-1, seq+nt, mer2c_dic)
        else:
            mer2c_dic[seq] = 0
    fill(k, "", mer2c_dic)
    return mer2c_dic


################################################################################

def get_min_hit_end_distance(l1, l2):
    """
    Given two lists of kmer hit ends on a sequence (from get_kmer_hit_ends()),
    return the minimum distance between two hit ends in the two lists.
    In other words, the closest positions in both lists.

    >>> l1 = [2,10,20]
    >>> l2 = [12,15,30]
    >>> get_min_hit_end_distance(l1,l2)
    2

    """
    assert l1, "l1 empty"
    assert l2, "l2 empty"
    min_dist = 1000000
    for e1 in l1:
        for e2 in l2:
            dist_e1e2 = abs(e1-e2)
            if dist_e1e2 < min_dist:
                min_dist = dist_e1e2
    assert min_dist != 1000000, "no min_dist extracted"
    return min_dist


################################################################################

def get_kmer_hit_ends(seq, kmer):
    """
    Given a sequence and a k-mer, return match end positions (1-based) of k-mer
    in sequence. If not hits, return empty list.

    >>> seq = "AACGAAACG"
    >>> kmer = "ACG"
    >>> get_kmer_hit_ends(seq, kmer)
    [4, 9]
    >>> kmer = "ACC"
    >>> get_kmer_hit_ends(seq, kmer)
    []

    """
    assert seq, "seq empty"
    assert kmer, "kmer empty"
    k = len(kmer)
    hit_list = []
    for i in range(len(seq)-k+1):
        end = i+k
        check_kmer = seq[i:end]
        if check_kmer == kmer:
            hit_list.append(end)
    return hit_list


################################################################################

def seqs_dic_count_kmer_freqs(seqs_dic, k,
                              rna=False,
                              perc=False,
                              return_ratios=False,
                              report_key_error=True,
                              convert_to_uc=False):
    """
    Given a dictionary with sequences seqs_dic, count how many times each
    k-mer is found over all sequences (== get k-mer frequencies).
    Return k-mer frequencies count dictionary.
    By default, a DNA dictionary is used, and key errors will be reported.

    rna:
    Instead of DNA dictionary, use RNA dictionary (ACGU) for counting
    k-mers.
    perc:
    If True, make percentages out of ratios (*100).
    return_ratios:
    Return di-nucleotide ratios instead of frequencies (== counts).
    report_key_error:
    If True, report key error (di-nucleotide not in count_dic).
    convert_to_uc:
    Convert sequences to uppercase before counting.

    >>> seqs_dic = {'seq1': 'AACGTC', 'seq2': 'GGACT'}
    >>> seqs_dic_count_kmer_freqs(seqs_dic, 2)
    {'AA': 1, 'AC': 2, 'AG': 0, 'AT': 0, 'CA': 0, 'CC': 0, 'CG': 1, 'CT': 1, 'GA': 1, 'GC': 0, 'GG': 1, 'GT': 1, 'TA': 0, 'TC': 1, 'TG': 0, 'TT': 0}
    >>> seqs_dic = {'seq1': 'AAACGT'}
    >>> seqs_dic_count_kmer_freqs(seqs_dic, 2, return_ratios=True, perc=True)
    {'AA': 40.0, 'AC': 20.0, 'AG': 0.0, 'AT': 0.0, 'CA': 0.0, 'CC': 0.0, 'CG': 20.0, 'CT': 0.0, 'GA': 0.0, 'GC': 0.0, 'GG': 0.0, 'GT': 20.0, 'TA': 0.0, 'TC': 0.0, 'TG': 0.0, 'TT': 0.0}

    """
    # Checks.
    assert seqs_dic, "given dictinary seqs_dic empty"
    assert k, "invalid k given"
    assert k > 0, "invalid k given"
    # Get k-mer dictionary.
    count_dic = get_kmer_dic(k, rna=rna)
    # Count k-mers for all sequences in seqs_dic.
    total_c = 0
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        if convert_to_uc:
            seq = seq.upper()
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            if report_key_error:
                assert kmer in count_dic, "k-mer \"%s\" not in count_dic" %(kmer)
            if kmer in count_dic:
                count_dic[kmer] += 1
                total_c += 1
    assert total_c, "no k-mers counted for given seqs_dic (sequence lengths < set k ?)"

    # Calculate ratios.
    if return_ratios:
        for kmer in count_dic:
            ratio = count_dic[kmer] / total_c
            if perc:
                count_dic[kmer] = ratio*100
            else:
                count_dic[kmer] = ratio
    # Return k-mer counts or ratios.
    return count_dic


################################################################################

def phylop_norm_test_scores(test_pp_con_out,
                             dec_round=4,
                             int_whole_nr=True):
    """
    Read in phyloP .pp.con file scores for test set,
    normalize values to -1 ... 1 and overwrite (!) existing phyloP .pp.con file.
    Normalization is min-max for negative and positive phyloP scores
    separately.

    int_whole_nr:
        If True, output whole numbers without decimal places.

    """
    # Mean normalization for phyloP scores.
    test_con_dic = {}
    seq_id = ""
    pp_max = -1000
    pp_min = 1000

    # Read in positive set phyloP scores.
    with open(test_pp_con_out) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                test_con_dic[seq_id] = []
            else:
                pp_sc = float(line.strip())
                test_con_dic[seq_id].append(pp_sc)
                if pp_sc > pp_max:
                    pp_max = pp_sc
                if pp_sc < pp_min:
                    pp_min = pp_sc
    f.closed
    assert test_con_dic, "no entries read into test_con_dic dictionary"

    # Individual max values for min-max of positive+negative values.
    pp_neg_max = abs(pp_min)
    pp_neg_min = 0
    pp_pos_max = pp_max
    pp_pos_min = 0

    # Mean normalize test phylop scores and overwrite original file.
    OUTP = open(test_pp_con_out,"w")

    for seq_id in test_con_dic:
        OUTP.write(">%s\n" %(seq_id))
        for pp_sc in test_con_dic[seq_id]:
            if pp_sc == 0:
                OUTP.write("0\n")
            else:
                if pp_sc < 0:
                    pp_sc_abs = abs(pp_sc)
                    pp_sc_norm = min_max_normalize(pp_sc_abs, pp_neg_max, pp_neg_min)
                    pp_sc_norm = round(pp_sc_norm, dec_round)
                    pp_sc_norm = -1*pp_sc_norm
                else:
                    pp_sc_norm = min_max_normalize(pp_sc, pp_pos_max, pp_pos_min)
                    pp_sc_norm = round(pp_sc_norm, dec_round)
                if pp_sc_norm == 0:
                    pp_sc_norm = 0
                if int_whole_nr and not pp_sc_norm % 1:
                    OUTP.write("%i\n" %(int(pp_sc_norm)))
                else:
                    OUTP.write("%s\n" %(str(pp_sc_norm)))
    OUTP.close()


################################################################################

def phylop_norm_train_scores(pos_pp_con_out, neg_pp_con_out,
                             dec_round=4,
                             int_whole_nr=True):
    """
    Read in phyloP .pp.con file scores for positive and negative set,
    normalize values to -1 ... 1 and overwrite (!) existing phyloP .pp.con files.
    Normalization is min-max for negative and positive phyloP scores
    separately. Min + max values are extracted from the union of positive
    and negative set.

    int_whole_nr:
        If True, output whole numbers without decimal places.

    """
    # Mean normalization for phyloP scores.
    pos_con_dic = {}
    neg_con_dic = {}
    seq_id = ""
    pp_max = -1000
    pp_min = 1000

    # Read in positive set phyloP scores.
    with open(pos_pp_con_out) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                pos_con_dic[seq_id] = []
            else:
                pp_sc = float(line.strip())
                pos_con_dic[seq_id].append(pp_sc)
                if pp_sc > pp_max:
                    pp_max = pp_sc
                if pp_sc < pp_min:
                    pp_min = pp_sc
    f.closed
    assert pos_con_dic, "no entries read into pos_con_dic dictionary"

    # Read in negative set phyloP scores.
    with open(neg_pp_con_out) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                neg_con_dic[seq_id] = []
            else:
                pp_sc = float(line.strip())
                neg_con_dic[seq_id].append(pp_sc)
                if pp_sc > pp_max:
                    pp_max = pp_sc
                if pp_sc < pp_min:
                    pp_min = pp_sc
    f.closed
    assert neg_con_dic, "no entries read into neg_con_dic dictionary"

    # Individual max values for min-max of positive+negative values.
    pp_neg_max = abs(pp_min)
    pp_neg_min = 0
    pp_pos_max = pp_max
    pp_pos_min = 0

    # Mean normalize positive phylop scores and overwrite original file.
    OUTP = open(pos_pp_con_out,"w")

    for seq_id in pos_con_dic:
        OUTP.write(">%s\n" %(seq_id))
        for pp_sc in pos_con_dic[seq_id]:
            if pp_sc == 0:
                OUTP.write("0\n")
            else:
                if pp_sc < 0:
                    pp_sc_abs = abs(pp_sc)
                    pp_sc_norm = min_max_normalize(pp_sc_abs, pp_neg_max, pp_neg_min)
                    pp_sc_norm = round(pp_sc_norm, dec_round)
                    pp_sc_norm = -1*pp_sc_norm
                else:
                    pp_sc_norm = min_max_normalize(pp_sc, pp_pos_max, pp_pos_min)
                    pp_sc_norm = round(pp_sc_norm, dec_round)
                if pp_sc_norm == 0:
                    pp_sc_norm = 0
                if int_whole_nr and not pp_sc_norm % 1:
                    OUTP.write("%i\n" %(int(pp_sc_norm)))
                else:
                    OUTP.write("%s\n" %(str(pp_sc_norm)))
    OUTP.close()

    # Mean normalize negative phylop scores and overwrite original file.
    OUTN = open(neg_pp_con_out,"w")

    for seq_id in neg_con_dic:
        OUTN.write(">%s\n" %(seq_id))
        for pp_sc in neg_con_dic[seq_id]:
            if pp_sc == 0:
                OUTN.write("0\n")
            else:
                if pp_sc < 0:
                    pp_sc_abs = abs(pp_sc)
                    pp_sc_norm = min_max_normalize(pp_sc_abs, pp_neg_max, pp_neg_min)
                    pp_sc_norm = round(pp_sc_norm, dec_round)
                    pp_sc_norm = -1*pp_sc_norm
                else:
                    pp_sc_norm = min_max_normalize(pp_sc, pp_pos_max, pp_pos_min)
                    pp_sc_norm = round(pp_sc_norm, dec_round)
                if pp_sc_norm == 0:
                    pp_sc_norm = 0
                if int_whole_nr and not pp_sc_norm % 1:
                    OUTN.write("%i\n" %(int(pp_sc_norm)))
                else:
                    OUTN.write("%s\n" %(str(pp_sc_norm)))
    OUTN.close()


################################################################################

def feat_min_max_norm_train_scores(pos_feat_out, neg_feat_out,
                                   p_values=False,
                                   dec_round=4,
                                   int_whole_nr=True):
    """
    Read in feature files for positive and negative set, min max normalize
    values, and overwrite (!) existing feature files.
    Min max normalization resulting in new scores from 0 to 1.

    p_values:
        If True, treat scores as p-values, i.e., normalized score
        == 1 - score
    int_whole_nr:
        If True, output whole numbers without decimal places.

    """
    pos_sc_dic = {}
    neg_sc_dic = {}
    site_id = ""
    sc_max = -1000000
    sc_min = 1000000

    # Read in positive scores.
    with open(pos_feat_out) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                site_id = m.group(1)
                pos_sc_dic[site_id] = []
            else:
                sc = float(line.strip())
                pos_sc_dic[site_id].append(sc)
                if sc > sc_max:
                    sc_max = sc
                if sc < sc_min:
                    sc_min = sc
    f.closed
    assert pos_sc_dic, "no entries read into pos_sc_dic dictionary"

    # Read in negative scores.
    with open(neg_feat_out) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                site_id = m.group(1)
                neg_sc_dic[site_id] = []
            else:
                sc = float(line.strip())
                neg_sc_dic[site_id].append(sc)
                if sc > sc_max:
                    sc_max = sc
                if sc < sc_min:
                    sc_min = sc
    f.closed
    assert neg_sc_dic, "no entries read into neg_sc_dic dictionary"

    # Min max normalize positive scores and output to original file.
    OUTP = open(pos_feat_out,"w")
    for site_id in pos_sc_dic:
        OUTP.write(">%s\n" %(site_id))
        for sc in pos_sc_dic[site_id]:
            if sc == 0:
                OUTP.write("0\n")
            else:
                if p_values:
                    sc_norm = 1 - sc
                else:
                    sc_norm = min_max_normalize(sc, sc_max, sc_min)
                sc_norm = round(sc_norm, dec_round)
                if int_whole_nr and not sc_norm % 1:
                    OUTP.write("%i\n" %(int(sc_norm)))
                else:
                    OUTP.write("%s\n" %(str(sc_norm)))
    OUTP.close()

    # Min max normalize negative scores and output to original file.
    OUTN = open(neg_feat_out,"w")
    for site_id in neg_sc_dic:
        OUTN.write(">%s\n" %(site_id))
        for sc in neg_sc_dic[site_id]:
            if sc == 0:
                OUTN.write("0\n")
            else:
                if p_values:
                    sc_norm = 1 - sc
                else:
                    sc_norm = min_max_normalize(sc, sc_max, sc_min)
                sc_norm = round(sc_norm, dec_round)
                if int_whole_nr and not sc_norm % 1:
                    OUTN.write("%i\n" %(int(sc_norm)))
                else:
                    OUTN.write("%s\n" %(str(sc_norm)))
    OUTN.close()


################################################################################

def bed_get_feature_annotations(in_bed, feat_bed, feat_out,
                                feat_type="C",
                                stats_dic=None,
                                disable_pol=False):

    """
    Overlap in_bed with feat_bed, and annotate overlapping regions
    depending on set feat_type (C, N).
    C: categorical, one-hot, store 1 for overlapping position and
    0 for not overlapping position.
    N: numerical, i.e., use column 5 feat_bed score to store as score
    for each overlapping position, and 0 for not overlapping position.
    Store feature file to feat_out.

    Format of feat_out file depends on feat_type:
    if "C":
    id1<tab>00000111111 ...
    if "N":
    >id1
    value1
    value2
    ...

    in_bed:
        Input BED regions, annotate each position.
    feat_bed:
        Feature BED regions, use for annotating in_bed positions.
    feat_out:
        Output feature annotation file. Format depends on feat_type.
    feat_type:
        "C" for categorical, or "N" for numerical output annotations.
    disable_pol:
        If yes, disable strandedness (== do not set -s in intersectBed),
        i.e., do not differentiate between strands when adding
        annotations.

    >>> in_bed = "test_data/feat_in.bed"
    >>> feat_bed_old_nick = "test_data/feat_old_nick.bed"
    >>> feat_bed_feat_666 = "test_data/feat_666.bed"
    >>> old_nick_exp1 = "test_data/feat_old_nick_1.exp.out"
    >>> old_nick_exp2 = "test_data/feat_old_nick_2.exp.out"
    >>> feat_666_exp1 = "test_data/feat_666_1.exp.out"
    >>> feat_666_exp2 = "test_data/feat_666_2.exp.out"
    >>> old_nick_out = "test_data/test.tmp.old_nick"
    >>> feat_666_out = "test_data/test.tmp.feat_666"
    >>> bed_get_feature_annotations(in_bed, feat_bed_old_nick, old_nick_out, feat_type="C", disable_pol=True)
    >>> diff_two_files_identical(old_nick_out, old_nick_exp1)
    True
    >>> bed_get_feature_annotations(in_bed, feat_bed_feat_666, feat_666_out, feat_type="N", disable_pol=True)
    >>> diff_two_files_identical(feat_666_out, feat_666_exp1)
    True
    >>> bed_get_feature_annotations(in_bed, feat_bed_old_nick, old_nick_out, feat_type="C", disable_pol=False)
    >>> diff_two_files_identical(old_nick_out, old_nick_exp2)
    True
    >>> bed_get_feature_annotations(in_bed, feat_bed_feat_666, feat_666_out, feat_type="N", disable_pol=False)
    >>> diff_two_files_identical(feat_666_out, feat_666_exp2)
    True

    """
    # Checks.
    ftl = ["C", "N"]
    assert feat_type in ftl, "invalid feat_type given (allowed: C,N)"

    # Temp overlap results file.
    random_id = uuid.uuid1()
    tmp_out = str(random_id) + ".tmp.out"

    if stats_dic is not None:
        stats_dic["total_pos"] = 0
        stats_dic["feat_type"] = feat_type
        stats_dic["zero_sites"] = 0
        stats_dic["total_sites"] = 0
        stats_dic["mean_l"] = 0
        stats_dic["median_l"] = 0
        stats_dic["min_l"] = 0
        stats_dic["max_l"] = 0
        stats_dic["stdev_l"] = 0
        if feat_type == "C":
            stats_dic["0"] = 0
            stats_dic["1"] = 0
        else:
            stats_dic["mean"] = 0
            stats_dic["stdev"] = 0
            stats_dic["zero_pos"] = 0
            # Value list.
            v_list = []
        # BED region lengths list.
        len_list = []

    # Read in in_bed, store start + end coordinates.
    id2s_dic = {}
    id2e_dic = {}
    # Store positional values list for each site in dic.
    id2vl_dic = {}
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            assert site_id not in id2s_dic, "non-unique site ID \"%s\" in in_bed" %(site_id)
            id2s_dic[site_id] = site_s
            id2e_dic[site_id] = site_e
            site_l = site_e - site_s
            id2vl_dic[site_id] = ["0"]*site_l
    f.closed
    assert id2s_dic, "given in_bed \"%s\" empty?" %(in_bed)

    # Store feature region lengths.
    if stats_dic:
        with open(feat_bed) as f:
            for line in f:
                row = line.strip()
                cols = line.strip().split("\t")
                site_s = int(cols[1])
                site_e = int(cols[2])
                site_l = site_e - site_s
                len_list.append(site_l)
        f.closed

    # Run overlap calculation to get overlapping regions.
    intersect_params = "-s -wb"
    if disable_pol:
        intersect_params = "-wb"
    intersect_bed_files(in_bed, feat_bed, intersect_params, tmp_out)

    # Get annotations.
    with open(tmp_out) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            s = int(cols[1]) + 1 # Make one-based.
            e = int(cols[2])
            site_id = cols[3]
            site_s = id2s_dic[site_id] + 1 # Make one-based.
            site_e = id2e_dic[site_id]
            site_pol = cols[5]
            score = cols[10]
            # + case.
            if site_pol == "+" or disable_pol:
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = i - site_s
                        if feat_type == "C":
                            id2vl_dic[site_id][li] = "1"
                        else:
                            id2vl_dic[site_id][li] = score
            else:
                for i in range(site_s, site_e+1):
                    if i >= s and i <= e:
                        # Get list index.
                        li = site_e - i
                        if feat_type == "C":
                            id2vl_dic[site_id][li] = "1"
                        else:
                            id2vl_dic[site_id][li] = score
    f.closed

    # Output annotations to file.
    OUTLAB = open(feat_out,"w")

    # Output labels for each site.
    for site_id in id2vl_dic:
        if feat_type == "C":
            # List to string.
            label_str = "".join(id2vl_dic[site_id])
            OUTLAB.write("%s\t%s\n" %(site_id, label_str))
            if stats_dic:
                stats_dic["total_sites"] += 1
                site_0 = True
                for v in id2vl_dic[site_id]:
                    stats_dic[v] += 1
                    if v == "1":
                        site_0 = False
                    stats_dic["total_pos"] += 1
                if site_0:
                    stats_dic["zero_sites"] += 1
        else:
            OUTLAB.write(">%s\n" %(site_id))
            site_0 = True
            for v in id2vl_dic[site_id]:
                OUTLAB.write("%s\n" %(v))
                if stats_dic:
                    v_list.append(float(v))
                    if v == "0":
                        stats_dic["zero_pos"] += 1
                    else:
                        site_0 = False
            if stats_dic:
                stats_dic["total_sites"] += 1
                if site_0:
                    stats_dic["zero_sites"] += 1
    OUTLAB.close()

    # Additional stats if feat_type numerical.
    if stats_dic:
        if feat_type == "N":
            assert v_list, "no values stored in v_list"
            stats_dic["mean"] = statistics.mean(v_list)
            stats_dic["stdev"] = statistics.stdev(v_list)
            stats_dic["total_pos"] = len(v_list)

        assert len_list, "no lengths stored in length list"
        stats_dic["mean_l"] = statistics.mean(len_list)
        stats_dic["median_l"] = statistics.median(len_list)
        stats_dic["stdev_l"] = statistics.stdev(len_list)
        stats_dic["max_l"] = max(len_list)
        stats_dic["min_l"] = min(len_list)

    # Take out the trash.
    litter_street = True
    if litter_street:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


################################################################################

def get_valid_file_ending(s):
    """
    Modified after:
    https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename

    def get_valid_filename(s):
        s = str(s).strip().replace(' ', '_')
        return re.sub(r'(?u)[^-\w.]', '', s)

    In addition, start and end of file ending should start with word or
    number.

    >>> s = "___.hallole123.so_hallole123.___"
    >>> get_valid_file_ending(s)
    'hallole123.so_hallole123'
    >>> get_valid_file_ending("john's new arctic warfare")
    'johns_new_arctic_warfare'

    """
    assert s, "given s empty"
    # Strip and replace spaces with _.
    s = str(s).strip().replace(' ', '_')
    # Remove non-word characters from start and end.
    m = re.search('\W*([a-zA-Z0-9].+[a-zA-Z0-9])\W*', s)
    if m:
        return re.sub(r'(?u)[^-\w.]', '', m.group(1))
    else:
        return re.sub(r'(?u)[^-\w.]', '', s)


################################################################################

def load_eval_data(args,
                   load_negatives=False,
                   return_graphs=False,
                   train_folder=False,
                   num_features=4,
                   undirected=True,
                   str_elem_1h=False):

    """
    Load training data for graphprot2 eval, to generate motifs and profiles.

    """

    # Checks.
    assert os.path.isdir(args.in_gt_folder), "--gt-in folder does not exist"
    # Feature file containing info for features used for model training.
    if train_folder:
        assert os.path.isdir(train_folder), "model training folder %s does not exist" %(train_folder)
        feat_file = train_folder + "/" + "features.out"
    else:
        assert os.path.isdir(args.in_train_folder), "--train-in folder does not exist"
        feat_file = args.in_train_folder + "/" + "features.out"
    assert os.path.exists(feat_file), "%s features file expected but not does not exist" %(feat_file)

    # graphprot2 predict output folder.
    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Channel info output file.
    channel_infos_out = out_folder + "/" + "channel_infos.out"
    channel_info_list = []
    channel_nr = 0

    # Read in feature info.
    fid2type_dic = {}
    fid2cat_dic = {} # Store category labels or numerical score IDs in list.
    fid2norm_dic = {}
    fid2row_dic = {}
    print("Read in feature infos from %s ... " %(feat_file))
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            feat_type = cols[1]
            feat_cat_list = cols[2].split(",")
            feat_cat_list.sort()
            feat_norm = cols[3]
            fid2row_dic[feat_id] = row
            assert feat_id not in fid2type_dic, "feature ID \"%s\" found twice in feature file" %(feat_id)
            fid2type_dic[feat_id] = feat_type
            fid2cat_dic[feat_id] = feat_cat_list
            fid2norm_dic[feat_id] = feat_norm
    f.closed
    assert fid2type_dic, "no feature infos read in from graphprot2 train feature file %s" %(feat_file)
    # Check for base pair model.
    assert "bpp.str" not in fid2type_dic, "--train-in model was trained with base pair information, but graphprot2 eval so far does not support the visualization of base pair information"

    # Read in features.out from graphprot2 gt and check.
    gt_feat_file = args.in_gt_folder + "/" + "features.out"
    assert os.path.exists(gt_feat_file), "%s features file expected but not does not exist" %(gt_feat_file)
    gt_fid2row_dic = {}
    print("Read in feature infos from %s ... " %(gt_feat_file))
    with open(gt_feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            gt_fid2row_dic[feat_id] = row
    f.closed
    assert gt_fid2row_dic, "no feature infos found in --gt-in feature file %s" %(gt_feat_file)
    # Compare gt+train features.
    for fid in fid2row_dic:
        assert fid in gt_fid2row_dic, "graphprot2 train feature ID \"%s\" not found in %s" %(fid, gt_feat_file)

    # Context + bp settings.
    seqs_all_uc = True
    bps_mode = 1

    # Check sequence feature.
    assert "fa" in fid2type_dic, "feature ID \"fa\" not in feature file"
    assert fid2cat_dic["fa"] == ["A", "C", "G", "U"], "sequence feature alphabet != A,C,G,U"

    # Read in FASTA sequences.
    pos_fa_in = args.in_gt_folder + "/" + "positives.fa"
    if load_negatives:
        pos_fa_in = args.in_gt_folder + "/" + "negatives.fa"
    assert os.path.exists(pos_fa_in), "--gt-in folder does not contain %s"  %(pos_fa_in)
    seqs_dic = read_fasta_into_dic(pos_fa_in, all_uc=seqs_all_uc)
    assert seqs_dic, "no sequences read in from FASTA file \"%s\"" %(pos_fa_in)

    # Get uppercase (viewpoint) region start and ends for each sequence.
    vp_dic = extract_uc_region_coords_from_fasta(seqs_dic)

    # Data dictionaries.
    bpp_dic = {}
    feat_dic = {}

    # Init feat_dic (storing node feature vector data) with sequence one-hot encodings.
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        feat_dic[seq_id] = string_vectorizer(seq, custom_alphabet=fid2cat_dic["fa"])

    # Channel info dictionary.
    ch_info_dic = {}

    # Add sequence one-hot channels.
    ch_info_dic["fa"] = ["C", [], [], "-"]
    for c in fid2cat_dic["fa"]:
        channel_nr += 1
        channel_id = c
        channel_info = "%i\t%s\tfa\tC\tone_hot" %(channel_nr, channel_id)
        channel_info_list.append(channel_info)
        ch_info_dic["fa"][1].append(channel_nr-1)
        ch_info_dic["fa"][2].append(channel_id)

    # Check and read in more data.
    for fid, ftype in sorted(fid2type_dic.items()): # fid e.g. fa, ftype: C,N.
        if fid == "fa": # already added to feat_dic (first item).
            continue
        # base pairs not supported by eval so far, so do not load.
        #if fid == "bpp.str":
        #    test_bpp_in = args.in_folder + "/" + "test.bpp.str"
        #    assert os.path.exists(test_bpp_in), "--in folder does not contain %s"  %(test_bpp_in)
        #    print("Read in base pair data ... ")
        #    bpp_dic = read_bpp_into_dic(test_bpp_in, vp_dic,
        #                                bps_mode=args.bps_mode)
        #    assert bpp_dic, "no base pair information read in (bpp_dic empty)"

        # All features (additional to .fa) like .elem_p.str, .con, .eia, .tra, .rra, or user defined.
        feat_alphabet = fid2cat_dic[fid]
        pos_feat_in = args.in_gt_folder + "/positives." + fid
        if load_negatives:
            pos_feat_in = args.in_gt_folder + "/negatives." + fid
        assert os.path.exists(pos_feat_in), "--in folder does not contain %s"  %(pos_feat_in)
        print("Read in .%s annotations ... " %(fid))
        n_to_1h = False
        # Special case: convert elem_p.str probabilities to 1-hot encoding.
        if fid == "elem_p.str" and str_elem_1h:
            n_to_1h = True
        feat_dic = read_feat_into_dic(pos_feat_in, ftype,
                                      feat_dic=feat_dic,
                                      n_to_1h=n_to_1h,
                                      label_list=feat_alphabet)
        assert feat_dic, "no .%s information read in (feat_dic empty)" %(fid)
        if fid == "elem_p.str" and str_elem_1h:
            ftype = "C"
        ch_info_dic[fid] = [ftype, [], [], "-"]
        if ftype == "N":
            for c in feat_alphabet:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[fid]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, fid, encoding)
                channel_info_list.append(channel_info)
                ch_info_dic[fid][1].append(channel_nr-1)
                ch_info_dic[fid][2].append(channel_id)
                ch_info_dic[fid][3] = encoding
        elif ftype == "C":
            for c in feat_alphabet:
                channel_nr += 1
                #channel_id = fid + "_" + c
                channel_id = c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, fid)
                channel_info_list.append(channel_info)
                ch_info_dic[fid][1].append(channel_nr-1)
                ch_info_dic[fid][2].append(channel_id)
                ch_info_dic[fid][3] = "-"
        else:
            assert False, "invalid feature type given (%s) for feature %s" %(ftype,fid)

    # Output channel infos.
    CIOUT = open(channel_infos_out, "w")
    CIOUT.write("ch\tch_id\tfeat_id\tfeat_type\tencoding\n")
    for ch_info in channel_info_list:
        CIOUT.write("%s\n" %(ch_info))
    CIOUT.close()

    """
    Generate feature+edges lists or graph list (if return_graphs=True)
    to return.

    """
    # Sequence ID list + label list.
    seq_ids_list = []
    label_list = []
    idx2id_dic = {}
    id2idx_dic = {}
    i = 0
    for seq_id,seq in sorted(seqs_dic.items()):
        seq_ids_list.append(seq_id)
        label_list.append(i)
        id2idx_dic[seq_id] = i
        idx2id_dic[i] = seq_id
        i += 1

    # Store node data in list of 2d lists.
    all_features = []
    # if return_graphs=True.
    all_graphs = []

    for idx, label in enumerate(label_list):
        seq_id = seq_ids_list[idx]
        seq = seqs_dic[seq_id]
        l_seq = len(seq)

        # Checks.
        check_num_feat = len(feat_dic[seq_id][0])
        assert num_features == check_num_feat, "# features (num_features) from model parameter file != loaded number of node features (%i != %i)" %(model_num_feat, check_num_feat)

        if return_graphs:
            edge_index_1 = []
            edge_index_2 = []
            for ni in range(l_seq - 1):
                edge_index_1.append(ni)
                edge_index_2.append(ni+1)
                if undirected:
                    edge_index_1.append(ni+1)
                    edge_index_2.append(ni)
            edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
            x = torch.tensor(feat_dic[seq_id], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=label)
            all_graphs.append(data)
        else:
            # Appendo.
            all_features.append(feat_dic[seq_id])

    # Return some double talking jive data.
    if return_graphs:
        assert all_graphs, "all_graphs empty"
        return seqs_dic, idx2id_dic, all_graphs, ch_info_dic
    else:
        assert all_features, "all_features empty"
        return seqs_dic, idx2id_dic, all_features, ch_info_dic


################################################################################

def load_predict_data(args,
                      return_graphs=False):

    """
    Load prediction data from GraphProt2 predict output folder
    and return either as list of graphs or list of feature lists.

    """

    # Checks.
    assert os.path.isdir(args.in_folder), "--in folder does not exist"
    assert os.path.isdir(args.train_in_folder), "--train-in model folder does not exist"

    # Feature file containing info for features used for model training.
    feat_file = args.train_in_folder + "/" + "features.out"
    assert os.path.exists(feat_file), "%s features file expected but not does not exist" %(feat_file)

    # Read in model parameters.
    params_file = args.train_in_folder + "/final.params"
    assert os.path.isfile(params_file), "missing model training parameter file %s" %(params_file)
    params_dic = read_settings_into_dic(params_file)
    assert "num_features" in params_dic, "num_features info missing in model parameter file %s" %(params_file)
    model_num_feat = int(params_dic["num_features"])

    # graphprot2 predict output folder.
    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Channel info output file.
    channel_infos_out = out_folder + "/" + "channel_infos.out"
    channel_info_list = []
    channel_nr = 0

    # Read in feature info.
    fid2type_dic = {}
    fid2cat_dic = {} # Store category labels or numerical score IDs in list.
    fid2norm_dic = {}
    fid2row_dic = {}
    print("Read in feature infos from %s ... " %(feat_file))
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            feat_type = cols[1]
            feat_cat_list = cols[2].split(",")
            feat_cat_list.sort()
            feat_norm = cols[3]
            fid2row_dic[feat_id] = row
            assert feat_id not in fid2type_dic, "feature ID \"%s\" found twice in feature file" %(feat_id)
            fid2type_dic[feat_id] = feat_type
            fid2cat_dic[feat_id] = feat_cat_list
            fid2norm_dic[feat_id] = feat_norm
    f.closed
    assert fid2type_dic, "no feature infos read in from graphprot2 train feature file %s" %(feat_file)
    # Check sequence feature.
    assert "fa" in fid2type_dic, "feature ID \"fa\" not in feature file"
    assert fid2cat_dic["fa"] == ["A", "C", "G", "U"], "sequence feature alphabet != A,C,G,U"

    # args.mode == 2 (predict position-wise scoring profiles) check.
    if args.mode == 2:
        assert "bpp.str" not in fid2type_dic, "--model-in model was trained with base pair information, but profile prediction mode (--mode 2) so far does not support base pair information. Use --mode 1 or train models without base pair information"

    # Read in features.out from graphprot2 gp and check.
    gp_feat_file = args.in_folder + "/" + "features.out"
    assert os.path.exists(gp_feat_file), "%s features file expected but not does not exist" %(gp_feat_file)
    gp_fid2row_dic = {}
    print("Read in feature infos from %s ... " %(gp_feat_file))
    with open(gp_feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            gp_fid2row_dic[feat_id] = row
    f.closed
    assert gp_fid2row_dic, "no feature infos found in graphprot2 gp feature file %s" %(gp_feat_file)
    #assert len(fid2row_dic) == len(gp_fid2row_dic), "# features in gp + train feature files differ"
    for fid in fid2row_dic:
        assert fid in gp_fid2row_dic, "graphprot2 train feature ID \"%s\" not found in %s" %(fid, gp_feat_file)
        # If predict_profiles, "fa" feature between gp + train can differ.
        if not (args.mode == 2 and fid == "fa"):
            assert gp_fid2row_dic[fid] == fid2row_dic[fid], "feature infos for feature ID \"%s\" differ between gp + train feature files (train: \"%s\", gp: \"%s\")" %(fid, gp_fid2row_dic[fid], fid2row_dic[fid])

    # Get base pair cutoff from train.
    bps_cutoff = False
    gp2_train_param_file = args.train_in_folder + "/settings.graphprot2_train.out"
    assert os.path.isfile(gp2_train_param_file), "missing graphprot2 train parameter file %s" %(gp2_train_param_file)
    gp2_train_param_dic = read_settings_into_dic(gp2_train_param_file)
    if "bpp.str" in fid2type_dic:
        assert "bps_cutoff" in gp2_train_param_dic, "bps_cutoff info missing in graphprot2 train parameter file %s" %(gp2_train_param_file)
        bps_cutoff = float(cols[1])
    # Get str_elem_1h info.
    str_elem_1h = False
    assert "str_elem_1h" in gp2_train_param_dic, "str_elem_1h info missing in graphprot2 train parameter file %s" %(gp2_train_param_file)
    if gp2_train_param_dic["str_elem_1h"] == "True":
        str_elem_1h = True

    # Read in FASTA sequences.
    test_fa_in = args.in_folder + "/" + "test.fa"
    assert os.path.exists(test_fa_in), "--in folder does not contain %s"  %(test_fa_in)
    test_seqs_dic = read_fasta_into_dic(test_fa_in, all_uc=True)
    assert test_seqs_dic, "no sequences read in from FASTA file \"%s\"" %(test_fa_in)

    # Get uppercase (viewpoint) region start and ends for each sequence.
    vp_dic = extract_uc_region_coords_from_fasta(test_seqs_dic)

    # Data dictionaries.
    bpp_dic = {}
    feat_dic = {}

    # Init feat_dic (storing node feature vector data) with sequence one-hot encodings.
    for seq_id in test_seqs_dic:
        seq = test_seqs_dic[seq_id]
        feat_dic[seq_id] = string_vectorizer(seq, custom_alphabet=fid2cat_dic["fa"])

    # Check and read in more data.
    for fid, ftype in sorted(fid2type_dic.items()): # fid e.g. fa, ftype: C,N.
        if fid == "fa":
            # Add sequence one-hot channels.
            for c in fid2cat_dic["fa"]:
                channel_nr += 1
                channel_id = c
                channel_info = "%i\t%s\tfa\tC\tone_hot" %(channel_nr, channel_id)
                channel_info_list.append(channel_info)
            continue
        if fid == "bpp.str":
            test_bpp_in = args.in_folder + "/" + "test.bpp.str"
            assert os.path.exists(test_bpp_in), "--in folder does not contain %s"  %(test_bpp_in)
            print("Read in base pair data ... ")
            bpp_dic = read_bpp_into_dic(test_bpp_in, vp_dic,
                                        bps_mode=args.bps_mode)
            assert bpp_dic, "no base pair information read in (bpp_dic empty)"
        else:
            # All features (additional to .fa) like .elem_p.str, .con, .eia, .tra, .rra, or user defined.
            feat_alphabet = fid2cat_dic[fid]
            test_feat_in = args.in_folder + "/test." + fid

            assert os.path.exists(test_feat_in), "--in folder does not contain %s"  %(test_feat_in)
            print("Read in .%s annotations ... " %(fid))
            n_to_1h = False
            # Special case: convert elem_p.str probabilities to 1-hot encoding.
            if fid == "elem_p.str" and str_elem_1h:
                n_to_1h = True
            feat_dic = read_feat_into_dic(test_feat_in, ftype,
                                          feat_dic=feat_dic,
                                          n_to_1h=n_to_1h,
                                          label_list=feat_alphabet)
            assert feat_dic, "no .%s information read in (feat_dic empty)" %(fid)
            if fid == "elem_p.str" and str_elem_1h:
                ftype = "C"
            if ftype == "N":
                for c in feat_alphabet:
                    channel_nr += 1
                    channel_id = c
                    encoding = fid2norm_dic[fid]
                    channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, fid, encoding)
                    channel_info_list.append(channel_info)
            elif ftype == "C":
                for c in feat_alphabet:
                    channel_nr += 1
                    #channel_id = fid + "_" + c
                    channel_id = c
                    channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, fid)
                    channel_info_list.append(channel_info)
            else:
                assert False, "invalid feature type given (%s) for feature %s" %(ftype,fid)

    # Output channel infos.
    CIOUT = open(channel_infos_out, "w")
    CIOUT.write("ch\tch_id\tfeat_id\tfeat_type\tencoding\n")
    for ch_info in channel_info_list:
        CIOUT.write("%s\n" %(ch_info))
    CIOUT.close()

    """
    Generate feature+edges lists or graph list (if return_graphs=True)
    to return.

    """
    # Sequence ID list + label list.
    seq_ids_list = []
    label_list = []
    idx2id_dic = {}
    id2idx_dic = {}
    i = 0
    for seq_id,seq in sorted(test_seqs_dic.items()):
        seq_ids_list.append(seq_id)
        label_list.append(i)
        id2idx_dic[seq_id] = i
        idx2id_dic[i] = seq_id
        i += 1

    # If return_graphs=False.
    all_features = []
    # if return_graphs=True.
    all_graphs = []

    for idx, label in enumerate(label_list):
        seq_id = seq_ids_list[idx]
        seq = test_seqs_dic[seq_id]
        l_seq = len(seq)

        # Make edge indices for backbone.
        edge_index_1 = []
        edge_index_2 = []
        # Edge indices 0-based!
        for n_idx in range(l_seq - 1):
            edge_index_1.append(n_idx)
            edge_index_2.append(n_idx+1)
            # In case of undirected graphs, add backward edges too.
            if args.undirected:
                edge_index_1.append(n_idx+1)
                edge_index_2.append(n_idx)

        # Add base pair edges.
        if bpp_dic:
            vp_s = vp_dic[seq_id][0] # 1-based.
            vp_e = vp_dic[seq_id][1] # 1-based.
            # Entry e.g. 'CLIP_01': ['130-150,0.33', '160-200,0.44', '240-260,0.55']
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1)) # 1-based.
                p2 = int(m.group(2)) # 1-based.
                bpp_value = float(m.group(3))
                g_p1 = p1 - 1 # 0-based base pair index.
                g_p2 = p2 - 1 # 0-based base pair index.
                # Filter.
                if bpp_value < args.bps_cutoff: continue
                # Add base pair depending on set mode.
                add_edge = False
                if args.bps_mode == 1:
                    if (p1 >= vp_s and p1 <= vp_e) or (p2 >= vp_s and p2 <= vp_e):
                        add_edge = True
                elif args.bps_mode == 2:
                    if p1 >= vp_s and p2 <= vp_e:
                        add_edge = True
                if add_edge:
                    edge_index_1.append(g_p1)
                    edge_index_1.append(g_p2)
                    if args.undirected:
                        edge_index_1.append(g_p2)
                        edge_index_2.append(g_p1)

        # Checks.
        check_num_feat = len(feat_dic[seq_id][0])
        assert model_num_feat == check_num_feat, "# features (num_features) from model parameter file != loaded number of node features (%i != %i)" %(model_num_feat, check_num_feat)

        if return_graphs:
            edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
            x = torch.tensor(feat_dic[seq_id], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=label)
            all_graphs.append(data)
        else:
            #all_edges.append([edge_index_1, edge_index_2])
            all_features.append(feat_dic[seq_id])

    # Return some double talking jive data.
    if return_graphs:
        assert all_graphs, "no graphs constructed (all_graphs empty)"
        return test_seqs_dic, idx2id_dic, all_graphs
    else:
        assert all_features, "no graphs constructed (all_features empty)"
        return test_seqs_dic, bpp_dic, idx2id_dic, all_features


################################################################################

def load_ws_predict_data(args,
                         data_id="data",
                         predict_con_ext=False,
                         add_info_out=True):

    """
    Load whole site prediction data from GraphProt2 predict output folder
    and store in PyG format.

    Needs a data ID in args to construct output folder.

    """

    # Checks.
    assert os.path.isdir(args.in_folder), "--in folder does not exist"
    assert os.path.isdir(args.train_in_folder), "--train-in model folder does not exist"

    # Feature file containing info for features used for model training.
    feat_file = args.train_in_folder + "/" + "features.out"
    assert os.path.exists(feat_file), "%s features file expected but not does not exist" %(feat_file)

    # Data ID.
    data_id = data_id
    # graphprot2 predict output folder.
    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # PyTorch geometric data output folder.
    data_out_folder = args.out_folder + "/" + data_id
    if not os.path.exists(data_out_folder):
        os.makedirs(data_out_folder)

    # Raw and processed data subfolders (structure dictated by PyG).
    raw_out_folder = data_out_folder + "/raw"
    proc_out_folder = data_out_folder + "/processed"
    if not os.path.exists(raw_out_folder):
        os.makedirs(raw_out_folder)
    if not os.path.exists(proc_out_folder):
        os.makedirs(proc_out_folder)

    # Output additional infos (sequences + IDs)?
    add_out_folder = data_out_folder + "/add_info"
    test_add_ids_out_file = False
    test_add_seqs_out_file = False
    if add_info_out:
        if not os.path.exists(add_out_folder):
            os.makedirs(add_out_folder)
        test_add_ids_out_file = add_out_folder + "/test.add_info.ids"
        test_add_seqs_out_file = add_out_folder + "/test.add_info.fa"

    # Delete old .pt files.
    processed_file = proc_out_folder + "/data.pt"
    if os.path.exists(processed_file):
        print("Remove existing data.pt file ... ")
        os.remove(processed_file)

    # Channel info output file.
    channel_infos_out = out_folder + "/" + "channel_infos.out"
    channel_info_list = []
    channel_nr = 0

    # Read in feature info.
    fid2type_dic = {}
    fid2cat_dic = {} # Store category labels or numerical score IDs in list.
    fid2norm_dic = {}
    fid2row_dic = {}
    print("Read in feature infos from %s ... " %(feat_file))
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            feat_type = cols[1]
            feat_cat_list = cols[2].split(",")
            feat_cat_list.sort()
            feat_norm = cols[3]
            fid2row_dic[feat_id] = row
            assert feat_id not in fid2type_dic, "feature ID \"%s\" found twice in feature file" %(feat_id)
            fid2type_dic[feat_id] = feat_type
            fid2cat_dic[feat_id] = feat_cat_list
            fid2norm_dic[feat_id] = feat_norm
    f.closed
    assert fid2type_dic, "no feature infos read in from graphprot2 train feature file %s" %(feat_file)

    if args.mode == 2:
        assert "bpp.str" not in fid2type_dic, "--model-in model was trained with base pair information, but profile prediction mode (--mode 2) so far does not support base pair information. Use --mode 1 or train models without base pair information"

    # Read in features.out from graphprot2 gp and check.
    gp_feat_file = args.in_folder + "/" + "features.out"
    assert os.path.exists(gp_feat_file), "%s features file expected but not does not exist" %(gp_feat_file)
    gp_fid2row_dic = {}
    print("Read in feature infos from %s ... " %(gp_feat_file))
    with open(gp_feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            gp_fid2row_dic[feat_id] = row
    f.closed
    assert gp_fid2row_dic, "no feature infos found in graphprot2 gp feature file %s" %(gp_feat_file)
    #assert len(fid2row_dic) == len(gp_fid2row_dic), "# features in gp + train feature files differ"
    for fid in fid2row_dic:
        assert fid in gp_fid2row_dic, "graphprot2 train feature ID \"%s\" not found in %s" %(fid, gp_feat_file)
        # If predict_profiles, "fa" feature between gp + train can differ.
        if not (args.mode == 2 and fid == "fa"):
            assert gp_fid2row_dic[fid] == fid2row_dic[fid], "feature infos for feature ID \"%s\" differ between gp + train feature files (train: \"%s\", gp: \"%s\")" %(fid, gp_fid2row_dic[fid], fid2row_dic[fid])

    # Get sequence context extension set in gp.
    con_ext = False
    gp_settings_file = args.in_folder + "/settings.graphprot2_gp.out"
    if os.path.exists(gp_settings_file):
        with open(gp_settings_file) as f:
            for line in f:
                cols = line.strip().split("\t")
                if cols[0] == "con_ext":
                    if cols[1] == "False":
                        con_ext = False
                    else:
                        con_ext = int(cols[1])
        f.closed
    # Get base pair + con_ext settings from train.
    bps_mode = 1
    bps_cutoff = 0.5
    con_ext_train = False
    train_settings_file = args.train_in_folder + "/settings.graphprot2_train.out"
    if os.path.exists(train_settings_file):
        with open(train_settings_file) as f:
            for line in f:
                cols = line.strip().split("\t")
                if cols[0] == "bps_mode":
                    bps_mode = int(cols[1])
                if cols[0] == "bps_cutoff":
                    bps_cutoff = float(cols[1])
                if cols[0] == "con_ext":
                    if cols[1] == "False":
                        con_ext_train = False
                    else:
                        con_ext_train = int(cols[1])
        f.closed
    # Compare gp + train context settings.
    seqs_all_uc = False
    if con_ext:
        if not con_ext_train:
            print("WARNING: --con-ext prediction data incompatible with trained model")
            print("           Changing lowercase context regions to uppercase to")
            print("            restore compatibility with trained model. To use")
            print("             lowercase context regions in predictions train")
            print("              a compatible model (graphprot2 gt --con-ext")
            print("              and graphprot2 train --uc-context disabled)")
            seqs_all_uc = True

    # Double talking jive fake graph switches.
    add_fake_g=False
    fake_g_lcuc=False
    # If we have a lowercase uppercase model.
    if con_ext_train:
        if con_ext:
            if predict_con_ext:
                if con_ext_train != predict_con_ext:
                    print("WARNING: set context extension differs between trained model and --con-ext (%i != %i)" %(con_ext_train, predict_con_ext))
            else:
                if con_ext_train != con_ext:
                    print("WARNING: --con-ext differs between trained model and --in dataset (%i != %i)" %(con_ext_train, con_ext))
        else:
            # Only a problem for --mode 1 whole site prediction.
            if args.mode == 1:
                assert False, "Provided model (--model-in) was trained on upper- and lowercase context, but --in dataset does not contain lowercase context. Please provide a compatible model+dataset combination for whole site prediction (--mode 1)."

        if args.mode == 2:
            # Convert profile prediction sequences to uppercase.
            seqs_all_uc = True
            # For profile prediction, we need to add a UC LC fake graph here.
            add_fake_g = True
            fake_g_lcuc = True

    # Check sequence feature.
    assert "fa" in fid2type_dic, "feature ID \"fa\" not in feature file"

    if con_ext_train:
        assert fid2cat_dic["fa"] == ["A", "C", "G", "U", "a", "c", "g", "u"], "sequence feature alphabet != A,C,G,U,a,c,g,u"

    else:
        assert fid2cat_dic["fa"] == ["A", "C", "G", "U"], "sequence feature alphabet != A,C,G,U"

    # Read in FASTA sequences.
    test_fa_in = args.in_folder + "/" + "test.fa"
    assert os.path.exists(test_fa_in), "--in folder does not contain %s"  %(test_fa_in)
    test_seqs_dic = read_fasta_into_dic(test_fa_in, all_uc=seqs_all_uc)
    assert test_seqs_dic, "no sequences read in from FASTA file \"%s\"" %(test_fa_in)

    # Get uppercase (viewpoint) region start and ends for each sequence.
    test_vp_dic = extract_uc_region_coords_from_fasta(test_seqs_dic)

    # Additional features.
    test_pc_con_in = args.in_folder + "/" + "test.pc.con"
    test_pp_con_in = args.in_folder + "/" + "test.pp.con"
    test_tra_in = args.in_folder + "/" + "test.tra"
    test_rra_in = args.in_folder + "/" + "test.rra"
    test_eia_in = args.in_folder + "/" + "test.eia"
    test_str_elem_p_in = args.in_folder + "/" + "test.elem_p.str"
    test_bpp_in = args.in_folder + "/" + "test.bpp.str"

    # Dictionaries.
    test_pc_con_dic = False
    test_pp_con_dic = False
    test_tra_dic = False
    test_rra_dic = False
    test_eia_dic = False
    test_str_elem_p_dic = False
    test_bpp_dic = False

    # Check and read in data.
    for fid, ftype in sorted(fid2type_dic.items()):
        if fid == "fa":
            continue
        if fid == "bpp.str":
            assert os.path.exists(test_bpp_in), "--in folder does not contain %s"  %(test_bpp_in)
            print("Read in base pair data ... ")
            test_bpp_dic = read_bpp_into_dic(test_bpp_in, test_vp_dic,
                                            bps_mode=bps_mode)
        if fid == "eia":
            feat_id = "eia"
            eia_alphabet = fid2cat_dic[feat_id]
            assert os.path.exists(test_eia_in), "--in folder does not contain %s"  %(test_eia_in)
            print("Read in exon-intron annotations ... ")
            test_eia_dic = read_feat_into_dic(test_eia_in, "C",
                                             label_list=eia_alphabet)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = feat_id + "_" + c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, feat_id)
                channel_info_list.append(channel_info)

        if fid == "elem_p.str":
            feat_id = "elem_p.str"
            assert os.path.exists(test_str_elem_p_in), "--in folder does not contain %s"  %(test_str_elem_p_in)
            print("Read structural elements probabilities ... ")
            test_str_elem_p_dic = read_str_elem_p_into_dic(test_str_elem_p_in)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[feat_id]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, feat_id, encoding)
                channel_info_list.append(channel_info)

        if fid == "pc.con":
            feat_id = "pc.con"
            assert os.path.exists(test_pc_con_in), "--in folder does not contain %s"  %(test_pc_con_in)
            print("Read in phastCons scores ... ")
            test_pc_con_dic = read_con_into_dic(test_pc_con_in)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[feat_id]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, feat_id, encoding)
                channel_info_list.append(channel_info)

        if fid == "pp.con":
            feat_id = "pp.con"
            assert os.path.exists(test_pp_con_in), "--in folder does not contain %s"  %(test_pp_con_in)
            print("Read in phyloP scores ... ")
            test_pp_con_dic = read_con_into_dic(test_pp_con_in)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[feat_id]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, feat_id, encoding)
                channel_info_list.append(channel_info)

        if fid == "rra":
            feat_id = "rra"
            rra_alphabet = fid2cat_dic["rra"]
            assert os.path.exists(test_rra_in), "--in folder does not contain %s"  %(test_rra_in)
            print("Read in repeat region annotations ... ")
            test_rra_dic = read_feat_into_dic(test_rra_in, "C",
                                             label_list=rra_alphabet)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = feat_id + "_" + c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, feat_id)
                channel_info_list.append(channel_info)

        if fid == "tra":
            feat_id = "tra"
            tra_alphabet = fid2cat_dic["tra"]
            assert os.path.exists(test_tra_in), "--in folder does not contain %s"  %(test_tra_in)
            print("Read in transcript region annotations ... ")
            test_tra_dic = read_feat_into_dic(test_tra_in, "C",
                                             label_list=tra_alphabet)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = feat_id + "_" + c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, feat_id)
                channel_info_list.append(channel_info)

    # Output channel infos.
    CIOUT = open(channel_infos_out, "w")
    CIOUT.write("ch\tch_id\tfeat_id\tfeat_type\tencoding\n")
    for ch_info in channel_info_list:
        CIOUT.write("%s\n" %(ch_info))
    CIOUT.close()

    print("Convert data to PyG format ... ")

    # Convert data to PyTorch geometric format.
    anl, agi, ae, ana, g_idx, n_idx = generate_geometric_data(test_seqs_dic,
                                                        test_vp_dic,
                                                        pc_con_dic=test_pc_con_dic,
                                                        pp_con_dic=test_pp_con_dic,
                                                        eia_dic=test_eia_dic,
                                                        rra_dic=test_rra_dic,
                                                        tra_dic=test_tra_dic,
                                                        str_elem_p_dic=test_str_elem_p_dic,
                                                        bpp_dic=test_bpp_dic,
                                                        add_ids_out_file=test_add_ids_out_file,
                                                        add_seqs_out_file=test_add_seqs_out_file,
                                                        bps_mode=bps_mode,
                                                        add_fake_g=add_fake_g,
                                                        fake_g_lcuc=fake_g_lcuc,
                                                        plfold_bpp_cutoff=bps_cutoff)

    print("Store PyG data on HD ... ")

    # RAW output files.
    agi_file = raw_out_folder + "/" + data_id + "_graph_indicator.txt"
    anl_file = raw_out_folder + "/" + data_id + "_node_labels.txt"
    ana_file = raw_out_folder + "/" + data_id + "_node_attributes.txt"
    ae_file = raw_out_folder + "/" + data_id + "_A.txt"
    # Write to files.
    f = open(agi_file, 'w')
    f.writelines([str(e) + "\n" for e in agi])
    f.close()
    f = open(anl_file, 'w')
    f.writelines([str(e) + "\n" for e in anl])
    f.close()
    if ana:
        f = open(ana_file, 'w')
        f.writelines([s + "\n" for s in ana])
        f.close()
    else:
        if os.path.exists(ana_file):
            os.remove(ana_file)
    f = open(ae_file, 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in ae])
    f.close()
    # Return test sequences dic.
    return test_seqs_dic


################################################################################

def make_seqs_dic_uc(seqs_dic):
    """
    Make all sequences (dictionary with key==sequence ID and value==sequence)
    inside given seqs_dic uppercase.

    >>> seqs_dic = {'id1':'acguACGUacgu'}
    >>> make_seqs_dic_uc(seqs_dic)
    >>> seqs_dic
    {'id1': 'ACGUACGUACGU'}

    """
    assert seqs_dic, "given seqs_dic empty"
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id].upper()
        seqs_dic[seq_id] = seq






















################################################################################

def load_training_data(args,
                       li2label_dic=None):

    """
    Load training data from data folder generated by GraphProt2 gt and
    return training data as:
    seqs_dic, idx2id_dic, label_list, all_graphs

    li2label_dic:
        Class label to RBP label/name dictionary. For associating the
        positive class label to the RBP name in generic model cross
        validation (if --gm-cv is set).

    """

    # Checks.
    assert os.path.exists(args.in_folder), "--in folder does not exist"

    # Feature file containing info for features inside --in folder.
    feat_file = args.in_folder + "/" + "features.out"
    assert os.path.exists(feat_file), "%s features file expected but not does not exist" %(feat_file)

    # graphprot2 train output folder.
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Channel info output file.
    channel_infos_out = args.out_folder + "/" + "channel_infos.out"
    channel_info_list = []
    channel_nr = 0

    """
    Read in feature info.

    features.out example content:
    fa	C	A,C,G,U	-
    bpp.str	N	bp_prob	prob
    elem_p.str	N	p_u,p_e,p_h,p_i,p_m,p_s	prob
    pc.con	N	phastcons_score	prob
    pp.con	N	phylop_score	mean
    tra	C	A,B,C,E,F,S,T,Z	-
    rra	C	N,R	-

    """
    fid2type_dic = {}
    fid2cat_dic = {} # Store category labels or numerical score IDs in list.
    fid2norm_dic = {}
    print("Read in feature infos from %s ... " %(feat_file))
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            feat_type = cols[1] # C,N
            feat_cat_list = cols[2].split(",")
            feat_cat_list.sort()
            feat_norm = cols[3]
            assert feat_id not in fid2type_dic, "feature ID \"%s\" found twice in feature file" %(feat_id)
            fid2type_dic[feat_id] = feat_type
            fid2cat_dic[feat_id] = feat_cat_list
            fid2norm_dic[feat_id] = feat_norm
    f.closed
    assert fid2type_dic, "no feature IDs read in from feature file %s" %(feat_file)

    # Read in FASTA sequences.
    pos_fa_in = args.in_folder + "/" + "positives.fa"
    neg_fa_in = args.in_folder + "/" + "negatives.fa"
    assert os.path.exists(pos_fa_in), "--in folder does not contain %s"  %(pos_fa_in)
    assert os.path.exists(neg_fa_in), "--in folder does not contain %s"  %(neg_fa_in)

    # Check sequence feature.
    assert "fa" in fid2type_dic, "feature ID \"fa\" not in feature file"
    seqs_all_uc = False
    if fid2cat_dic["fa"] == ["A", "C", "G", "U"]:
        seqs_all_uc = True
    elif fid2cat_dic["fa"] == ["A", "C", "G", "U", "a", "c", "g", "u"]:
        if args.uc_context:
            seqs_all_uc = True
    else:
        assert False, "sequence feature alphabet != A,C,G,U(,a,c,g,u)"

    # Read in sequences.
    seqs_dic = read_fasta_into_dic(pos_fa_in, all_uc=seqs_all_uc)
    pos_ids_dic = {}
    for seq_id in seqs_dic:
        pos_ids_dic[seq_id] = 1
    seqs_dic = read_fasta_into_dic(neg_fa_in, all_uc=seqs_all_uc,
                                   seqs_dic=seqs_dic)
    assert seqs_dic, "no sequences read from FASTA files"
    neg_ids_dic = {}
    for seq_id in seqs_dic:
        if seq_id not in pos_ids_dic:
            neg_ids_dic[seq_id] = 1

    # Check for 4 (8) distinct nucleotides.
    cc_dic = seqs_dic_count_chars(seqs_dic)
    allowed_nt_dic = {'A': 1, 'C': 1, 'G': 1, 'U': 1}
    c_nts = 4
    if not seqs_all_uc:
        allowed_nt_dic = {'A': 1, 'C': 1, 'G': 1, 'U': 1, 'a': 1, 'c': 1, 'g': 1, 'u': 1}
        c_nts = 8
    for nt in cc_dic:
        if nt not in allowed_nt_dic:
            assert False, "sequences with invalid character \"%s\" encountered (allowed characters: ACGU(acgu)" %(nt)
    assert len(cc_dic) == c_nts, "# of distinct nucleotide characters in sequences != expected # (%i != %i)" %(len(cc_dic), c_nts)

    # Get con_ext info used to create dataset.
    con_ext_str = "False"
    gt_settings_file = args.in_folder + "/settings.graphprot2_gt.out"
    assert os.path.exists(gt_settings_file), "graphprot2 gt settings file %s not found" %(gt_settings_file)
    with open(gt_settings_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            if cols[0] == "con_ext":
                if cols[1] != "False":
                    con_ext_str = cols[1]
    f.closed
    if seqs_all_uc:
        con_ext_str = "False"
    # Add info to train settings file.
    train_settings_file = args.out_folder + "/settings.graphprot2_train.out"
    SETOUT = open(train_settings_file, "a")
    SETOUT.write("con_ext\t%s\n" %(con_ext_str))
    SETOUT.close()

    # Get uppercase (viewpoint) region start and ends for each sequence.
    vp_dic = extract_uc_region_coords_from_fasta(seqs_dic)

    # Check for individually selected features.
    indiv_feat_dic = {}
    if args.use_pc_con:
        indiv_feat_dic["pc.con"] = 1
    if args.use_pp_con:
        indiv_feat_dic["pp.con"] = 1
    if args.use_eia:
        indiv_feat_dic["eia"] = 1
    if args.use_tra:
        indiv_feat_dic["tra"] = 1
    if args.use_rra:
        indiv_feat_dic["rra"] = 1
    if args.use_str_elem_p:
        indiv_feat_dic["elem_p.str"] = 1
    if args.use_bps:
        indiv_feat_dic["bpp.str"] = 1

    # Looking for additional features.
    std_fid_dic = {"pc.con" : 1,
                    "pp.con" : 1,
                    "eia" : 1,
                    "tra" : 1,
                    "rra" : 1,
                    "fa" : 1,
                    "elem_p.str" : 1,
                    "bpp.str" : 1}
    add_fid_dic = {}
    for fid in fid2type_dic:
        if fid not in std_fid_dic:
            add_fid_dic[fid] = 1
    if args.use_add_feat:
        for fid in add_fid_dic:
            indiv_feat_dic[fid] = 1

    # Remove features from fid2type_dic.
    if indiv_feat_dic:
        del_feat_list = []
        for fid in fid2type_dic:
            if fid == "fa":
                continue
            if fid not in indiv_feat_dic:
                del_feat_list.append(fid)
        for fid in del_feat_list:
            del fid2type_dic[fid]
    # If only_seq, remove all other found features.
    if args.only_seq:
        fid2type_dic = {}
        fid2type_dic["fa"] = "C"

    # Data dictionaries.
    bpp_dic = {}
    feat_dic = {}

    # Init feat_dic (storing node feature vector data) with sequence one-hot encodings.
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id]
        feat_dic[seq_id] = string_vectorizer(seq, custom_alphabet=fid2cat_dic["fa"])
    # Add sequence one-hot channels.
    for c in fid2cat_dic["fa"]:
        channel_nr += 1
        channel_id = c
        channel_info = "%i\t%s\tfa\tC\tone_hot" %(channel_nr, channel_id)
        channel_info_list.append(channel_info)

    # Check and read in more data.
    for fid, ftype in sorted(fid2type_dic.items()): # fid e.g. fa, ftype: C,N.
        if fid == "fa": # already added to feat_dic (first item).
            continue
        if fid == "bpp.str":
            pos_bpp_in = args.in_folder + "/" + "positives.bpp.str"
            neg_bpp_in = args.in_folder + "/" + "negatives.bpp.str"
            assert os.path.exists(pos_bpp_in), "--in folder does not contain %s"  %(pos_bpp_in)
            assert os.path.exists(neg_bpp_in), "--in folder does not contain %s"  %(neg_bpp_in)
            print("Read in base pair data ... ")
            bpp_dic = read_bpp_into_dic(pos_bpp_in, vp_dic,
                                        bps_mode=args.bps_mode)
            bpp_dic = read_bpp_into_dic(neg_bpp_in, vp_dic,
                                        bpp_dic=bpp_dic,
                                        bps_mode=args.bps_mode)
            assert bpp_dic, "no base pair information read in (bpp_dic empty)"

        else:
            # All features (additional to .fa) like .elem_p.str, .con, .eia, .tra, .rra, or user defined.
            feat_alphabet = fid2cat_dic[fid]
            pos_feat_in = args.in_folder + "/positives." + fid
            neg_feat_in = args.in_folder + "/negatives." + fid
            assert os.path.exists(pos_feat_in), "--in folder does not contain %s"  %(pos_feat_in)
            assert os.path.exists(neg_feat_in), "--in folder does not contain %s"  %(neg_feat_in)
            print("Read in .%s annotations ... " %(fid))
            n_to_1h = False
            # Special case: convert elem_p.str probabilities to 1-hot encoding.
            if fid == "elem_p.str" and args.str_elem_1h:
                n_to_1h = True
            feat_dic = read_feat_into_dic(pos_feat_in, ftype,
                                          feat_dic=feat_dic,
                                          n_to_1h=n_to_1h,
                                          label_list=feat_alphabet)
            feat_dic = read_feat_into_dic(neg_feat_in, ftype,
                                          feat_dic=feat_dic,
                                          n_to_1h=n_to_1h,
                                          label_list=feat_alphabet)
            assert feat_dic, "no .%s information read in (feat_dic empty)" %(fid)
            if fid == "elem_p.str" and args.str_elem_1h:
                ftype = "C"
            if ftype == "N":
                for c in feat_alphabet:
                    channel_nr += 1
                    channel_id = c
                    encoding = fid2norm_dic[fid]
                    channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, fid, encoding)
                    channel_info_list.append(channel_info)
            elif ftype == "C":
                for c in feat_alphabet:
                    channel_nr += 1
                    #channel_id = fid + "_" + c
                    channel_id = c
                    channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, fid)
                    channel_info_list.append(channel_info)
            else:
                assert False, "invalid feature type given (%s) for feature %s" %(ftype,fid)

    # Check for same feature vector lengths.
    fvl_dic = {}
    for seq_id in feat_dic:
        fvl_dic[len(feat_dic[seq_id][0])] = 1
    len_fvl_dic = len(fvl_dic)
    assert len_fvl_dic == 1, "Various feature vector lengths (%i) encountered in feat_dic" %(len_fvl_dic)

    # Write used features.out file to graphprot2 train output folder.
    feat_table_out = args.out_folder + "/" + "features.out"
    FEATOUT = open(feat_table_out, "w")
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            if feat_id in fid2type_dic:
                if feat_id == "fa":
                    if seqs_all_uc:
                        FEATOUT.write("fa\tC\tA,C,G,U\t-\n")
                    else:
                        FEATOUT.write("fa\tC\tA,C,G,U,a,c,g,u\t-\n")
                elif feat_id == "elem_p.str" and args.str_elem_1h:
                    FEATOUT.write("elem_p.str\tC\tE,H,I,M,S\t-\n")
                else:
                    FEATOUT.write("%s\n" %(row))
    f.closed
    FEATOUT.close()

    # Output channel infos.
    CIOUT = open(channel_infos_out, "w")
    CIOUT.write("ch\tch_id\tfeat_id\tfeat_type\tencoding\n")
    for ch_info in channel_info_list:
        CIOUT.write("%s\n" %(ch_info))
    CIOUT.close()

    """
    Generate graphs.

    """
    # Sequence ID list + label list.
    seq_ids_list = []
    sorted_pos_ids_list = []
    label_list = []
    idx2id_dic = {}
    id2idx_dic = {}
    i = 0
    for seq_id,c in sorted(pos_ids_dic.items()):
        seq_ids_list.append(seq_id)
        sorted_pos_ids_list.append(seq_id)
        label_list.append(1)
        id2idx_dic[seq_id] = i
        idx2id_dic[i] = seq_id
        i += 1
    for seq_id,c in sorted(neg_ids_dic.items()):
        seq_ids_list.append(seq_id)
        label_list.append(0)
        id2idx_dic[seq_id] = i
        idx2id_dic[i] = seq_id
        i += 1

    """
    In case of generic model cross validation (--gm-cv), create new label_list.
    Use n labels for n RBPs + + "0" label for negatives.

    """
    if args.gm_cv:
        # Seen labels (== RBP names) dictionary.
        label_dic = {}
        # Site ID to label dictionary.
        id2l_dic = {}
        # Label index.
        li = 0
        for seq_id in sorted_pos_ids_list:
            m = re.search("(.+?)_", seq_id)
            if m:
                label = m.group(1)
                if not label in label_dic:
                    li += 1
                label_dic[label] = li
                id2l_dic[seq_id] = li
                if li2label_dic is not None:
                    li2label_dic[li] = label
            else:
                assert False, "Generic data RBP label extraction failed for \"%s\"" % (seq_id)
        # Construct label list for positives.
        label_list = []
        for seq_id in sorted_pos_ids_list:
            label = id2l_dic[seq_id]
            labels.append(label)
        # Add negatives to label vector.
        label_list = label_list + [0]*len(neg_ids_dic)
        assert len(label_list) == len(seqs_dic), "len(label_list) != len(seqs_dic)"

    # Construct graphs list.
    all_graphs = []

    for idx, label in enumerate(label_list):
        seq_id = seq_ids_list[idx]
        seq = seqs_dic[seq_id]
        l_seq = len(seq)

        # Make edge indices for backbone.
        edge_index_1 = []
        edge_index_2 = []
        # Edge indices 0-based!
        for n_idx in range(l_seq - 1):
            edge_index_1.append(n_idx)
            edge_index_2.append(n_idx+1)
            # In case of undirected graphs, add backward edges too.
            if args.undirected:
                edge_index_1.append(n_idx+1)
                edge_index_2.append(n_idx)

        # Add base pair edges.
        if bpp_dic:
            vp_s = vp_dic[seq_id][0] # 1-based.
            vp_e = vp_dic[seq_id][1] # 1-based.
            # Entry e.g. 'CLIP_01': ['130-150,0.33', '160-200,0.44', '240-260,0.55']
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1)) # 1-based.
                p2 = int(m.group(2)) # 1-based.
                bpp_value = float(m.group(3))
                g_p1 = p1 - 1 # 0-based base pair index.
                g_p2 = p2 - 1 # 0-based base pair index.
                # Filter.
                if bpp_value < args.bps_cutoff: continue
                # Add base pair depending on set mode.
                add_edge = False
                if args.bps_mode == 1:
                    if (p1 >= vp_s and p1 <= vp_e) or (p2 >= vp_s and p2 <= vp_e):
                        add_edge = True
                elif args.bps_mode == 2:
                    if p1 >= vp_s and p2 <= vp_e:
                        add_edge = True
                if add_edge:
                    edge_index_1.append(g_p1)
                    edge_index_2.append(g_p2)
                    if args.undirected:
                        edge_index_1.append(g_p2)
                        edge_index_2.append(g_p1)

        # Merge edge indices.
        edge_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
        # x: node feature matrix with shape [num_nodes, num_node_features].
        x = torch.tensor(feat_dic[seq_id], dtype=torch.float)
        # Make a PyG Data object / graph from features and edges.
        # Data class description:
        # A plain old python object modeling a single graph with various (optional) attributes:
        # x, edge_index, edge_attr, y (Graph or node targets with arbitrary shape)
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
        data = Data(x=x, edge_index=edge_index, y=label)
        all_graphs.append(data)

    assert all_graphs, "no graphs constructed (all_graphs empty)"

    """
    ~~~ RETURNS ~~~

    seqs_dic:
        Sequences dictionary.
    idx2id_dic:
        list index (label_list and all_graphs list) to sequence ID
        mapping.
    label_list:
        Class label list (indices correspond to all_graphs list)
    all_graphs:
        List of PyG dataset graphs (indices correspond to label_list)

    """
    return seqs_dic, idx2id_dic, label_list, all_graphs


################################################################################

def load_geo_training_data(args,
                           data_id="data",
                           li2label_dic=None,
                           add_info_out=True):

    """
    Load training data from data folder generated by GraphProt2 gt and
    store data in PyG format.

    features.out example:
    fa	C	A,C,G,U	-
    bpp.str	N	bp_prob	prob
    elem_p.str	N	p_u,p_e,p_h,p_i,p_m,p_s	prob
    pc.con	N	phastcons_score	prob
    pp.con	N	phylop_score	mean
    tra	C	A,B,C,E,F,S,T,Z	-
    rra	C	N,R	-

    """

    # Checks.
    assert os.path.exists(args.in_folder), "--in folder does not exist"

    # Feature file containing info for features inside --in folder.
    feat_file = args.in_folder + "/" + "features.out"
    assert os.path.exists(feat_file), "%s features file expected but not does not exist" %(feat_file)

    # Data ID.
    data_id = data_id
    # graphprot2 train output folder.
    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # PyTorch geometric data output folder.
    data_out_folder = args.out_folder + "/" + data_id
    if not os.path.exists(data_out_folder):
        os.makedirs(data_out_folder)

    # Raw and processed data subfolders (structure dictated by PTG).
    raw_out_folder = data_out_folder + "/raw"
    proc_out_folder = data_out_folder + "/processed"
    if not os.path.exists(raw_out_folder):
        os.makedirs(raw_out_folder)
    if not os.path.exists(proc_out_folder):
        os.makedirs(proc_out_folder)

    # Output additional infos (sequences + IDs)?
    add_out_folder = data_out_folder + "/add_info"
    pos_add_ids_out_file = False
    neg_add_ids_out_file = False
    pos_add_seqs_out_file = False
    neg_add_seqs_out_file = False
    if add_info_out:
        if not os.path.exists(add_out_folder):
            os.makedirs(add_out_folder)
        pos_add_ids_out_file = add_out_folder + "/positives.add_info.ids"
        neg_add_ids_out_file = add_out_folder + "/negatives.add_info.ids"
        pos_add_seqs_out_file = add_out_folder + "/positives.add_info.fa"
        neg_add_seqs_out_file = add_out_folder + "/negatives.add_info.fa"

    # Delete old .pt files.
    processed_file = proc_out_folder + "/data.pt"
    if os.path.exists(processed_file):
        print("Remove existing data.pt file ... ")
        os.remove(processed_file)

    # Channel info output file.
    channel_infos_out = out_folder + "/" + "channel_infos.out"
    channel_info_list = []
    channel_nr = 0

    # Read in feature info.
    fid2type_dic = {}
    fid2cat_dic = {} # Store category labels or numerical score IDs in list.
    fid2norm_dic = {}
    print("Read in feature infos from %s ... " %(feat_file))
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            feat_type = cols[1]
            feat_cat_list = cols[2].split(",")
            feat_cat_list.sort()
            feat_norm = cols[3]
            assert feat_id not in fid2type_dic, "feature ID \"%s\" found twice in feature file" %(feat_id)
            fid2type_dic[feat_id] = feat_type
            fid2cat_dic[feat_id] = feat_cat_list
            fid2norm_dic[feat_id] = feat_norm
    f.closed
    assert fid2type_dic, "no feature IDs read in from feature file %s" %(feat_file)

    # Read in FASTA sequences.
    pos_fa_in = args.in_folder + "/" + "positives.fa"
    neg_fa_in = args.in_folder + "/" + "negatives.fa"
    assert os.path.exists(pos_fa_in), "--in folder does not contain %s"  %(pos_fa_in)
    assert os.path.exists(neg_fa_in), "--in folder does not contain %s"  %(neg_fa_in)

    # Check sequence feature.
    assert "fa" in fid2type_dic, "feature ID \"fa\" not in feature file"
    seqs_all_uc = False
    if fid2cat_dic["fa"] == ["A", "C", "G", "U"]:
        seqs_all_uc = True
    elif fid2cat_dic["fa"] == ["A", "C", "G", "U", "a", "c", "g", "u"]:
        if args.uc_context:
            seqs_all_uc = True
    else:
        assert False, "sequence feature alphabet != A,C,G,U(,a,c,g,u)"

    # Read in sequences.
    pos_seqs_dic = read_fasta_into_dic(pos_fa_in, all_uc=seqs_all_uc)
    neg_seqs_dic = read_fasta_into_dic(neg_fa_in, all_uc=seqs_all_uc)
    assert pos_seqs_dic, "no sequences read in from FASTA file \"%s\"" %(pos_fa_in)
    assert neg_seqs_dic, "no sequences read in from FASTA file \"%s\"" %(neg_fa_in)

    # Check for 4 (8) distinct nucleotides.
    pos_cc_dic = seqs_dic_count_chars(pos_seqs_dic)
    neg_cc_dic = seqs_dic_count_chars(neg_seqs_dic)
    allowed_nt_dic = {'A': 1, 'C': 1, 'G': 1, 'U': 1}
    c_nts = 4
    if not seqs_all_uc:
        allowed_nt_dic = {'A': 1, 'C': 1, 'G': 1, 'U': 1, 'a': 1, 'c': 1, 'g': 1, 'u': 1}
        c_nts = 8
    for nt in pos_cc_dic:
        if nt not in allowed_nt_dic:
            assert False, "positive sequences with invalid character \"%s\" encountered (allowed characters: ACGU(acgu)" %(nt)
    assert len(pos_cc_dic) == c_nts, "# of distinct nucleotide characters in positive set != expected # (%i != %i)" %(len(pos_cc_dic), c_nts)
    for nt in neg_cc_dic:
        if nt not in allowed_nt_dic:
            assert False, "negative sequences with invalid character \"%s\" encountered (allowed characters: ACGU(acgu)" %(nt)
    assert len(neg_cc_dic) == c_nts, "# of distinct nucleotide characters in negative set != expected # (%i != %i)" %(len(neg_cc_dic), c_nts)

    # Get con_ext info used to create dataset.
    con_ext_str = "False"
    gt_settings_file = args.in_folder + "/settings.graphprot2_gt.out"
    assert os.path.exists(gt_settings_file), "graphprot2 gt settings file %s not found" %(gt_settings_file)
    with open(gt_settings_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            if cols[0] == "con_ext":
                if cols[1] != "False":
                    con_ext_str = cols[1]
    f.closed
    if seqs_all_uc:
        con_ext_str = "False"
    # Add info to train settings file.
    train_settings_file = args.out_folder + "/settings.graphprot2_train.out"
    SETOUT = open(train_settings_file, "a")
    SETOUT.write("con_ext\t%s\n" %(con_ext_str))
    SETOUT.close()

    # Get uppercase (viewpoint) region start and ends for each sequence.
    pos_vp_dic = extract_uc_region_coords_from_fasta(pos_seqs_dic)
    neg_vp_dic = extract_uc_region_coords_from_fasta(neg_seqs_dic)

    # Additional features.
    pos_pc_con_in = args.in_folder + "/" + "positives.pc.con"
    neg_pc_con_in = args.in_folder + "/" + "negatives.pc.con"
    pos_pp_con_in = args.in_folder + "/" + "positives.pp.con"
    neg_pp_con_in = args.in_folder + "/" + "negatives.pp.con"
    pos_tra_in = args.in_folder + "/" + "positives.tra"
    neg_tra_in = args.in_folder + "/" + "negatives.tra"
    pos_rra_in = args.in_folder + "/" + "positives.rra"
    neg_rra_in = args.in_folder + "/" + "negatives.rra"
    pos_eia_in = args.in_folder + "/" + "positives.eia"
    neg_eia_in = args.in_folder + "/" + "negatives.eia"
    pos_str_elem_p_in = args.in_folder + "/" + "positives.elem_p.str"
    neg_str_elem_p_in = args.in_folder + "/" + "negatives.elem_p.str"
    pos_bpp_in = args.in_folder + "/" + "positives.bpp.str"
    neg_bpp_in = args.in_folder + "/" + "negatives.bpp.str"

    # Check for individually selected features.
    indiv_feat_dic = {}
    if args.use_pc_con:
        indiv_feat_dic["pc.con"] = 1
    if args.use_pp_con:
        indiv_feat_dic["pp.con"] = 1
    if args.use_eia:
        indiv_feat_dic["eia"] = 1
    if args.use_tra:
        indiv_feat_dic["tra"] = 1
    if args.use_rra:
        indiv_feat_dic["rra"] = 1
    if args.use_str_elem_p:
        indiv_feat_dic["elem_p.str"] = 1
    if args.use_bps:
        indiv_feat_dic["bpp.str"] = 1


    # Remove features from fid2type_dic.
    if indiv_feat_dic:
        del_feat_list = []
        for fid in fid2type_dic:
            if fid == "fa":
                continue
            if fid not in indiv_feat_dic:
                del_feat_list.append(fid)
        for fid in del_feat_list:
            del fid2type_dic[fid]

    if args.only_seq:
        fid2type_dic = {}
        fid2type_dic["fa"] = "C"

    # Dictionaries.
    pos_pc_con_dic = False
    neg_pc_con_dic = False
    pos_pp_con_dic = False
    neg_pp_con_dic = False
    pos_tra_dic = False
    neg_tra_dic = False
    pos_rra_dic = False
    neg_rra_dic = False
    pos_eia_dic = False
    neg_eia_dic = False
    pos_str_elem_p_dic = False
    neg_str_elem_p_dic = False
    pos_bpp_dic = False
    neg_bpp_dic = False

    # Check and read in data.
    for fid, ftype in sorted(fid2type_dic.items()):
        if fid == "fa":
            continue
        if fid == "bpp.str":
            assert os.path.exists(pos_bpp_in), "--in folder does not contain %s"  %(pos_bpp_in)
            assert os.path.exists(neg_bpp_in), "--in folder does not contain %s"  %(neg_bpp_in)
            print("Read in base pair data ... ")
            pos_bpp_dic = read_bpp_into_dic(pos_bpp_in, pos_vp_dic,
                                            bps_mode=args.bps_mode)
            neg_bpp_dic = read_bpp_into_dic(neg_bpp_in, neg_vp_dic,
                                            bps_mode=args.bps_mode)
        if fid == "eia":
            feat_id = "eia"
            eia_alphabet = fid2cat_dic[feat_id]
            assert os.path.exists(pos_eia_in), "--in folder does not contain %s"  %(pos_eia_in)
            assert os.path.exists(neg_eia_in), "--in folder does not contain %s"  %(neg_eia_in)
            print("Read in exon-intron annotations ... ")
            pos_eia_dic = read_feat_into_dic(pos_eia_in, "C",
                                             label_list=eia_alphabet)
            neg_eia_dic = read_feat_into_dic(neg_eia_in, "C",
                                             label_list=eia_alphabet)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = feat_id + "_" + c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, feat_id)
                channel_info_list.append(channel_info)

        if fid == "elem_p.str":
            feat_id = "elem_p.str"
            assert os.path.exists(pos_str_elem_p_in), "--in folder does not contain %s"  %(pos_str_elem_p_in)
            assert os.path.exists(neg_str_elem_p_in), "--in folder does not contain %s"  %(neg_str_elem_p_in)
            print("Read structural elements probabilities ... ")
            pos_str_elem_p_dic = read_str_elem_p_into_dic(pos_str_elem_p_in)
            neg_str_elem_p_dic = read_str_elem_p_into_dic(neg_str_elem_p_in)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[feat_id]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, feat_id, encoding)
                channel_info_list.append(channel_info)

        if fid == "pc.con":
            feat_id = "pc.con"
            assert os.path.exists(pos_pc_con_in), "--in folder does not contain %s"  %(pos_pc_con_in)
            assert os.path.exists(neg_pc_con_in), "--in folder does not contain %s"  %(neg_pc_con_in)
            print("Read in phastCons scores ... ")
            pos_pc_con_dic = read_con_into_dic(pos_pc_con_in)
            neg_pc_con_dic = read_con_into_dic(neg_pc_con_in)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[feat_id]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, feat_id, encoding)
                channel_info_list.append(channel_info)

        if fid == "pp.con":
            feat_id = "pp.con"
            assert os.path.exists(pos_pp_con_in), "--in folder does not contain %s"  %(pos_pp_con_in)
            assert os.path.exists(neg_pp_con_in), "--in folder does not contain %s"  %(neg_pp_con_in)
            print("Read in phyloP scores ... ")
            pos_pp_con_dic = read_con_into_dic(pos_pp_con_in)
            neg_pp_con_dic = read_con_into_dic(neg_pp_con_in)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = c
                encoding = fid2norm_dic[feat_id]
                channel_info = "%i\t%s\t%s\tN\t%s" %(channel_nr, channel_id, feat_id, encoding)
                channel_info_list.append(channel_info)

        if fid == "rra":
            feat_id = "rra"
            rra_alphabet = fid2cat_dic["rra"]
            assert os.path.exists(pos_rra_in), "--in folder does not contain %s"  %(pos_rra_in)
            assert os.path.exists(neg_rra_in), "--in folder does not contain %s"  %(neg_rra_in)
            print("Read in repeat region annotations ... ")
            pos_rra_dic = read_feat_into_dic(pos_rra_in, "C",
                                             label_list=rra_alphabet)
            neg_rra_dic = read_feat_into_dic(neg_rra_in, "C",
                                             label_list=rra_alphabet)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = feat_id + "_" + c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, feat_id)
                channel_info_list.append(channel_info)

        if fid == "tra":
            feat_id = "tra"
            tra_alphabet = fid2cat_dic["tra"]
            assert os.path.exists(pos_tra_in), "--in folder does not contain %s"  %(pos_tra_in)
            assert os.path.exists(neg_tra_in), "--in folder does not contain %s"  %(neg_tra_in)
            print("Read in transcript region annotations ... ")
            pos_tra_dic = read_feat_into_dic(pos_tra_in, "C",
                                             label_list=tra_alphabet)
            neg_tra_dic = read_feat_into_dic(neg_tra_in, "C",
                                             label_list=tra_alphabet)
            for c in fid2cat_dic[feat_id]:
                channel_nr += 1
                channel_id = feat_id + "_" + c
                channel_info = "%i\t%s\t%s\tC\tone_hot" %(channel_nr, channel_id, feat_id)
                channel_info_list.append(channel_info)

    # Write used features.out file to graphprot2 train output folder.
    feat_table_out = args.out_folder + "/" + "features.out"
    FEATOUT = open(feat_table_out, "w")
    with open(feat_file) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            feat_id = cols[0]
            if feat_id in fid2type_dic:
                if feat_id == "fa":
                    if seqs_all_uc:
                        FEATOUT.write("fa\tC\tA,C,G,U\t-\n")
                    else:
                        FEATOUT.write("fa\tC\tA,C,G,U,a,c,g,u\t-\n")
                else:
                    FEATOUT.write("%s\n" %(row))
    f.closed
    FEATOUT.close()

    # Output channel infos.
    CIOUT = open(channel_infos_out, "w")
    CIOUT.write("ch\tch_id\tfeat_id\tfeat_type\tencoding\n")
    for ch_info in channel_info_list:
        CIOUT.write("%s\n" %(ch_info))
    CIOUT.close()

    print("Convert data to PyG format ... ")

    # Convert data to PyTorch geometric format.
    pos_anl, pos_agi, pos_ae, pos_ana, g_idx, n_idx = generate_geometric_data(pos_seqs_dic,
                                                                pos_vp_dic,
                                                                pc_con_dic=pos_pc_con_dic,
                                                                pp_con_dic=pos_pp_con_dic,
                                                                eia_dic=pos_eia_dic,
                                                                rra_dic=pos_rra_dic,
                                                                tra_dic=pos_tra_dic,
                                                                str_elem_p_dic=pos_str_elem_p_dic,
                                                                bpp_dic=pos_bpp_dic,
                                                                add_ids_out_file=pos_add_ids_out_file,
                                                                add_seqs_out_file=pos_add_seqs_out_file,
                                                                bps_mode=args.bps_mode,
                                                                plfold_bpp_cutoff=args.bps_cutoff)
    neg_anl, neg_agi, neg_ae, neg_ana, g_idx, n_idx = generate_geometric_data(neg_seqs_dic,
                                                                neg_vp_dic,
                                                                pc_con_dic=neg_pc_con_dic,
                                                                pp_con_dic=neg_pp_con_dic,
                                                                eia_dic=neg_eia_dic,
                                                                rra_dic=neg_rra_dic,
                                                                tra_dic=neg_tra_dic,
                                                                str_elem_p_dic=neg_str_elem_p_dic,
                                                                bpp_dic=neg_bpp_dic,
                                                                add_ids_out_file=neg_add_ids_out_file,
                                                                add_seqs_out_file=neg_add_seqs_out_file,
                                                                bps_mode=args.bps_mode,
                                                                g_idx=g_idx,
                                                                n_idx=n_idx,
                                                                plfold_bpp_cutoff=args.bps_cutoff)

    # Create labels.
    labels = [1]*len(pos_seqs_dic) + [0]*len(neg_seqs_dic)

   # If --gm-cv, use n labels for n RBPs + + "0" label for negatives.
    if args.gm_cv:
        # Seen labels dictionary.
        label_dic = {}
        # Site ID to label dictionary.
        id2l_dic = {}
        # Label index.
        li = 0
        for seq_id, seq in sorted(pos_seqs_dic.items()):
            # Get RBP ID from seq_id.
                m = re.search("(.+?)_", seq_id)
                if m:
                    label = m.group(1)
                    if not label in label_dic:
                        li += 1
                    label_dic[label] = li
                    id2l_dic[seq_id] = li
                    if li2label_dic is not None:
                        li2label_dic[li] = label
                else:
                    assert False, "Generic data RBP label extraction failed for \"%s\"" % (seq_id)
        # Construct positives label vector.
        labels = []
        for seq_id, seq in sorted(pos_seqs_dic.items()):
            label = id2l_dic[seq_id]
            labels.append(label)
        # Add negatives to label vector.
        labels = labels + [0]*len(neg_seqs_dic)

    # Concatenate geometric lists.
    anl = pos_anl + neg_anl # node (nucleotide) labels.
    agi = pos_agi + neg_agi # graph indicators.
    ae = pos_ae + neg_ae # graph edges.
    ana = pos_ana + neg_ana # node attributes.

    print("Store PyG data on HD ... ")

    # RAW output files.
    agi_file = raw_out_folder + "/" + data_id + "_graph_indicator.txt"
    agl_file = raw_out_folder + "/" + data_id + "_graph_labels.txt"
    anl_file = raw_out_folder + "/" + data_id + "_node_labels.txt"
    ana_file = raw_out_folder + "/" + data_id + "_node_attributes.txt"
    ae_file = raw_out_folder + "/" + data_id + "_A.txt"
    # Write to files.
    f = open(agi_file, 'w')
    f.writelines([str(e) + "\n" for e in agi])
    f.close()
    f = open(agl_file, 'w')
    f.writelines([str(e) + "\n" for e in labels])
    f.close()
    f = open(anl_file, 'w')
    f.writelines([str(e) + "\n" for e in anl])
    f.close()
    if ana:
        f = open(ana_file, 'w')
        f.writelines([s + "\n" for s in ana])
        f.close()
    else:
        if os.path.exists(ana_file):
            os.remove(ana_file)
    f = open(ae_file, 'w')
    f.writelines([str(e[0]) + ", " + str(e[1]) + "\n" for e in ae])
    f.close()


################################################################################

def generate_geometric_data(seqs_dic, vp_dic,
                            pc_con_dic=False,
                            pp_con_dic=False,
                            eia_dic=False,
                            rra_dic=False,
                            tra_dic=False,
                            str_elem_p_dic=False,
                            bpp_dic=False,
                            plfold_bpp_cutoff=0.2,
                            g_idx=False,
                            n_idx=False,
                            add_fake_g=False,
                            fake_g_lcuc=False,
                            add_ids_out_file=False,
                            add_seqs_out_file=False,
                            bps_mode=1):
    """
    Generate PyTorch Geometric graph format data.

    seqs_dic:
        Sequence ID -> sequence dictionary.
    vp_dic:
        Sequence ID -> [viewpont (uppercase) start, viewpoint end] dictionary.

    Return the following lists:
    all_nodes_labels       Nucleotide indices (dict_label_idx)
    all_graph_indicators   Graph indices, each node of a graph
                           gets same index
    all_edges              Indices of edges
    all_nodes_attributes   Node vectors
    Also return the latest graph and node indices:
    g_idx
    n_idx

    """

    # Checks.
    assert seqs_dic, "seqs_dic empty"
    # Nucleotide label to idx dictionary.
    dict_label_idx = {'A': '1',
                      'C': '2',
                      'G': '3',
                      'U': '4',
                      'a': '5',
                      'c': '6',
                      'g': '7',
                      'u': '8'}
    # Init lists.
    all_nodes_labels = []
    all_graph_indicators = []
    all_edges = []
    all_nodes_attributes = []

    # Init graph and node indices (if not given).
    if not g_idx:
        g_idx = 0
    if not n_idx:
        n_idx = 1

    # Store sequence ID -> site sequence.
    seqs_out_dic = {}
    # Sequence ID -> graph index.
    id2gidx_dic = {}

    for seq_id, seq in sorted(seqs_dic.items()):
        # Uppercase region (viewpoint) start + end + lengths.
        vp_s = vp_dic[seq_id][0] # 1-based.
        vp_e = vp_dic[seq_id][1] # 1-based.
        l_vp = vp_e - vp_s + 1
        l_seq = len(seq)
        # Length of graph.
        n_nodes = l_seq
        # Add graph indicator labels (length of graph).
        all_graph_indicators.extend([g_idx+1]*n_nodes)

        # Check for ID in given dictionaries.
        if bpp_dic:
            assert seq_id in bpp_dic, "sequence ID \"%s\" not in bpp_dic" %(seq_id)
        if eia_dic:
            assert seq_id in eia_dic, "sequence ID \"%s\" not in eia_dic" %(seq_id)
        if pc_con_dic:
            assert seq_id in pc_con_dic, "sequence ID \"%s\" not in pc_con_dic" %(seq_id)
        if pp_con_dic:
            assert seq_id in pp_con_dic, "sequence ID \"%s\" not in pp_con_dic" %(seq_id)
        if rra_dic:
            assert seq_id in rra_dic, "sequence ID \"%s\" not in rra_dic" %(seq_id)
        if str_elem_p_dic:
            assert seq_id in str_elem_p_dic, "sequence ID \"%s\" not in str_elem_p_dic" %(seq_id)
        if tra_dic:
            assert seq_id in tra_dic, "sequence ID \"%s\" not in tra_dic" %(seq_id)

        # Add feature values per position.
        g_i = 0
        for i,c in enumerate(seq): # i from 0.. l-1
            if seq_id in seqs_out_dic:
                seqs_out_dic[seq_id] += c
            else:
                seqs_out_dic[seq_id] = c
            all_nodes_labels.append(dict_label_idx[c])

            # Make feature vector [0,1, ..] for each graph node.
            feat_vector = []
            if eia_dic:
               feat_vector = feat_vector + eia_dic[seq_id][i]
            if str_elem_p_dic:
               feat_vector = feat_vector + str_elem_p_dic[seq_id][i]
            if pc_con_dic:
                feat_vector.append(pc_con_dic[seq_id][i])
            if pp_con_dic:
                feat_vector.append(pp_con_dic[seq_id][i])
            if rra_dic:
               feat_vector = feat_vector + rra_dic[seq_id][i]
            if tra_dic:
               feat_vector = feat_vector + tra_dic[seq_id][i]

            # Convert list elements to string and save in new list.
            if feat_vector:
                node_attribute = [str(att) for att in feat_vector]
                # Join elements separate by , to string and append to list.
                all_nodes_attributes.append(",".join(node_attribute))

            # Add backbone edge.
            if g_i > 0:
                all_edges.append((g_i-1+n_idx, g_i+n_idx))
                all_edges.append((g_i+n_idx, g_i-1+n_idx))
            # Increment graph node index.
            g_i += 1

        # Add base pair edges to graph.
        if bpp_dic:
            for entry in bpp_dic[seq_id]:
                m = re.search("(\d+)-(\d+),(.+)", entry)
                p1 = int(m.group(1))
                p2 = int(m.group(2))
                bpp_value = float(m.group(3))
                g_p1 = p1 - 1 # 0-based base pair index.
                g_p2 = p2 - 1 # 0-based base pair index.
                # Filter.
                if bpp_value < plfold_bpp_cutoff: continue
                # Add base pair depending on set mode.
                if bps_mode == 1:
                    if (p1 >= vp_s and p1 <= vp_e) or (p2 >= vp_s and p2 <= vp_e):
                        all_edges.append((g_p1+n_idx, g_p2+n_idx))
                        all_edges.append((g_p2+n_idx, g_p1+n_idx))
                elif bps_mode == 2:
                    if p1 >= vp_s and p2 <= vp_e:
                        all_edges.append((g_p1+n_idx, g_p2+n_idx))
                        all_edges.append((g_p2+n_idx, g_p1+n_idx))
                else:
                    assert False, "ERROR: invalid bps_mode given (valid values: 1,2)"

        # Update node and graph indices.
        n_idx += n_nodes
        g_idx += 1
        # Store sequence ID -> graph index.
        id2gidx_dic[seq_id] = g_idx

    assert g_idx, "no graphs added (g_idx == 0)"

    # Output additional infos.
    if add_ids_out_file:
        ADDOUT = open(add_ids_out_file, "w")
        for seq_id, gidx in sorted(id2gidx_dic.items()):
            ADDOUT.write("%s\t%i\n" %(seq_id, id2gidx_dic[seq_id]))
        ADDOUT.close()
    if add_seqs_out_file:
        fasta_output_dic(seqs_out_dic, add_seqs_out_file,
                         split=True,
                         split_size=60)

    # If double taking jive fake graph should be added.
    if add_fake_g:
        # Fake sequence.
        fake_seq = "ACGU"
        if fake_g_lcuc:
            fake_seq = "ACGUacgu"
        # Node indicators.
        n_nodes = len(fake_seq)
        all_graph_indicators.extend([g_idx+1]*n_nodes)
        # Node labels and edges.
        g_i = 0
        for i,c in enumerate(fake_seq):
            all_nodes_labels.append(dict_label_idx[c])
            # Add backbone edges.
            if g_i > 0:
                all_edges.append((g_i-1+n_idx, g_i+n_idx))
                all_edges.append((g_i+n_idx, g_i-1+n_idx))
            # Increment graph node index.
            g_i += 1
        # Node attributes.
        if all_nodes_attributes:
            # Add first element n_nodes times.
            for i in range(n_nodes):
                all_nodes_attributes.append(all_nodes_attributes[0])
        # Update node and graph indices.
        n_idx += n_nodes
        g_idx += 1

    return all_nodes_labels, all_graph_indicators, all_edges, all_nodes_attributes, g_idx, n_idx


################################################################################

def decompose_node_attr(x):
    """
    Get feature lists for one graph, by decomposing its node attributes list.

    """
    x = x.tolist()
    dict_onehot_label = {'1000': 1, '0100': 2, '0010': 3, '0001': 4}
    dict_onehot_nucleotide = {'1000': 'A', '0100': 'C', '0010': 'G', '0001': 'U'}
    dict_reg_exon_intron = {1: 'E', 0: 'I'}
    list_nucleotides = []
    list_attrs = []
    list_node_labels = []
    list_cons_1 = []
    list_cons_2 = []
    list_exons_introns = []

    for idx in range(len(x)):
        attrs = [str(att) for att in x[idx][:4]]  # first 4 elements.
        onehot = "".join([str(int(i)) for i in x[idx][4:]]) # Elements 5,6 ...
        list_attrs.append(",".join(attrs))
        list_node_labels.append(dict_onehot_label[onehot])
        list_nucleotides.append(dict_onehot_nucleotide[onehot])
        list_cons_1.append(str(x[idx][0]))
        list_cons_2.append(str(x[idx][1]))
        list_exons_introns.append(dict_reg_exon_intron[int(x[idx][2])])

    return list_attrs, list_node_labels, list_nucleotides, list_exons_introns, list_cons_1, list_cons_2


################################################################################

def read_settings_into_dic(settings_file):
    """
    Read settings file content into dictionary.
    Each row expected to have following format:
    setting_id<tab>setting_value
    Skip rows with > 2 entries.
    Dictionary format: str(col1) -> str(col2)

    >>> test_in = "test_data/test_settings.out"
    >>> read_settings_into_dic(test_in)
    {'peyote': '20.5', 'china_white': '43.1'}

    """
    assert settings_file, "file name expected"
    assert os.path.isfile(settings_file), "file %s does not exist" %(settings_file)
    set_dic = {}
    with open(settings_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            settings_id = cols[0]
            settings_val = cols[1]
            if settings_id not in set_dic:
                set_dic[settings_id] = settings_val
            else:
                assert False, "settings ID %s appears > 1 in given settings file" %(settings_id)
    f.closed
    assert set_dic, "set_dic empty (nothing read in?)"
    return set_dic


################################################################################

def read_feat_into_dic(feat_file, feat_type,
                       feat_dic=False,
                       n_to_1h=False,
                       label_list=False):
    """
    Read in feature data from feat_file into dictionary of lists.
    Mapping: sequence ID -> list of labels

    feat_type:
        Type of feature, set "C" for categorical and "N" for numerical
    label_list:
        Needed for C feature, supply label_list to do one-hot encoding
    n_to_1h:
        For structural elements probabilities, to convert them into
        one-hot encodings.

    1) Categorical data (C)
    Categorical (feat_type == C) data example, with label_list = ['E', 'I']:
    CLIP_1	EI
    CLIP_2	IE
    Generated one-hot lists:
    [[1, 0], [0, 1]]
    [[0, 1], [1, 0]]
    Generated dictionary:
    {'CLIP_1': [[1, 0], [0, 1]], 'CLIP_2': [[0, 1], [1, 0]]}

    2) Numerical data (N)
    Numerical (feat_type == N) data example:
    >CLIP_1
    0.1
    -0.2
    >CLIP_2
    0.4
    0.2
    Generated lists:
    [[0.1], [-0.2]]
    [[0.4], [0.2]]
    Generated dictionary:
    {'CLIP_1': [[0.1], [-0.2]], 'CLIP_2': [[0.4], [0.2]]}

    test.pp.con:
    >CLIP_01
    0.1
    0.2
    >CLIP_02
    0.4
    0.5

    test2.pp.con:
    >CLIP_01
    0.1	0.2
    0.3	0.4
    >CLIP_02
    0.5	0.6
    0.7	0.8

    >>> num_test_in = "test_data/test.pp.con"
    >>> read_feat_into_dic(num_test_in, "N")
    {'CLIP_01': [[0.1], [0.2]], 'CLIP_02': [[0.4], [0.5]]}
    >>> num_test_in = "test_data/test2.pp.con"
    >>> read_feat_into_dic(num_test_in, "N")
    {'CLIP_01': [[0.1, 0.2], [0.3, 0.4]], 'CLIP_02': [[0.5, 0.6], [0.7, 0.8]]}
    >>> add_feat_dic = {'CLIP_01': [[0.1], [0.2]], 'CLIP_02': [[0.4], [0.5]]}
    >>> num_test_in = "test_data/test.pp.con"
    >>> read_feat_into_dic(num_test_in, "N", feat_dic=add_feat_dic)
    {'CLIP_01': [[0.1, 0.1], [0.2, 0.2]], 'CLIP_02': [[0.4, 0.4], [0.5, 0.5]]}
    >>> cat_test_in = "test_data/test.tra"
    >>> tra_labels = ['C', 'F', 'N', 'T']
    >>> read_feat_into_dic(cat_test_in, "C", label_list=tra_labels)
    {'site1': [[0, 1, 0, 0], [0, 0, 0, 1]], 'site2': [[1, 0, 0, 0], [0, 0, 1, 0]]}

    """
    feat_dic_given = False
    if not feat_dic:
        feat_dic = {}
    else:
        feat_dic_given = True
    types = ['C', 'N']
    assert feat_type in types, "invalid feature type given (expects C or N)"
    if feat_type == 'C':
        assert label_list, "label_list needed if feat_type == C"
        with open(feat_file) as f:
            for line in f:
                cols = line.strip().split("\t")
                seq_id = cols[0]
                if seq_id not in feat_dic:
                    feat_dic[seq_id] = string_vectorizer(cols[1], custom_alphabet=label_list)
                else:
                    # feat_dic already populated / initialized.
                    add_list = string_vectorizer(cols[1], custom_alphabet=label_list)
                    assert add_list, "add_list empty (feat_file: %s, seq_id: %s)" %(feat_file, seq_id)
                    # Check.
                    l_old = len(feat_dic[seq_id])
                    l_add = len(add_list)
                    assert l_old == l_add, "existing list length in feat_dic != list length from feat_file to add (feat_file: %s, seq_id: %s)" %(feat_file, seq_id)
                    for i in range(l_old):
                        feat_dic[seq_id][i] += add_list[i]
        f.closed
    else:
        seq_id = ""
        pos_i = 0
        with open(feat_file) as f:
            for line in f:
                if re.search(">.+", line):
                    m = re.search(">(.+)", line)
                    seq_id = m.group(1)
                    # Init only necessary if no populated / initialized feat_dic given.
                    if not feat_dic_given:
                        feat_dic[seq_id] = []
                    pos_i = 0
                else:
                    vl = line.strip().split('\t')
                    for i,v in enumerate(vl):
                        vl[i] = float(v)
                    if n_to_1h:
                        vl_1h = convert_prob_list_to_1h(vl)
                        vl = vl_1h
                    if feat_dic_given:
                        for v in vl:
                            feat_dic[seq_id][pos_i].append(v)
                    else:
                        feat_dic[seq_id].append(vl)
                    pos_i += 1
        f.closed
    assert feat_dic, "feat_dic empty"
    return feat_dic


################################################################################

def revise_in_sites(in_bed, out_bed,
                    chr_len_dic, id2pl_dic, args,
                    transcript_regions=False):

    """
    Revise positive or negative sites as part of graphprot2 gt.
    Output rows with zero values in column 5, store original rows in
    dictionary. Return this site ID to row dictionary.
    Zero scores are necessary since twoBitToFa despises decimal scores.

    id2pl_dic:
        If given, store part lengths (lower case, uppercase, lowercase) in
        given dictionary.

    """
    # Checks.
    assert chr_len_dic, "chr_len_dic empty"
    assert id2pl_dic, "id2pl_dic empty"
    # Site ID to original row ID with scores dictionary.
    id2row_dic = {}
    # Store revised sites in output BED file.
    BEDOUT = open(out_bed, "w")
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_sc = cols[4]
            site_pol = cols[5]
            if not args.pre_con_ext_merge and args.con_ext:
                seq_ext_s = site_s
                seq_ext_e = site_e
                new_s = site_s - args.con_ext
                new_e = site_e + args.con_ext
                # Check for sequence ends.
                if transcript_regions:
                    # Truncate sites at transcript ends.
                    if new_s < 0:
                        new_s = 0
                    if new_e > chr_len_dic[chr_id]:
                        new_e = chr_len_dic[chr_id]
                else:
                    # Discard sites at chromosome ends.
                    if new_s < 0:
                        if site_id in id2pl_dic:
                            del id2pl_dic[site_id]
                        continue
                    if new_e > chr_len_dic[chr_id]:
                        if site_id in id2pl_dic:
                            del id2pl_dic[site_id]
                        continue
                us_lc_len = seq_ext_s - new_s
                ds_lc_len = new_e - seq_ext_e
                id2pl_dic[site_id][0] = us_lc_len
                id2pl_dic[site_id][2] = ds_lc_len
            else:
                new_s = site_s
                new_e = site_e
            site_len = new_e - new_s
            # Since score can be decimal, convert to 0 (twoBitToFa despises decimal scores).
            new_sc = "0"
            # Store old row.
            old_row = "%s\t%i\t%i\t%s\t%s\t%s" %(chr_id, new_s, new_e, site_id, site_sc, site_pol)
            new_row = "%s\t%i\t%i\t%s\t%s\t%s" %(chr_id, new_s, new_e, site_id, new_sc, site_pol)
            id2row_dic[site_id] = old_row
            BEDOUT.write("%s\n" %(new_row))
    f.closed
    BEDOUT.close()
    return id2row_dic


################################################################################

def process_test_sites(in_bed, out_bed, chr_len_dic,
                       id2pl_dic, args,
                       check_ids=False,
                       transcript_regions=False,
                       count_dic=None,
                       id_prefix=False):
    """
    Process --in sites from graphprot2 gp.

    """
    # Checks.
    assert chr_len_dic, "chr_len_dic empty"
    # Store BED rows with scores.
    id2row_dic = {}
    # Filtered output BED file, with "0" scores for sequence extraction.
    BEDOUT = open(out_bed, "w")
    # Counts.
    c_in = 0
    c_filt_ref = 0
    c_out = 0
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_sc = float(cols[4])
            site_pol = cols[5]
            site_len = site_e - site_s
            # Some checks.
            assert chr_id in chr_len_dic, "chromosome ID \"%s\" not in chr_len_dic" %(chr_id)
            assert site_pol == "+" or site_pol == "-", "invalid strand info (column 6) given in --in line \"%s\"" %(row)
            assert site_e > site_s, "--in site end <= site start (%i <= %i, site_id: %s)" %(site_e, site_s, site_id)
            # Check for valid coordinates (not outside chromosome).
            assert site_s >= 0, "--in site start < 0 (site_id: %s)" %(site_id)
            assert site_e <= chr_len_dic[chr_id], "--in site end > reference sequence length (%i > %i, site_id: %s, ref_id: %s)" %(site_e, chr_len_dic[chr_id], site_id, chr_id)
            c_in += 1
            # Restrict to standard chromosomes.
            if not transcript_regions:
                new_chr_id = check_convert_chr_id(chr_id)
                if not new_chr_id:
                    c_filt_ref += 1
                    continue
                else:
                    chr_id = new_chr_id
            # Make site polarities "+" for transcript sites.
            if transcript_regions:
                site_pol = "+"
            new_s = site_s
            new_e = site_e

            # IDs.
            c_out += 1
            # Remove white spaces from IDs.
            if check_ids:
                site_id = site_id.strip().replace(" ", "_")
            # New IDs.
            if id_prefix:
                new_site_id = id_prefix + "_" + str(c_out)
            else:
                new_site_id = site_id

            # Store future uppercase region length.
            id2pl_dic[new_site_id] = [0, site_len, 0]
            # --con-ext lowercase context extension.
            if args.con_ext:
                seq_ext_s = new_s
                seq_ext_e = new_e
                new_s = new_s - args.con_ext
                new_e = new_e + args.con_ext
                # Truncate sites at reference ends.
                if new_s < 0:
                    new_s = 0
                if new_e > chr_len_dic[chr_id]:
                    new_e = chr_len_dic[chr_id]
                us_lc_len = seq_ext_s - new_s
                ds_lc_len = new_e - seq_ext_e
                id2pl_dic[new_site_id][0] = us_lc_len
                id2pl_dic[new_site_id][2] = ds_lc_len
                if site_pol == "-":
                    id2pl_dic[new_site_id][0] = ds_lc_len
                    id2pl_dic[new_site_id][2] = us_lc_len

            # Check whether score is whole number.
            if not site_sc % 1:
                site_sc = int(site_sc)
            # Convert to string.
            site_sc = str(site_sc)
            new_sc = "0"
            # Store and print out sites.
            old_row = "%s\t%i\t%i\t%s\t%s\t%s" %(chr_id, new_s, new_e, new_site_id, site_sc, site_pol)
            new_row = "%s\t%i\t%i\t%s\t%s\t%s" %(chr_id, new_s, new_e, new_site_id, new_sc, site_pol)
            id2row_dic[new_site_id] = old_row
            BEDOUT.write("%s\n" %(new_row))
    f.closed
    BEDOUT.close()
    # Count stats dic.
    if count_dic is not None:
        count_dic['c_in'] = c_in
        count_dic['c_filt_ref'] = c_filt_ref
        count_dic['c_out'] = c_out
    assert id2row_dic, "id2row_dic empty"
    return id2row_dic


################################################################################

def process_in_sites(in_bed, out_bed, chr_len_dic, args,
                     transcript_regions=False,
                     id2pl_dic=None,
                     count_dic=None,
                     id_prefix="CLIP"):
    """
    Process --in or --neg-in sites as part of graphprot2 gt.
    Return dictionary of lists, with
    upstream lowercase length, uppercase length, downstream lowercase length
    for every site ID. E.g.
    {'id1' : [150, 61, 150], 'id2' : [100, 61, 150]}

    id2pl_dic:
        Site ID to part lengths list dictionary.
    count_dic:
        Count stats dictionary.

    """
    # Checks.
    assert chr_len_dic, "chr_len_dic empty"
    # Filtered output BED file.
    BEDOUT = open(out_bed, "w")
    # Part lengths dictionary.
    if id2pl_dic is None:
        id2pl_dic = {}
    # Min length ext.
    min_len_ext = int( (args.min_len - 1) / 2)
    # Counts.
    c_in = 0
    c_filt_max_len = 0
    c_filt_ref = 0
    c_filt_thr = 0
    c_chr_ends = 0
    c_out = 0
    with open(in_bed) as f:
        for line in f:
            row = line.strip()
            cols = line.strip().split("\t")
            chr_id = cols[0]
            site_s = int(cols[1])
            site_e = int(cols[2])
            site_id = cols[3]
            site_sc = float(cols[4])
            site_pol = cols[5]
            site_len = site_e - site_s
            # Checks.
            assert chr_id in chr_len_dic, "chromosome ID \"%s\" not in chr_len_dic" %(chr_id)
            assert site_pol == "+" or site_pol == "-", "invalid strand info (column 6) given in --in line \"%s\"" %(row)
            assert site_e > site_s, "--in site end <= site start (%i <= %i, site_id: %s)" %(site_e, site_s, site_id)
            # Check for valid coordinates (not outside chromosome).
            assert site_s >= 0, "--in site start < 0 (site_id: %s)" %(site_id)
            assert site_e <= chr_len_dic[chr_id], "--in site end > reference sequence length (%i > %i, site_id: %s, ref_id: %s)" %(site_e, chr_len_dic[chr_id], site_id, chr_id)
            c_in += 1
            # Filter by max_len.
            if site_len > args.max_len:
                c_filt_max_len += 1
                continue
            # Filter by score.
            if args.sc_thr is not None:
                if args.rev_filter:
                    if site_sc > args.sc_thr:
                        c_filt_thr += 1
                        continue
                else:
                    if site_sc < args.sc_thr:
                        c_filt_thr += 1
                        continue
            # Restrict to standard chromosomes.
            if not transcript_regions:
                new_chr_id = check_convert_chr_id(chr_id)
                if not new_chr_id:
                    c_filt_ref += 1
                    continue
                else:
                    chr_id = new_chr_id
            # Make site polarities "+" for transcript sites.
            if transcript_regions:
                site_pol = "+"
            # Process site coordinates according to set parameters (mode, ...).
            new_s = site_s
            new_e = site_e
            # 1: Take the center of each site.
            # 2: Take the complete site.
            # 3: Take the upstream end for each site.
            if args.mode == 1:
                # Take center position.
                new_e = get_center_position(site_s, site_e)
                new_s = new_e - 1
            elif args.mode == 2:
                # Take complete site, unless  --min-len or --max-len applies.
                if site_len < args.min_len:
                    new_e = get_center_position(site_s, site_e)
                    new_s = new_e - min_len_ext - 1
                    new_e = new_e + min_len_ext
            elif args.mode == 3:
                new_s = site_s
                new_e = site_s + 1
                if site_pol == "-":
                    new_s = site_e - 1
                    new_e = site_e
            else:
                assert False, "invalid mode set (args.mode value == %i)" %(args.mode)
            # Extend.
            new_s = new_s - args.seq_ext
            new_e = new_e + args.seq_ext
            # Truncate sites at reference ends.
            if new_s < 0:
                new_s = 0
            if new_e > chr_len_dic[chr_id]:
                new_e = chr_len_dic[chr_id]
            # IDs.
            c_out += 1
            new_site_id = id_prefix + "_" + str(c_out)
            if args.keep_ids:
                new_site_id = site_id
            # Site lengths.
            seq_ext_len = new_e - new_s
            # Store future uppercase region length.
            id2pl_dic[new_site_id] = [0, seq_ext_len, 0]
            # If context extension should be applied before merging overlapping sites.
            if args.pre_con_ext_merge:
                if args.con_ext:
                    seq_ext_s = new_s
                    seq_ext_e = new_e
                    new_s = new_s - args.con_ext
                    new_e = new_e + args.con_ext
                    # Truncate sites at reference ends.
                    if new_s < 0:
                        new_s = 0
                    if new_e > chr_len_dic[chr_id]:
                        new_e = chr_len_dic[chr_id]
                    us_lc_len = seq_ext_s - new_s
                    ds_lc_len = new_e - seq_ext_e
                    id2pl_dic[new_site_id][0] = us_lc_len
                    id2pl_dic[new_site_id][2] = ds_lc_len
                    if site_pol == "-":
                        id2pl_dic[new_site_id][0] = ds_lc_len
                        id2pl_dic[new_site_id][2] = us_lc_len

            # Store site length in list.
            new_site_len = new_e - new_s
            # Check whether score is whole number.
            if not site_sc % 1:
                site_sc = int(site_sc)
            # Convert to string.
            site_sc = str(site_sc)
            # Print out sites.
            BEDOUT.write("%s\t%i\t%i\t%s\t%s\t%s\n" %(chr_id, new_s, new_e, new_site_id, site_sc, site_pol) )
    f.closed
    BEDOUT.close()
    # Count stats dic.
    if count_dic is not None:
        count_dic['c_in'] = c_in
        count_dic['c_filt_max_len'] = c_filt_max_len
        count_dic['c_filt_thr'] = c_filt_thr
        count_dic['c_filt_ref'] = c_filt_ref
        count_dic['c_out'] = c_out
    return id2pl_dic


################################################################################

def scores_to_plot_df(scores,
                      stdev=False):
    """
    Given a list of scores, generate a dataframe with positions from 1 to
    length(scores_list) and scores list.

    Dictionary of lists intermediate
    data = {'pos': [1,2,3], 'scores': [0.2,0.4,0.5]}
    Then create dataframe with
    pd.DataFrame (data, columns = ['pos', 'scores'])

    stdev:
        Vector of standard deviations belonging to scores.

    """
    assert scores, "given scores list empty"
    if stdev:
        assert len(scores) == len(stdev), "len(scores) != len(stdev)"
        data = {'pos': [], 'score': [], 'stdev': []}
    else:
        data = {'pos': [], 'score': []}
    for i,s in enumerate(scores):
        data['pos'].append(i+1)
        data['score'].append(s)
        if stdev:
            data['stdev'].append(stdev[i])
    if stdev:
        plot_df = pd.DataFrame(data, columns = ['pos', 'score', 'stdev'])
    else:
        plot_df = pd.DataFrame(data, columns = ['pos', 'score'])
    return plot_df


################################################################################

def convert_prob_list_to_1h(lst):
    """
    Convert list of probabilities or score values into one-hot encoding list,
    where element with highest prob./score gets 1, others 0.

    >>> lst = [0.3, 0.5, 0.2, 0.1, 0.1]
    >>> convert_prob_list_to_1h(lst)
    [0, 1, 0, 0, 0]

    """
    assert lst, "given lst empty"
    new_lst = [0]*len(lst)
    max_i = 0
    max_e = 0
    for i,e in enumerate(lst):
        if e > max_e:
            max_e = e
            max_i = i
    new_lst[max_i] = 1
    return new_lst


################################################################################

def seq_to_plot_df(seq, alphabet,
                   default_score=1,
                   scores=False):
    """
    Given a sequence, generate a pandas dataframe from it.

    Format example:
    sequence = "AACGT"
    alphabet = ["A", "C", "G", "T"]
    Intermediate dictionary of lists:
    data = {'A' : [1,1,0,0,0], 'C' : [0,0,1,0,0], 'G' : [0,0,0,1,0], 'T' : [0,0,0,0,1]}
    Final dataframe:
         A  C  G  T
    pos
    0    1  0  0  0
    1    1  0  0  0
    2    0  1  0  0
    3    0  0  1  0
    4    0  0  0  1

    seq:
        Sequence string to generate pandas dataframe for logo
        generation from.
    alphabet:
        List of sequence characters to consider for logo generation.
    scores:
        Scores list, to use instead of score of 1 for nucleotide.

    """
    assert seq, "empty sequence given"
    assert alphabet, "alphabet character list empty"
    if scores:
        assert len(seq) == len(scores), "length scores list != length sequence"
    alphabet.sort()
    data = {}
    for c in alphabet:
        data[c] = []
    for i,sc in enumerate(seq):
        assert sc in alphabet, "sequence character \"%s\" not in given alphabet" %(sc)
        score = default_score
        if scores:
            score = scores[i]
        for c in alphabet:
            if c == sc:
                data[c].append(score)
            else:
                data[c].append(0)
    plot_df = pd.DataFrame(data, columns = alphabet)
    plot_df.index.name = "pos"
    return plot_df


################################################################################

def add_importance_scores_plot(df, fig, gs, i,
                               color_dict=False,
                               y_label_size=9):
    """
    Make nucleotide importance scores plot.
    Normalized profile scores range from -1 .. 1.

    """
    ax = fig.add_subplot(gs[i, :])
    if color_dict:
        logo = logomaker.Logo(df, ax=ax, color_scheme=color_dict)
    else:
        logo = logomaker.Logo(df, ax=ax)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=True, bounds=[-1, 1])
    logo.style_spines(spines=['bottom'], visible=False)
    logo.ax.set_xticks([])
    logo.ax.set_yticks([-1, 0, 1])
    # plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    # ax.yaxis.set_tick_params(labelsize=7)
    logo.ax.set_yticklabels(['-1', '0', '1'])
    logo.ax.set_ylabel('score', labelpad=10, fontsize=y_label_size)


################################################################################

def add_label_plot(df, fig, gs, i,
                   color_dict=False,
                   y_label_size=9,
                   y_label="exon-intron"):
    """
    Make exon-intron label plot.

    """
    ax = fig.add_subplot(gs[i, :])
    if color_dict:
        logo = logomaker.Logo(df, ax=ax, vpad=0.4, color_scheme=color_dict)
    else:
        logo = logomaker.Logo(df, ax=ax, vpad=0.4)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=False, bounds=[0, 1])
    logo.style_spines(spines=['bottom'], visible=False)
    logo.ax.set_xticks([])
    logo.ax.set_yticks([])
    #logo.ax.set_axis_off()
    #logo.ax.set_yticklabels(['0', '1'])
    plt.yticks(fontsize=7)
    logo.ax.set_ylabel(y_label, labelpad=24, fontsize=y_label_size)


################################################################################

def make_motif_label_plot_df(feat_id, ch_info_dic, motif_matrix):
    """
    Make a feature dataframe for label data motif plotting.
    Label data: sequence, eia, tra, rra, elem_p.str,
                and additional categorical features

    """
    data = {}
    assert feat_id in ch_info_dic, "feat_id %s not in ch_info_dic" %(feat_id)
    feat_data = ch_info_dic[feat_id]
    feat_idxs = feat_data[1]
    feat_alphabet = feat_data[2]
    for c in feat_alphabet:
        data[c] = []
    for fv in motif_matrix:
        for i,fi in enumerate(feat_idxs):
            data[feat_alphabet[i]].append(fv[fi])

    feat_plot_df = pd.DataFrame(data, columns = feat_alphabet)
    feat_plot_df.index.name = "pos"
    return feat_plot_df


################################################################################

def make_motif_scores_plot_df(feat_id, ch_info_dic, motif_matrix,
                              stdev=False):
    """
    Make a feature dataframe for scores data motif plotting.
    Scores data: pc.con, pp.con, additional numerical features

    """
    assert feat_id in ch_info_dic, "feat_id %s not in ch_info_dic" %(feat_id)
    assert motif_matrix, "motif_matrix empty"
    scores = []
    feat_data = ch_info_dic[feat_id]
    feat_idxs = feat_data[1]
    assert len(feat_idxs) == 1, "len(feat_idxs) != 1 for feature %s" %(feat_id)
    feat_idx = feat_idxs[0]
    # Get scores list.
    for fv in motif_matrix:
        scores.append(fv[feat_idx])
    # Make dataframe.
    data = {}
    if stdev:
        assert len(scores) == len(stdev), "len(scores) != len(stdev) for feature ID %s" %(feat_id)
        data = {'pos': [], 'score': [], 'stdev': []}
    else:
        data = {'pos': [], 'score': []}
    for i,s in enumerate(scores):
        data['pos'].append(i+1)
        data['score'].append(s)
        if stdev:
            data['stdev'].append(stdev[i])
    if stdev:
        plot_df = pd.DataFrame(data, columns = ['pos', 'score', 'stdev'])
    else:
        plot_df = pd.DataFrame(data, columns = ['pos', 'score'])
    return plot_df


################################################################################

def add_motif_label_plot(df, fig, gs, i,
                         color_dict=False,
                         y_label_size=9,
                         y_label="exon-intron"):
    """
    Make exon-intron label plot.

    """
    ax = fig.add_subplot(gs[i, :])
    if color_dict:
        logo = logomaker.Logo(df, ax=ax, vpad=0.1, color_scheme=color_dict)
    else:
        # Nucleotides plot.
        logo = logomaker.Logo(df, ax=ax, vpad=0.1)
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=False, bounds=[0, 1])
    logo.style_spines(spines=['bottom'], visible=False)
    logo.ax.set_xticks([])
    logo.ax.set_yticks([])
    plt.yticks(fontsize=7)
    #logo.ax.set_axis_off()
    #logo.ax.set_yticklabels(['0', '1'])
    logo.ax.set_ylabel(y_label, labelpad=24, fontsize=y_label_size)


################################################################################

def add_phastcons_scores_plot(df, fig, gs, i,
                              stdev=False,
                              y_label_size=9,
                              disable_y_labels=False,
                              ylabel="phastCons score"):
    """
    Make phastCons conservation scores plot.
    phastCons values range from 0 .. 1.

    plt.bar options:
    color = 'red'
    width = .5
    align='edge' # tick alignment to bars

    Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    yerr=df['stdev']
    """
    ax = fig.add_subplot(gs[i, :])
    #ax = plt.gca()
    if stdev:
        df.plot(kind='bar', x='pos', y='score', yerr=df['stdev'], ecolor='grey', ax=ax, width = 1, legend=False)
    else:
        df.plot(kind='bar', x='pos', y='score', ax=ax, width = 1, legend=False)
    #ax.axhline(y=0, color='k', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.set_xlim(-0.5, max(df['pos']-0.5))
    if stdev:
        ax.errorbar(x=df['pos'], y=df['score'], yerr=df['stdev'], ecolor='grey', ls='none')
    ax.set_ylabel(ylabel, labelpad=12, fontsize=y_label_size)
    ax.set_xlabel('')
    # style using Axes methods
    #nn_logo.ax.set_xlim([20, 115])
    #ax.set_xticks([]) # no x-ticks.
    #nn_logo.ax.set_ylim([-.6, .75])
    if not disable_y_labels:
        ax.set_ylim([0, 1])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.yticks(fontsize=7)
    #nn_logo.ax.set_ylabel('score', labelpad=-1)
    #nn_logo.ax.set_ylabel('score')


################################################################################

def add_phylop_scores_plot(df, fig, gs, i,
                           stdev=False,
                           y_label_size=9,
                           ylabel="phyloP score"):
    """
    Make phyloP conservation scores plot.
    Normalized phyloP values range from -1 .. 1.

    plt.bar options:
    color = 'red'
    width = .5
    align='edge' # tick alignment to bars

    Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ecolor='#3376ae'

    """
    ax = fig.add_subplot(gs[i, :])
    #ax = plt.gca()

    if stdev:
        df.plot(kind='bar', x='pos', y='score', yerr=df['stdev'], ecolor='grey', ax=ax, width = 1, legend=False)
    else:
        df.plot(kind='bar', x='pos', y='score', ax=ax, width = 1, legend=False)
    #ax.axhline(y=0, color='k', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.set_xlim(-0.5, max(df['pos']-0.5))
    if stdev:
        ax.errorbar(x=df['pos'], y=df['score'], xerr=None, yerr=df['stdev'], ecolor='grey', ls='none')
    ax.set_ylabel(ylabel, labelpad=9, fontsize=y_label_size)
    ax.set_xlabel('')
    # style using Axes methods
    #nn_logo.ax.set_xlim([20, 115])
    #ax.set_xticks([]) # no x-ticks.
    #nn_logo.ax.set_ylim([-.6, .75])
    ax.set_ylim([-1, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['-1', '0', '1'])
    ax.set_xticks([])
    ax.set_xticklabels([])
    plt.yticks(fontsize=7)
    #nn_logo.ax.set_ylabel('score', labelpad=-1)
    #nn_logo.ax.set_ylabel('score')


################################################################################

def make_feature_attribution_plot(seq, profile_scores, feat_list,
                                  ch_info_dic, plot_out_file,
                                  seq_label_plot=False):
    """
    Make a feature attribution plot, showing for each sequence position
    the importance score, as well as additional features in subplots.
    logomaker (pip install logomaker) is used for plotting.

    Dependencies:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import logomaker

    """

    # Checks.
    assert seq, "given seq empty"
    assert profile_scores, "given profile_scores list empty"
    assert feat_list, "given feat_list list empty"
    assert ch_info_dic, "given ch_info_dic list empty"
    assert plot_out_file, "given plot_out_file empty"

    # Dataframe for importance scores.
    seq_alphabet = ch_info_dic["fa"][2]
    is_df = seq_to_plot_df(seq, seq_alphabet, scores=profile_scores)
    # Number of plots.
    n_subplots = 1
    height_ratios = [2]

    # Optional sequence label plot.
    if seq_label_plot:
        sl_df = seq_to_plot_df(seq, seq_alphabet)
        n_subplots += 1
        height_ratios.append(1)

    # Heights and number of additional plots.
    for fid in ch_info_dic:
        if fid == "fa":
            continue
        n_subplots += 1
        height_ratios.append(1)

    # Init plot.
    fig_width = 8
    fig_height = 0.8 * n_subplots
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows=n_subplots, ncols=1, height_ratios=height_ratios)

    # Plot subplots.
    i_plot = 0
    color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000'}
    add_importance_scores_plot(is_df, fig, gs, i_plot,
                               color_dict=color_dict,
                               y_label_size=5.5)

    # Plot optional sequence label plot.
    if seq_label_plot:
        i_plot += 1
        color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000'}
        add_label_plot(sl_df, fig, gs, i_plot, color_dict=color_dict, y_label="sequence",
                       y_label_size=4)

    # Plot additional plots.
    for fid, fdt in sorted(ch_info_dic.items()):
        if fid == "fa":
            continue
        feat_type = fdt[0]
        feat_idxs = fdt[1]
        feat_alphabet = fdt[2]
        feat_encoding = fdt[3]
        l_idxs = len(feat_idxs)
        # Plot index.
        i_plot += 1
        if feat_type == "C":
            # Categorical data.
            feat_str = ""
            for fv in feat_list:
                for i,fi in enumerate(feat_idxs):
                    if fv[fi] == 1:
                        feat_str += feat_alphabet[i]
                        break
            c_df = seq_to_plot_df(feat_str, feat_alphabet)
            color_dict = False
            add_label_plot(c_df, fig, gs, i_plot, color_dict=color_dict, y_label=fid,
                           y_label_size=4)
        elif feat_type == "N":
            # Numerical data.
            data = {}
            color_dict = False
            # Check.
            for c in feat_alphabet:
                data[c] = []
            for fv in feat_list:
                for i,fi in enumerate(feat_idxs):
                    data[feat_alphabet[i]].append(fv[fi])
            #plot_df = pd.DataFrame(data, columns = feat_alphabet)
            #plot_df.index.name = "pos"
            if fid == "pc.con":
                assert l_idxs == 1, "len(feat_idxs) != 1 for pc.con feature (instead: %i)" %(l_idxs)
                pc_con_df = scores_to_plot_df(data[feat_alphabet[0]])
                add_phastcons_scores_plot(pc_con_df, fig, gs, i_plot,
                                          y_label_size=4)
            elif fid == "pp.con":
                assert l_idxs == 1, "len(feat_idxs) != 1 for pp.con feature (instead: %i)" %(l_idxs)
                pp_con_df = scores_to_plot_df(data[feat_alphabet[0]])
                add_phylop_scores_plot(pp_con_df, fig, gs, i_plot,
                                       y_label_size=4)
            elif fid == "elem_p.str":
                assert l_idxs == 5, "len(feat_idxs) != 5 for elem_p.str feature (instead: %i)" %(l_idxs)
                elem_plot_df = pd.DataFrame(data, columns = feat_alphabet)
                elem_plot_df.index.name = "pos"
                add_label_plot(elem_plot_df, fig, gs, i_plot,
                               color_dict=color_dict,
                               y_label=fid,
                               y_label_size=4)
            else:
                # All other numerical values.
                assert l_idxs == 1, "len(feat_idxs) != 1 for additional numerical %s feature (instead: %i)" %(fid, l_idxs)
                add_n_df = scores_to_plot_df(data[feat_alphabet[0]])
                if feat_encoding == "-":
                    add_phastcons_scores_plot(add_n_df, fig, gs, i_plot,
                                              disable_y_labels=True,
                                              ylabel=fid,
                                              y_label_size=4)
                elif feat_encoding == "prob": # 0..1
                    add_phastcons_scores_plot(add_n_df, fig, gs, i_plot,
                                              ylabel=fid,
                                              y_label_size=4)
                elif feat_encoding == "minmax_norm": # 0..1
                    add_phastcons_scores_plot(add_n_df, fig, gs, i_plot,
                                              ylabel=fid,
                                              y_label_size=4)
                else:
                    assert False, "invalid feature normalization string given for additional numerical %s feature (got: %s)" %(fid, feat_encoding)

    # Store plot.
    fig.savefig(plot_out_file, dpi=150, transparent=False)
    plt.close(fig)


################################################################################

def make_feature_attribution_plot_old(seq, profile_scores, plot_out_file,
                                  seq_alphabet=["A","C","G","U"],
                                  eia_alphabet=["E", "I"],
                                  rra_alphabet=["N", "R"],
                                  tra_alphabet=["C", "F", "T"],
                                  seq_label_plot=False,
                                  exon_intron_labels=False,
                                  repeat_region_labels=False,
                                  transcript_region_labels=False,
                                  phastcons_scores=False,
                                  phylop_scores=False):
    """
    Make a feature attribution plot, showing for each sequence position
    the importance score, as well as additional features in subplots.
    logomaker (pip install logomaker) is used for plotting.

    Dependencies:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import logomaker

    seq:
        Binding site RNA sequence.
    profile_scores:
        List with importance scores (profile scores produced by model
        for each sequence position).
    plot_out_file:
        Plot output file. E.g. "results_out/site1_plot.png"
    seq_alphabet:
        Sequence alphabet, by default == RNA alphabet (ACGU).
    eia_alphabet:
        Exon-intron labels alphabet.
    rra_alphabet:
        Repeat region labels alphabet.
    tra_alphabet:
        Transcript region labels alphabet.
    seq_label_plot:
        Make RNA sequence label plot.
    exon_intron_labels:
        List of exon-intron labels for the binding site.
    repeat_region_labels:
        List of repeat region labels for the binding site.
    transcript_region_labels:
        List of transcript region labels for the binding site.
    phastcons_scores:
        List of phastCons scores for the binding site.
    phylop_scores:
        List of phyloP scores for the binding site.

    Original colors:
    'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000',

    Different greys:
    'A' : '#616161', 'C': '#7f7f7f',  'G': '#a0a0a0',  'U': '#d2d2d2'
    'a' : '#616161', 'c': '#7f7f7f',  'g': '#a0a0a0',  'u': '#d2d2d2'

    """

    assert seq, "given seq empty"
    assert profile_scores, "given profile_scores list empty"
    assert plot_out_file, "given plot_out_file empty"

    # Dataframe for importance scores.
    is_df = seq_to_plot_df(seq, seq_alphabet, scores=profile_scores)
    # Number of plots.
    n_subplots = 1
    height_ratios = [2]

    # Dataframes for additional subplots.
    if seq_label_plot:
        sl_df = seq_to_plot_df(seq, seq_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    if exon_intron_labels:
        assert len(exon_intron_labels) == len(seq), "length exon_intron_labels list != length seq"
        ei_seq = "".join(exon_intron_labels)
        ei_df = seq_to_plot_df(ei_seq, eia_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    if repeat_region_labels:
        assert len(repeat_region_labels) == len(seq), "length repeat_region_labels list != length seq"
        rra_seq = "".join(repeat_region_labels)
        rra_df = seq_to_plot_df(rra_seq, rra_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    if transcript_region_labels:
        assert len(transcript_region_labels) == len(seq), "length transcript_region_labels list != length seq"
        tra_seq = "".join(transcript_region_labels)
        tra_df = seq_to_plot_df(tra_seq, tra_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    if phastcons_scores:
        assert len(phastcons_scores) == len(seq), "length phastcons_scores list != length seq"
        pc_con_df = scores_to_plot_df(phastcons_scores)
        n_subplots += 1
        height_ratios.append(1)
    if phylop_scores:
        assert len(phylop_scores) == len(seq), "length phylop_scores list != length seq"
        pp_con_df = scores_to_plot_df(phylop_scores)
        n_subplots += 1
        height_ratios.append(1)

    # Init plot.
    fig_width = 8
    fig_height = 0.8 * n_subplots
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows=n_subplots, ncols=1, height_ratios=height_ratios)

    # Plot subplots.
    i = 0
    color_dict = False
    if seq_alphabet == ["A","C","G","U"]:
        color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000'}
    elif seq_alphabet == ["A","C","G","U","a","c","g","u"]:
        color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000', 'a' : '#008000', 'c': '#0000ff',  'g': '#ffa600',  'u': '#ff0000'}
    else:
        assert False, "invalid seq_alphabet given"
    add_importance_scores_plot(is_df, fig, gs, i,
                               color_dict=color_dict,
                               y_label_size=5.5)
    if seq_label_plot:
        i += 1
        color_dict = False
        if seq_alphabet == ["A","C","G","U"]:
            color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000'}
        elif seq_alphabet == ["A","C","G","U","a","c","g","u"]:
            color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000', 'a' : '#008000', 'c': '#0000ff',  'g': '#ffa600',  'u': '#ff0000'}
        else:
            assert False, "invalid seq_alphabet given"
        add_label_plot(sl_df, fig, gs, i, color_dict=color_dict, y_label="sequence",
                       y_label_size=4)
    if exon_intron_labels:
        i += 1
        color_dict = False
        if eia_alphabet == ["E", "I"]:
            color_dict = {'E' : 'grey', 'I': 'lightgrey'}
        add_label_plot(ei_df, fig, gs, i, color_dict=color_dict,
                       y_label_size=4)
    if repeat_region_labels:
        i += 1
        color_dict = False
        if rra_alphabet == ["N", "R"]:
            color_dict = {'N' : 'grey', 'R': 'lightgrey'}
        add_label_plot(rra_df, fig, gs, i, color_dict=color_dict,
                       y_label_size=4)
    if transcript_region_labels:
        i += 1
        color_dict = False
        if tra_alphabet == ["C", "F", "T"]:
            color_dict = {'C' : '#616161', 'F': '#7f7f7f',  'T': '#a0a0a0'}
        add_label_plot(tra_df, fig, gs, i, color_dict=color_dict,
                       y_label_size=4)
    if phastcons_scores:
        i += 1
        add_phastcons_scores_plot(pc_con_df, fig, gs, i,
                                  y_label_size=4)
    if phylop_scores:
        i += 1
        add_phylop_scores_plot(pp_con_df, fig, gs, i,
                               y_label_size=4)

    # Store plot.
    fig.savefig(plot_out_file, dpi=150, transparent=False)
    plt.close(fig)


################################################################################

def motif_seqs_to_plot_df(motif_seqs_ll, alphabet=['A', 'C', 'G', 'U']):
    """
    Given a list of sequence character lists (same lengths), make
    a position-wise character probability matrix in form of dataframe.
    Should work for RNA, eia, rra ... motifs.

    E.g.
    motif_seqs_ll = [["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "C"], ["C", "C", "A", "A", "G"], ["C", "C", "A", "A", "U"]]
    alphabet = ["A", "C", "G", "U"]
    Intermediate dic of lists.
    data = {'A': [0.5, 0.5, 1.0, 1.0, 0.25], 'C': [0.5, 0.5, 0, 0, 0.25], 'G': [0, 0, 0, 0, 0.25], 'U': [0, 0, 0, 0, 0.25]}
    Final dataframe:
         A    C    G  T
    pos
    0    0.5  0.5  0  0
    ...

    motif_seqs_ll:
        list of sequence character lists (same lengths).
    alphabet:
        List of sequence characters to consider for logo generation.

    """

    data = {}
    for c in alphabet:
        data[c] = []
    # Number of motifs.
    c_motifs = len(motif_seqs_ll)
    # Length of motif (length of lists).
    motif_len = len(motif_seqs_ll[0])
    # Check for same lengths.
    for l in motif_seqs_ll:
        assert len(l) == motif_len, "differing motif list lengths encountered (%i != %i)" %(len(l), motif_len)

    for i in range(motif_len):
        cc_dic = {}
        for c in alphabet:
            cc_dic[c] = 0
        for j in range(c_motifs):
            e = motif_seqs_ll[j][i]
            cc_dic[e] += 1
        # Get character probabilities at position i of motif.
        for c in cc_dic:
            cc = cc_dic[c]
            if cc:
                cc_dic[c] = cc / c_motifs
        for c in cc_dic:
            data[c].append(cc_dic[c])
    plot_df = pd.DataFrame(data, columns = alphabet)
    plot_df.index.name = "pos"
    return plot_df


################################################################################

def motif_scores_to_plot_df(motif_sc_ll):
    """
    Given a list of score lists (same length), calculate average score
    for each position, store in dataframe and return dataframe.

    motif_sc_ll = [[1,1,1,1],[2,2,2,2],[3,3,3,3]]
    Dictionary of lists intermediate:
    data = {'pos': [1,2,3,4], 'scores': [2.0, 2.0, 2.0, 2.0]}
    Then create dataframe with
    pd.DataFrame (data, columns = ['pos', 'scores'])

    """

    assert motif_sc_ll, "given scores list empty"
    # Get mean scores list.
    scores = list(np.mean(motif_sc_ll, axis=0))
    stdev = list(np.std(motif_sc_ll, axis=0))
    # Prepare for dataframe.
    data = {'pos': [], 'score': [], 'stdev': []}
    for i,s in enumerate(scores):
        data['pos'].append(i+1)
        data['score'].append(s)
        data['stdev'].append(stdev[i])
    # Stuff into dataframe.
    plot_df = pd.DataFrame(data, columns = ['pos', 'score', 'stdev'])
    return plot_df

################################################################################

def make_motif_plot(motif_matrix, ch_info_dic, motif_out_file,
                    fid2stdev_dic=False):
    """
    Plot motif using a 2D list with size motif_size*num_features, which
    stores the average values for all feature channels.

    fid2stdev_dic:
        Feature ID to list of standard deviations (list length == motif size).
        Store score stdev at each motif position. For one-channel numerical
        features (pc.con, pp.con, additional numerical features).

    """

    # First make sequence plot, then alphabetically rest of features.
    seq_df = make_motif_label_plot_df("fa", ch_info_dic, motif_matrix)

    # Number of plots.
    n_subplots = 1
    height_ratios = [2.5]

    # Heights and number of additional plots.
    for fid in ch_info_dic:
        if fid == "fa":
            continue
        n_subplots += 1
        height_ratios.append(1)

    # Init plot.
    fig_width = 4.5
    fig_height = 1.5 * n_subplots
    if n_subplots == 1:
        fig_height = 2.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows=n_subplots, ncols=1, height_ratios=height_ratios)

    # Plot subplots.
    i_plot = 0
    # Sequence motif plot.
    color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000'}
    add_motif_label_plot(seq_df, fig, gs, i_plot, color_dict=color_dict, y_label="sequence")

    # Plot additional plots.
    for fid, fdt in sorted(ch_info_dic.items()):
        if fid == "fa":
            continue
        feat_type = fdt[0]
        feat_idxs = fdt[1]
        feat_alphabet = fdt[2]
        feat_encoding = fdt[3]
        l_idxs = len(feat_idxs)
        color_dict = False
        stdev = False
        # Plot index.
        i_plot += 1
        # Categorical data.
        if feat_type == "C":
            c_df = make_motif_label_plot_df(fid, ch_info_dic, motif_matrix)
            add_motif_label_plot(c_df, fig, gs, i_plot,
                                 color_dict=color_dict,
                                 y_label_size=7,
                                 y_label=fid)
        elif feat_type == "N":
            if fid == "pc.con":
                if fid2stdev_dic:
                    assert fid in fid2stdev_dic, "fid2stdev_dic given but missing feature ID %s" %(fid)
                    stdev = fid2stdev_dic[fid]
                pc_df = make_motif_scores_plot_df(fid, ch_info_dic, motif_matrix,
                                                  stdev=stdev)
                add_phastcons_scores_plot(pc_df, fig, gs, i_plot,
                                          stdev=stdev,
                                          y_label_size=4)
            elif fid == "pp.con":
                if fid2stdev_dic:
                    assert fid in fid2stdev_dic, "fid2stdev_dic given but missing feature ID %s" %(fid)
                    stdev = fid2stdev_dic[fid]
                pp_df = make_motif_scores_plot_df(fid, ch_info_dic, motif_matrix,
                                                  stdev=stdev)
                add_phylop_scores_plot(pp_df, fig, gs, i_plot,
                                       stdev=stdev,
                                       y_label_size=4)
            elif fid == "elem_p.str":
                elem_df = make_motif_label_plot_df(fid, ch_info_dic, motif_matrix)
                add_motif_label_plot(elem_df, fig, gs, i_plot,
                                     color_dict=color_dict,
                                     y_label_size=7,
                                     y_label=fid)
            else:
                # Additional numerical features, treat like pc.con, pp.con.
                assert l_idxs == 1, "len(feat_idxs) != 1 for additional numerical %s feature (instead: %i)" %(fid, l_idxs)
                if fid2stdev_dic:
                    assert fid in fid2stdev_dic, "fid2stdev_dic given but missing feature ID %s" %(fid)
                    stdev = fid2stdev_dic[fid]
                add_n_df = make_motif_scores_plot_df(fid, ch_info_dic, motif_matrix,
                                                  stdev=stdev)
                if feat_encoding == "-":
                    add_phastcons_scores_plot(add_n_df, fig, gs, i_plot,
                                              stdev=stdev,
                                              disable_y_labels=True,
                                              ylabel=fid,
                                              y_label_size=4)
                elif feat_encoding == "prob": # 0..1
                    add_phastcons_scores_plot(add_n_df, fig, gs, i_plot,
                                              stdev=stdev,
                                              ylabel=fid,
                                              y_label_size=4)
                elif feat_encoding == "minmax_norm": # 0..1
                    add_phastcons_scores_plot(add_n_df, fig, gs, i_plot,
                                              stdev=stdev,
                                              ylabel=fid,
                                              y_label_size=4)
                else:
                    assert False, "invalid feature normalization string given for additional numerical %s feature (got: %s)" %(fid, feat_encoding)

    # Store plot.
    fig.savefig(motif_out_file, dpi=150, transparent=False)
    plt.close(fig)


################################################################################

def make_motif_plot_old(motif_seqs_ll, motif_out_file,
                    motif_eia_ll=False,
                    motif_rra_ll=False,
                    motif_tra_ll=False,
                    motif_pc_ll=False,
                    motif_pp_ll=False,
                    seq_alphabet=['A', 'C', 'G', 'U'],
                    eia_alphabet=['E', 'I'],
                    rra_alphabet=['N', 'R'],
                    tra_alphabet=['C', 'F', 'T']):
    """
    Plot motifs.
    Apart from sequence motif (motif_seqs_ll), optionally create motif for
    exon-intron feature (motif_eia_ll)
    phastCons scores (motif_pc_ll)
    phyloP scores (motif_pp_ll)

    motif_seqs_ll:
        List of sequence character lists (same lengths)
    motif_eia_ll:
        List of exon-intron character lists (same lengths)
    motif_rra_ll:
        List of repeat region character lists (same lengths)
    motif_tra_ll:
        List of transcript region character lists (same lengths)
    motif_pc_ll:
        List of phastCons score lists (same lengths)
    motif_pp_ll:
        List of phyloP score lists (same lengths)
    seq_alphabet:
        Sequence alphabet, by default == RNA alphabet (ACGU).
    eia_alphabet:
        Exon-intron labels alphabet.
    rra_alphabet:
        Repeat region labels alphabet.
    tra_alphabet:
        Transcript region labels alphabet.

    """
    # Get motif size from data.
    motif_size = len(motif_seqs_ll[0])

    # Sequence motif dataframe.
    seq_df = motif_seqs_to_plot_df(motif_seqs_ll, alphabet=seq_alphabet)
    # Number of plots.
    n_subplots = 1
    height_ratios = [2.5]
    # Exon-intron motif dataframe.
    if motif_eia_ll:
        eia_df = motif_seqs_to_plot_df(motif_eia_ll, alphabet=eia_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    # Repeat region motif dataframe.
    if motif_rra_ll:
        rra_df = motif_seqs_to_plot_df(motif_rra_ll, alphabet=rra_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    # Transcript region motif dataframe.
    if motif_tra_ll:
        tra_df = motif_seqs_to_plot_df(motif_tra_ll, alphabet=tra_alphabet)
        n_subplots += 1
        height_ratios.append(1)
    # phastCons scores dataframe.
    if motif_pc_ll:
        pc_df = motif_scores_to_plot_df(motif_pc_ll)
        n_subplots += 1
        height_ratios.append(1)
    # phyloP scores dataframe.
    if motif_pp_ll:
        pp_df = motif_scores_to_plot_df(motif_pp_ll)
        n_subplots += 1
        height_ratios.append(1)

    # Init plot.
    fig_width = 4.5
    fig_height = 1.5 * n_subplots
    if n_subplots == 1:
        fig_height = 2.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(nrows=n_subplots, ncols=1, height_ratios=height_ratios)

    # Plot subplots.
    i = 0
    # Sequence motif.
    color_dict = False
    if seq_alphabet == ["A","C","G","U"]:
        color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000'}
    elif seq_alphabet == ["A","C","G","U","a","c","g","u"]:
        color_dict = {'A' : '#008000', 'C': '#0000ff',  'G': '#ffa600',  'U': '#ff0000', 'a' : '#008000', 'c': '#0000ff',  'g': '#ffa600',  'u': '#ff0000'}
    else:
        assert False, "invalid seq_alphabet given"
    add_motif_label_plot(seq_df, fig, gs, i, color_dict=color_dict, y_label="sequence")
    # Exon-intron motif.
    if motif_eia_ll:
        i += 1
        color_dict = False
        if eia_alphabet == ['E', 'I']:
            color_dict = {'E' : 'grey', 'I': 'lightgrey'}
        add_motif_label_plot(eia_df, fig, gs, i,
                             y_label="exon-intron",
                             y_label_size=7,
                             color_dict=color_dict)
    # Repeat region motif.
    if motif_rra_ll:
        i += 1
        color_dict = False
        if rra_alphabet == ['N', 'R']:
            color_dict = {'N' : 'grey', 'R': 'lightgrey'}
        add_motif_label_plot(rra_df, fig, gs, i,
                             y_label="repeat region",
                             y_label_size=7,
                             color_dict=color_dict)
    # Transcript region motif.
    if motif_tra_ll:
        i += 1
        color_dict = False
        if tra_alphabet == ["C", "F", "T"]:
            color_dict = {'C' : '#616161', 'F': '#7f7f7f',  'T': '#a0a0a0'}
        add_motif_label_plot(tra_df, fig, gs, i,
                             y_label="transcript region",
                             y_label_size=7,
                             color_dict=color_dict)
    # phastCons motif.
    if motif_pc_ll:
        i += 1
        add_phastcons_scores_plot(pc_df, fig, gs, i,
                                  y_label_size=7,
                                  stdev=True)
    # phyloP motif.
    if motif_pp_ll:
        i += 1
        add_phylop_scores_plot(pp_df, fig, gs, i,
                               y_label_size=7,
                               stdev=True)

    # Store plot.
    fig.savefig(motif_out_file, dpi=150, transparent=False)


################################################################################

def process_custom_bp_file(bpp_file, bpp_out, seq_dic,
                           stats_dic=None):
    """
    Sanity check custom base pair information file, process and output to
    bpp_out.

    """

    bpp_dic = {}
    seq_id = ""
    seq_id_max_dic = {}
    seq_id_min_dic = {}
    # Go through base pairs file, extract sequences.
    with open(bpp_file) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                seq_id_max_dic[seq_id] = 0
                seq_id_min_dic[seq_id] = 100000
                bpp_dic[seq_id] = []
            else:
                cols = line.strip().split("\t")
                s = int(cols[0])
                e = int(cols[1])
                bpp = "1"
                if len(cols) == 3:
                    bpp = cols[2]
                assert s < e, "--bp-in base pair start coordinate >= end coordinate encountered (%i >= %i)" %(s, e)
                if s < seq_id_min_dic[seq_id]:
                    seq_id_min_dic[seq_id] = s
                if e > seq_id_max_dic[seq_id]:
                    seq_id_max_dic[seq_id] = e
                bpp_dic[seq_id].append([s, e, bpp])
    f.closed
    assert bpp_dic, "no base pairs read in from --bp-in"
    # Pairing rules.
    bp_nts_dic = {
        "A" : ["U"],
        "C" : ["G"],
        "G" : ["C","U"],
        "U" : ["A","G"]
    }

    # Statistics dictionary.
    if stats_dic is not None:
        stats_dic["bp_c"] = 0
        stats_dic["nobpsites_c"] = 0
        stats_dic["seqlen_sum"] = 0
        pbp_list = []

    # Check for lengths.
    for seq_id in seq_dic:
        assert seq_id in bpp_dic, "--in ID \"%s\" missing in --bp-in file" %(seq_id)
        seq = seq_dic[seq_id]
        seq_l = len(seq_dic[seq_id])
        min_s = seq_id_min_dic[seq_id]
        max_e = seq_id_max_dic[seq_id]
        assert min_s >= 1, "--bp-in ID \"%s\" minimum start index < 1"
        assert max_e <= seq_l, "--bp-in ID \"%s\" maximum end index > --in sequence length (%i > %i)" %(max_e, seq_l)
        # Check for valid base pairs.
        seq = seq.upper()
        seq_list = list(seq)
        for row in bpp_dic[seq_id]:
            s = row[0] - 1
            e = row[1] - 1
            nt1 = seq_list[s]
            nt2 = seq_list[e]
            nt2_list = bp_nts_dic[nt1]
            assert nt2 in nt2_list, "--bp-in contains incompatible base pair indices  (%s cannot pair with %s)" %(nt1, nt2)
        stats_dic["seqlen_sum"] += seq_l

    # Output bpp file.
    BPPOUT = open(bpp_out, "w")
    for seq_id in bpp_dic:
        bpp_list = bpp_dic[seq_id]
        if not bpp_list:
            if stats_dic is not None:
                stats_dic["nobpsites_c"] += 1
        BPPOUT.write(">%s\n" %(seq_id))
        for row in bpp_list:
            if stats_dic is not None:
                pbp_list.append(float(row[2]))
                stats_dic["bp_c"] += 1
            BPPOUT.write("%i\t%i\t%s\n" %(row[0], row[1], row[2]))
    BPPOUT.close()

    # Average base pair probability and stdev.
    if stats_dic is not None:
        stats_dic["bp_p"] = [statistics.mean(pbp_list)]
        stats_dic["bp_p"] += [statistics.stdev(pbp_list)]


################################################################################

def create_test_set_lengths_plot(test_len_list, out_plot,
                                 theme=1,
                                 scale_zero_max=False):
    """
    Create a box plot, showing the distribution of test set lengths.
    Given a list of test set lengths, create a dataframe
    using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    Midnight Blue theme.

    ffffff : white
    190250 : midnight blue
    fcc826 : yellowish
    fd3b9d : pinkish
    2f19f3 : dash blue

    """
    # Checker.
    assert test_len_list, "given list test_len_list empty"
    if scale_zero_max:
        # Get maximum length for scaling.
        max_l = max(test_len_list)
        # Get next highest number % 10.
        max_y = max_l
        while max_y % 10:
             max_y += 1
    # Make pandas dataframe.
    test_label = "Test set"
    data = {'set': [], 'length': []}
    test_c = len(test_len_list)
    data['set'] += test_c*[test_label]
    data['length'] += test_len_list
    df = pd.DataFrame (data, columns = ['set','length'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="length", data=df, palette=['cyan'],
                    width=0.7, linewidth = 1.5, boxprops=dict(alpha=.7))
        # Modify.
        ax.set_ylabel("Length (nt)",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        if scale_zero_max:
            ax.set_ylim([0,max_y])
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight')

    elif theme == 2:
        # Theme colors.
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"

        # Custom flier (outlier) edge and face colors.
        flierprops = dict(markersize=5, markerfacecolor=box_color, markeredgecolor=text_color)
        boxprops = dict(color=box_color, edgecolor=text_color)
        medianprops = dict(color=text_color)
        meanprops = dict(color=text_color)
        whiskerprops = dict(color=text_color)
        capprops = dict(color=text_color)

        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="length", data=df,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    meanprops=meanprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    width=0.7, linewidth = 1.5)

        # Modify.
        ax.set_ylabel("Length (nt)",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        if scale_zero_max:
            ax.set_ylim([0,max_y])
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight', transparent=True)


################################################################################

def create_test_set_entropy_plot(test_entr_list, out_plot,
                                 theme=1):
    """
    Create a box plot, showing the distribution of sequence entropies for
    the test dataset.
    Given a list entropies for the test dataset, create a dataframe
    using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    """
    # Checker.
    assert test_entr_list, "given list test_entr_list empty"
    # Make pandas dataframe.
    test_label = "Test set"
    data = {'set': [], 'entropy': []}
    test_c = len(test_entr_list)
    data['set'] += test_c*[test_label]
    data['entropy'] += test_entr_list
    df = pd.DataFrame (data, columns = ['set','entropy'])

    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="entropy", data=df, palette=['cyan'],
                    width=0.7, linewidth = 1.5, boxprops=dict(alpha=.7))
        # Modify.
        ax.set_ylabel("Sequence complexity",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight')

    elif theme == 2:
        """
        Midnight Blue theme:
        ====================

        HTML Hex colors:
        ffffff : white
        190250 : midnight blue
        fcc826 : yellowish
        fd3b9d : pinkish
        2f19f3 : dash blue

        bgcolor="#190250"
        text="#ffffff"
        link="#fd3b9d"
        vlink="#fd3b9d"
        alink="#fd3b9d"

        Editing matplotlib boxplot element props:
        (from matplotlib.axes.Axes.boxplot)
        boxprops
        whiskerprops
        flierprops
        medianprops
        meanprops

        """
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Custom flier (outlier) edge and face colors.
        flierprops = dict(markersize=5, markerfacecolor=box_color, markeredgecolor=text_color)
        boxprops = dict(color=box_color, edgecolor=text_color)
        medianprops = dict(color=text_color)
        meanprops = dict(color=text_color)
        whiskerprops = dict(color=text_color)
        capprops = dict(color=text_color)
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.boxplot(x="set", y="entropy", data=df,
                    flierprops=flierprops,
                    boxprops=boxprops,
                    meanprops=meanprops,
                    medianprops=medianprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    width=0.7, linewidth = 1.5)
        # Modify.
        ax.set_ylabel("Sequence complexity",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        ax.set(xlabel=None)
        # Store plot.
        fig.savefig(out_plot, dpi=125, bbox_inches='tight', transparent=True)



################################################################################

def create_test_set_dint_plot(test_dintp_dic, out_plot,
                              theme=1):
    """
    Create a grouped bar plot, showing the di-nucleotide percentages
    (16 classes) in the test dataset.
    Given a dictionary for test dataset, with key being
    di-nucleotide and value the percentage.
    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    MV colors:
    #69e9f6, #f154b2

    """

    # Checker.
    assert test_dintp_dic, "given dictionary test_dintp_dic empty"
    # Make pandas dataframe.
    test_label = "Test set"
    data = {'dint': [], 'perc': []}
    for dint in test_dintp_dic:
        data['dint'].append(dint)
        data['perc'].append(test_dintp_dic[dint])
    df = pd.DataFrame (data, columns = ['dint', 'perc'])
    y_label = "Percentage (%)"
    if theme == 1:
        theme_palette = []
        for dint in test_dintp_dic:
            theme_palette.append("#69e9f6")
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.barplot(x="dint", y="perc", data=df, ecolor="darkgrey",
                        palette=theme_palette,
                        edgecolor="lightgrey")
        fig.set_figwidth(11)
        fig.set_figheight(3.5)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        theme_palette = []
        for dint in test_dintp_dic:
            theme_palette.append("blue")
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.barplot(x="dint", y="perc", data=df, ecolor="#fcc826",
                        palette=theme_palette,
                        edgecolor="#fcc826")
        fig.set_figwidth(11)
        fig.set_figheight(3.5)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def generate_test_set_top_kmer_table(test_kmer_dic,
                                     top=5,
                                     val_type="c"):
    """
    Given k-mer count dictionaries for the test dataset, generate
    a markdown table with top x k-mers (sorted by decending dictionary value).

    val_type:
    Specify type of stored dictionary value.
    c : count (count of k-mer)
    r : ratio (k-mer count / total k-mer count)
    p : percentage ( (k-mer count / total k-mer count) * 100)

    """
    assert test_kmer_dic, "given dictionary test_kmer_dic empty"
    assert re.search("^[c|p|r]$", val_type), "invalid val_type given"
    # Get size of k.
    k = 0
    for kmer in test_kmer_dic:
        k = len(kmer)
        break
    # Expected kmer number.
    exp_kmer_nr = pow(4,k)
    test_kmer_nr = 0
    neg_kmer_nr = 0
    for kmer in test_kmer_dic:
        kc = test_kmer_dic[kmer]
        if kc:
            test_kmer_nr += 1
    test_kmer_perc = "%.2f " %((test_kmer_nr / exp_kmer_nr) * 100) + " %"
    # Adjust decimal places based on k-mer size.
    dc_p = 2
    dc_r = 4
    if k > 3:
        for i in range(k-3):
            dc_p += 1
            dc_r += 1
    dc_p_str = "%."+str(dc_p)+"f"
    dc_r_str = "%."+str(dc_r)+"f"
    add_ch = ""
    if val_type == "p":
        add_ch = " %"
        # Format test_kmer_dic to two decimal places.
        for kmer in test_kmer_dic:
            new_v = dc_p_str % test_kmer_dic[kmer]
            test_kmer_dic[kmer] = new_v
    elif val_type == "r":
        # Format percentage to four decimal places.
        for kmer in test_kmer_dic:
            new_v = dc_r_str % test_kmer_dic[kmer]
            test_kmer_dic[kmer] = new_v

    # Get top j k-mers.
    i = 0
    test_topk_list = []

    for kmer, v in sorted(test_kmer_dic.items(), key=lambda item: item[1], reverse=True):
        i += 1
        if i > top:
            break
        test_topk_list.append(kmer)

    # Generate markdown table.
    mdtable = "| Rank | &nbsp; &nbsp; Test set &nbsp; &nbsp; |\n"
    mdtable += "| :-: | :-: |\n"
    for i in range(top):
        test_kmer = test_topk_list[i]
        pos = i + 1
        mdtable += "| %i | %s (%s%s) |\n" %(pos, test_kmer, str(test_kmer_dic[test_kmer]), add_ch)

    mdtable += "| ... | &nbsp; |\n"
    mdtable += "| # distinct k-mers | %i (%s) |\n" %(test_kmer_nr, test_kmer_perc)

    # Return markdown table.
    return mdtable


################################################################################

def create_test_set_str_elem_plot(test_str_stats_dic, out_plot,
                                  theme=1):
    """
    Create a bar plot, showing average probabilities of secondary
    structure elements (U, E, H, I, M, S) in the test set.
    test_str_stats_dic contains statistics for test set (mean + stdev values).
    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    Stats dictionary content.
    stats_dic["U"] = [pu_mean, pu_stdev]
    stats_dic["S"] = [ps_mean, ps_stdev]
    stats_dic["E"] = [pe_mean, pe_stdev]
    stats_dic["H"] = [ph_mean, ph_stdev]
    stats_dic["I"] = [pi_mean, pi_stdev]
    stats_dic["M"] = [pm_mean, pm_stdev]

    """
    # Checker.
    assert test_str_stats_dic, "given dictionary test_str_stats_dic empty"
    # Make pandas dataframe.
    data = {'elem': [], 'mean_p': [], 'stdev_p': []}
    theme1_palette = []
    theme2_palette = []
    for el in test_str_stats_dic:
        if not re.search("^[U|S|E|H|I|M]$", el):
            continue
        data['elem'].append(el)
        data['mean_p'].append(test_str_stats_dic[el][0])
        data['stdev_p'].append(test_str_stats_dic[el][1])
        theme1_palette.append("#69e9f6")
        theme2_palette.append("blue")
    df = pd.DataFrame (data, columns = ['elem', 'mean_p', 'stdev_p'])
    y_label = "Mean probability"
    if theme == 1:
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.barplot(x="elem", y="mean_p", data=df, ecolor="darkgrey",
                        palette=theme1_palette, # yerr=df['stdev_p'],
                        edgecolor="lightgrey")
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight')

    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.barplot(x="elem", y="mean_p", data=df, ecolor="#fcc826",
                        palette=theme2_palette, # yerr=df['stdev_p'],
                        edgecolor="#fcc826")
        fig.set_figwidth(5)
        fig.set_figheight(4)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=14)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def create_test_set_region_annot_plot(test_ra_dic, out_plot,
                                      plot_labels,
                                      perc=False,
                                      theme=1):
    """
    Create a bar plot for region labels, given plot_labels to define what
    counts to plot. If perc=True, look for "total_pos" dictionary entry
    to normalize counts and plot percentages.

    Input dictionary has following keys:
    labels, total_pos

    Create a dataframe using Pandas, and use seaborn for plotting.
    Store plot in out_plot.

    MV colors:
    #69e9f6, #f154b2

    """
    # Checker.
    assert test_ra_dic, "given dictionary test_ra_dic empty"
    assert plot_labels, "given labels to plot list empty"
    if perc:
        assert test_ra_dic["total_pos"], "total_pos key missing in test_ra_dic"
    # Make pandas dataframe.
    data = {'label': [], 'count': []}
    for l in test_ra_dic:
        if l in plot_labels:
            lc = test_ra_dic[l]
            if perc:
                lc = (lc / test_ra_dic["total_pos"]) * 100
            data['label'].append(l)
            data['count'].append(lc)
    df = pd.DataFrame (data, columns = ['count', 'label'])
    y_label = "# positions"
    if perc:
        y_label = "Percentage (%)"
    if theme == 1:
        theme_palette = []
        for dint in test_ra_dic:
            theme_palette.append("#69e9f6")
        # Make plot.
        sns.set(style="darkgrid")
        fig, ax = plt.subplots()
        sns.barplot(x="label", y="count", data=df, ecolor="darkgrey",
                    palette=theme_palette,
                    edgecolor="lightgrey")
        fig.set_figwidth(8)
        fig.set_figheight(4)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight')
    elif theme == 2:
        text_color = "#fcc826"
        plot_color = "#fd3b9d"
        box_color = "#2f19f3"
        theme_palette = []
        for dint in test_ra_dic:
            theme_palette.append("blue")
        # Make plot.
        sns.set(style="darkgrid", rc={ "axes.labelcolor": text_color, "text.color": text_color, "xtick.color": text_color, "ytick.color": text_color, "grid.color": plot_color, "axes.edgecolor": plot_color})
        fig, ax = plt.subplots()
        sns.barplot(x="label", y="count", data=df, ecolor="#fcc826",
                    palette=theme_palette,
                    edgecolor="#fcc826")
        fig.set_figwidth(11)
        fig.set_figheight(3.5)
        ax.set(xlabel=None)
        ax.set_ylabel(y_label,fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=12)
        fig.savefig(out_plot, dpi=100, bbox_inches='tight', transparent=True)


################################################################################

def gp2_gp_generate_html_report(test_seqs_dic, out_folder,
                                dataset_type, gp2lib_path,
                                html_report_out=False,
                                id2ucr_dic=False,
                                plots_subfolder="plots_graphprot2_gp",
                                test_str_stats_dic=False,
                                test_phastcons_stats_dic=False,
                                test_phylop_stats_dic=False,
                                test_eia_stats_dic=False,
                                test_tra_stats_dic=False,
                                test_rra_stats_dic=False,
                                target_gbtc_dic=False,
                                all_gbtc_dic=False,
                                t2hc_dic=False,
                                t2i_dic=False,
                                theme=1,
                                kmer_top=10,
                                target_top=10,
                                rna=True,
                                ):
    """
    Generate HTML report for graphprot2 gp, providing statistics for the
    generated prediction dataset.

    test_seqs_dic:
        Sequences dictionary.
    test_str_stats_dic:
        Structure statistics dictionary
    test_phastcons_stats_dic:
        Phastcons scores statistics dictionary
    out_folder:
        graphprot2 gp results output folder, to store report in.
    id2ucr_dic:
        Sequence ID to uppercase sequence start + end, with format:
        sequence_id -> [uppercase_start,uppercase_end]
        where both positions are 1-based.
        If given, use only subsequences defined by this dictionary for
        generating sequence stats.
    rna:
        Set True if input sequences are RNA.
    html_report_out:
        HTML report output file.
    target_gbtc_dic:
        Gene biotype counts for target set dictionary.
    all_gbtc_dic:
        Gene biotype counts for all genes dictionary (gene biotype -> count)
    t2hc_dic:
        Transcript ID to hit count (# sites on transcript) dictionary.
    t2i_dic:
        Transcript ID to info list dictionary.

    """
    # Checks.
    ds_types = {'s':1, 't':1, 'g':1}
    assert dataset_type in ds_types, "invalid dataset type given (expected g, s, or t)"
    # Import markdown to generate report.
    from markdown import markdown

    # Output subfolder for plots.
    plots_folder = plots_subfolder
    plots_out_folder = out_folder + "/" + plots_folder
    if not os.path.exists(plots_out_folder):
        os.makedirs(plots_out_folder)
    # Output files.
    html_out = out_folder + "/" + "report.graphprot2_gp.html"
    if html_report_out:
        html_out = html_report_out
    # Plot files.
    lengths_plot = "set_lengths_plot.png"
    entropy_plot = "sequence_complexity_plot.png"
    dint_plot = "dint_percentages_plot.png"
    str_elem_plot = "str_elem_plot.png"
    phastcons_plot = "phastcons_plot.png"
    phylop_plot = "phylop_plot.png"
    eia_plot = "exon_intron_region_plot.png"
    tra_plot = "transcript_region_plot.png"
    rra_plot = "repeat_region_plot.png"
    lengths_plot_out = plots_out_folder + "/" + lengths_plot
    entropy_plot_out = plots_out_folder + "/" + entropy_plot
    dint_plot_out = plots_out_folder + "/" + dint_plot
    str_elem_plot_out = plots_out_folder + "/" + str_elem_plot
    phastcons_plot_out = plots_out_folder + "/" + phastcons_plot
    phylop_plot_out = plots_out_folder + "/" + phylop_plot
    eia_plot_out = plots_out_folder + "/" + eia_plot
    tra_plot_out = plots_out_folder + "/" + tra_plot
    rra_plot_out = plots_out_folder + "/" + rra_plot

    """
    If only uppercase part of sequences should be used for stats,
    prune sequence dictionary based on uppercase region start
    and end info stored in id2ucr_dic.
    """
    if id2ucr_dic:
        for seq_id in test_seqs_dic:
            seq = test_seqs_dic[seq_id]
            uc_s = id2ucr_dic[seq_id][0]
            uc_e = id2ucr_dic[seq_id][1]
            test_seqs_dic[seq_id] = seq[uc_s-1:uc_e]

    print("Generate statistics for HTML report ... ")

    # Site numbers.
    c_test_out = len(test_seqs_dic)
    # Site lengths.
    test_len_list = get_seq_len_list_from_dic(test_seqs_dic)
    # Get entropy scores for sequences.
    test_entr_list = seqs_dic_calc_entropies(test_seqs_dic, rna=rna,
                                             uc_part_only=True)

    # Get set nucleotide frequencies.
    test_ntc_dic = seqs_dic_count_nt_freqs(test_seqs_dic, rna=rna,
                                           convert_to_uc=True)
    # Get nucleotide ratios.
    test_ntr_dic = ntc_dic_to_ratio_dic(test_ntc_dic, perc=True)

    # Get dinucleotide percentages.
    test_dintr_dic = seqs_dic_count_kmer_freqs(test_seqs_dic, 2, rna=rna,
                                              return_ratios=True,
                                              perc=True,
                                              report_key_error=True,
                                              convert_to_uc=True)
    # Get 3-mer percentages.
    test_3mer_dic = seqs_dic_count_kmer_freqs(test_seqs_dic, 3, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    # Get 4-mer percentages.
    test_4mer_dic = seqs_dic_count_kmer_freqs(test_seqs_dic, 4, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)
    # Get 5-mer percentages.
    test_5mer_dic = seqs_dic_count_kmer_freqs(test_seqs_dic, 5, rna=rna,
                                             return_ratios=True,
                                             perc=True,
                                             report_key_error=True,
                                             convert_to_uc=True)

    # Logo paths.
    logo1_path = gp2lib_path + "/content/logo1.png"
    logo2_path = gp2lib_path + "/content/logo2.png"
    logo3_path = gp2lib_path + "/content/logo3.png"

    # Create theme-specific HTML header.
    if theme == 1:
        mdtext = """
<head>
<title>GraphProt2 - Prediction Set Generation Report</title>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="600" />

""" %(logo1_path)
    elif theme == 2:
        mdtext = """
<head>
<title>GraphProt2 - Prediction Set Generation Report</title>
<style>
h1 {color:#fd3b9d;}
h2 {color:#fd3b9d;}
h3 {color:#fd3b9d;}
</style>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="500" />

<body style="font-family:sans-serif" bgcolor="#190250" text="#fcc826" link="#fd3b9d" vlink="#fd3b9d" alink="#fd3b9d">

""" %(logo2_path)
    elif theme == 3:
        mdtext = """
<head>
<title>GraphProt2 - Prediction Set Generation Report</title>
<style>
h1 {color:#1fcc2c;}
h2 {color:#1fcc2c;}
h3 {color:#1fcc2c;}
</style>
</head>

<img src="%s" alt="gp2_logo"
	title="gp2_logo" width="400" />

<body style="font-family:monospace" bgcolor="#1d271e" text="#1fcc2c" link="#1fcc2c" vlink="#1fcc2c" alink="#1fcc2c">

""" %(logo3_path)
    else:
        assert False, "invalid theme ID given"

    # Add first section markdown.
    mdtext += """

# Prediction set generation report

List of available statistics for the prediction dataset generated
by GraphProt2 (graphprot2 gp):

- [Prediction dataset statistics](#set-stats)
- [Site length distribution](#len-plot)
- [Sequence complexity distribution](#ent-plot)
- [Di-nucleotide distribution](#dint-plot)
- [Top k-mer statistics](#kmer-stats)"""

    if test_str_stats_dic:
        mdtext += "\n"
        if 'S' in test_str_stats_dic:
            mdtext += "- [Structural elements distribution](#str-elem-plot)\n"
            mdtext += "- [Secondary structure statistics](#bp-stats)"
        else:
            # If only --bp-in selected.
            mdtext += "- [Secondary structure statistics](#bp-stats)"
    if test_phastcons_stats_dic or test_phylop_stats_dic:
        mdtext += "\n"
        mdtext += "- [Conservation scores distribution](#con-plot)\n"
        mdtext += "- [Conservation scores statistics](#con-stats)"
    if test_eia_stats_dic:
        occ_labels = ["F", "T"]
        mdtext += "\n"
        mdtext += "- [Exon-intron region distribution](#eia-plot)\n"
        mdtext += "- [Exon-intron region statistics](#eia-stats)"
    if test_tra_stats_dic:
        occ_labels = ["S", "E", "A", "Z", "B"]
        mdtext += "\n"
        mdtext += "- [Transcript region distribution](#tra-plot)\n"
        mdtext += "- [Transcript region statistics](#tra-stats)"
    if test_rra_stats_dic:
        mdtext += "\n"
        mdtext += "- [Repeat region distribution](#rra-plot)\n"
        mdtext += "- [Repeat region statistics](#rra-stats)"
    if target_gbtc_dic and all_gbtc_dic:
        mdtext += "\n"
        mdtext += "- [Target gene biotype statistics](#gbt-stats)"
    if t2hc_dic and t2i_dic:
        mdtext += "\n"
        mdtext += "- [Target region overlap statistics](#tro-stats)"
    mdtext += "\n&nbsp;\n"

    # Make general stats table.
    mdtext += """
## Prediction dataset statistics ### {#set-stats}

**Table:** Prediction dataset statistics regarding sequence lengths
(min, max, mean, and median length) in nucleotides (nt),
sequence complexity (mean Shannon entropy over all sequences in the set)
and nucleotide contents (A, C, G, U).

"""
    mdtext += "| Attribute | &nbsp; Prediction set &nbsp; | \n"
    mdtext += "| :-: | :-: |\n"
    mdtext += "| # sites | %i |\n" %(c_test_out)
    mdtext += "| min site length | %i |\n" %(min(test_len_list))
    mdtext += "| max site length | %i |\n" %(max(test_len_list))
    mdtext += "| mean site length | %.1f |\n" %(statistics.mean(test_len_list))
    mdtext += "| median site length | %i |\n" %(statistics.median(test_len_list))
    mdtext += "| mean complexity | %.3f |\n" %(statistics.mean(test_entr_list))
    mdtext += '| %A |' + " %.2f |\n" %(test_ntr_dic["A"])
    mdtext += '| %C |' + " %.2f |\n" %(test_ntr_dic["C"])
    mdtext += '| %G |' + " %.2f |\n" %(test_ntr_dic["G"])
    mdtext += '| %U |' + " %.2f |\n" %(test_ntr_dic["U"])
    mdtext += "\n&nbsp;\n&nbsp;\n"

    # Make site length distribution box plot.
    create_test_set_lengths_plot(test_len_list, lengths_plot_out,
                                 theme=theme)

    lengths_plot_path = plots_folder + "/" + lengths_plot

    mdtext += """
## Site length distribution ### {#len-plot}

Site length distribution in the prediction set. Lengths differences are due
to --in sequences or sites of various lengths.

"""
    mdtext += '<img src="' + lengths_plot_path + '" alt="Site length distribution"' + "\n"
    mdtext += 'title="Site length distribution" width="500" />' + "\n"
    mdtext += """

**Figure:** Site length distribution for the prediction dataset.

&nbsp;

"""
    # Make sequence complexity box plot.
    create_test_set_entropy_plot(test_entr_list, entropy_plot_out,
                                 theme=theme)
    entropy_plot_path = plots_folder + "/" + entropy_plot

    mdtext += """
## Sequence complexity distribution ### {#ent-plot}

The Shannon entropy is calculated for each sequence to measure
its information content (i.e., its complexity). A sequence with
equal amounts of all four nucleotides has an entropy value of 1.0
(highest possible). A sequence with equal amounts of two nucleotides
has an entropy value of 0.5. Finally, the lowest possible entropy is
achieved by a sequence which contains only one type of nucleotide.
Find the formula used to compute Shannon's entropy
[here](https://www.ncbi.nlm.nih.gov/pubmed/15215465) (see CE formula).


"""
    mdtext += '<img src="' + entropy_plot_path + '" alt="Sequence complexity distribution"' + "\n"
    mdtext += 'title="Sequence complexity distribution" width="500" />' + "\n"
    mdtext += """

**Figure:** Sequence complexity (Shannon entropy
computed for each sequence) distributions for the prediction dataset.

&nbsp;

"""

    # Make di-nucleotide bar plot.
    create_test_set_dint_plot(test_dintr_dic, dint_plot_out,
                              theme=theme)
    dint_plot_path = plots_folder + "/" + dint_plot

    mdtext += """
## Di-nucleotide distribution ### {#dint-plot}

Di-nucleotide percentages for the prediction dataset.

"""
    mdtext += '<img src="' + dint_plot_path + '" alt="Di-nucleotide distribution"' + "\n"
    mdtext += 'title="Di-nucleotide distribution" width="600" />' + "\n"
    mdtext += """

**Figure:** Di-nucleotide percentages for the prediction dataset.

&nbsp;

"""
    # Make the k-mer tables.
    top3mertab = generate_test_set_top_kmer_table(test_3mer_dic,
                                                  top=kmer_top,
                                                  val_type="p")
    top4mertab = generate_test_set_top_kmer_table(test_4mer_dic,
                                                  top=kmer_top,
                                                  val_type="p")
    top5mertab = generate_test_set_top_kmer_table(test_5mer_dic,
                                                  top=kmer_top,
                                                  val_type="p")
    mdtext += """
## Top k-mer statistics ### {#kmer-stats}

**Table:** Top %i 3-mers for the prediction dataset and their percentages. In case of uniform distribution with all 3-mers present, each 3-mer would have a percentage = 1.5625.

""" %(kmer_top)
    mdtext += top3mertab
    mdtext += "\n&nbsp;\n"

    mdtext += """
**Table:** Top %i 4-mers for the prediction dataset and their percentages. In case of uniform distribution with all 4-mers present, each 4-mer would have a percentage = 0.390625.

""" %(kmer_top)
    mdtext += top4mertab
    mdtext += "\n&nbsp;\n"

    mdtext += """
**Table:** Top %i 5-mers for the prediction dataset and their percentages. In case of uniform distribution with all 5-mers present, each 5-mer would have a percentage = 0.09765625.

""" %(kmer_top)
    mdtext += top5mertab
    mdtext += "\n&nbsp;\n&nbsp;\n"

    if test_str_stats_dic:
        # Checks.
        assert test_str_stats_dic['seqlen_sum'], "unexpected total sequence length of 0 encountered"
        if 'S' in test_str_stats_dic:
            # Make structural elements bar plot.
            create_test_set_str_elem_plot(test_str_stats_dic,
                                          str_elem_plot_out,
                                          theme=theme)
            str_elem_plot_path = plots_folder + "/" + str_elem_plot

            mdtext += """
## Structural elements distribution ### {#str-elem-plot}

Mean position-wise probabilities of the different loop context structural elements are shown
for the prediction dataset. U: unpaired, E: external loop, H: hairpin loop,
I: internal loop, M: multi-loop, S: paired.

"""
            mdtext += '<img src="' + str_elem_plot_path + '" alt="Structural elements distribution"' + "\n"
            mdtext += 'title="Structural elements distribution" width="400" />' + "\n"
            mdtext += """

**Figure:** Mean position-wise probabilities of different loop context structural
elements for the prediction dataset.
U: unpaired, E: external loop, H: hairpin loop,
I: internal loop, M: multi-loop, S: paired.

&nbsp;

"""
        # Make base pair stats table.
        test_bps_per_100nt = test_str_stats_dic['bp_c'] / (test_str_stats_dic['seqlen_sum'] / 100)
        test_mean_bp_p = test_str_stats_dic['bp_p'][0]
        test_mean_bp_stdev = test_str_stats_dic['bp_p'][1]

        mdtext += """
## Secondary structure statistics ### {#bp-stats}

**Table:** Secondary structure statistics of the generated prediction set.

"""

        mdtext += "| &nbsp; &nbsp; &nbsp; Attribute &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; Prediction set &nbsp; &nbsp; &nbsp; &nbsp; | \n"
        mdtext += "| :-: | :-: |\n"
        mdtext += "| total sequence length | %i |\n" %(test_str_stats_dic['seqlen_sum'])
        mdtext += "| # base pairs | %i |\n" %(test_str_stats_dic['bp_c'])
        mdtext += "| base pairs per 100 nt | %.1f |\n" %(test_bps_per_100nt)
        mdtext += "| # no-base-pair sites | %i |\n" %(test_str_stats_dic['nobpsites_c'])
        mdtext += "| mean p(base pair) | %.4f (+-%.4f) |\n" %(test_mean_bp_p, test_mean_bp_stdev)
        if 'S' in test_str_stats_dic:
            mdtext += "| mean p(paired) | %.4f (+-%.4f) |\n" %(test_str_stats_dic['S'][0], test_str_stats_dic['S'][1])
            mdtext += "| mean p(unpaired) | %.4f (+-%.4f) |\n" %(test_str_stats_dic['U'][0], test_str_stats_dic['U'][1])
            mdtext += "| mean p(external loop) | %.4f (+-%.4f) |\n" %(test_str_stats_dic['E'][0], test_str_stats_dic['E'][1])
            mdtext += "| mean p(hairpin loop) | %.4f (+-%.4f) |\n" %(test_str_stats_dic['H'][0], test_str_stats_dic['H'][1])
            mdtext += "| mean p(internal loop) | %.4f (+-%.4f) |\n" %(test_str_stats_dic['I'][0], test_str_stats_dic['I'][1])
            mdtext += "| mean p(multi loop) | %.4f (+-%.4f) |\n" %(test_str_stats_dic['M'][0], test_str_stats_dic['M'][1])
        mdtext += "\n&nbsp;\n&nbsp;\n"


    # Conservation scores plots and stats.
    if test_phastcons_stats_dic or test_phylop_stats_dic:

        mdtext += """
## Conservation scores statistics ### {#con-stats}

**Table:** Conservation scores statistics. Note that phyloP statistics are
calculated before normalization (normalizing values to -1 .. 1).

"""
        mdtext += "| &nbsp; &nbsp; Attribute &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; Prediction set &nbsp; &nbsp; &nbsp; | \n"
        mdtext += "| :-: | :-: |\n"
        if test_phastcons_stats_dic:
            pc_zero_perc = "%.2f" % ((test_phastcons_stats_dic["zero_pos"] / test_phastcons_stats_dic["total_pos"]) * 100)
            mdtext += "| # phastCons scores | %i |\n" %(test_phastcons_stats_dic['total_pos'])
            mdtext += "| # zero scores | %i |\n" %(test_phastcons_stats_dic['zero_pos'])
            mdtext += '| % zero scores |' + " %s |\n" %(pc_zero_perc)
            mdtext += "| min score | %s |\n" %(str(test_phastcons_stats_dic['min']))
            mdtext += "| max score | %s |\n" %(str(test_phastcons_stats_dic['max']))
            mdtext += "| mean score | %.3f (+-%.3f) |\n" %(test_phastcons_stats_dic['mean'], test_phastcons_stats_dic['stdev'])

        if test_phylop_stats_dic:
            pp_zero_perc = "%.2f" % ((test_phylop_stats_dic["zero_pos"] / test_phylop_stats_dic["total_pos"]) * 100)
            mdtext += "| # phyloP scores | %i |\n" %(test_phylop_stats_dic['total_pos'])
            mdtext += "| # zero scores | %i |\n" %(test_phylop_stats_dic['zero_pos'])
            mdtext += '| % zero scores |' + " %s |\n" %(pp_zero_perc)
            mdtext += "| min score | %s |\n" %(str(test_phylop_stats_dic['min']))
            mdtext += "| max score | %s |\n" %(str(test_phylop_stats_dic['max']))
            mdtext += "| mean score | %.3f (+-%.3f) |\n" %(test_phylop_stats_dic['mean'], test_phylop_stats_dic['stdev'])
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Exon-intron region plots and stats.
    if test_eia_stats_dic:
        mdtext += """
## Exon-intron region distribution ### {#eia-plot}

Distribution of exon and intron regions for the prediction set.

"""
        # EIA plot.
        create_test_set_region_annot_plot(test_eia_stats_dic, eia_plot_out,
                                          ["E", "I", "N"],
                                          perc=True, theme=theme)

        eia_plot_path = plots_folder + "/" + eia_plot
        mdtext += '<img src="' + eia_plot_path + '" alt="Exon-intron region distribution"' + "\n"
        mdtext += 'title="Exon-intron region distribution" width="550" />' + "\n"
        mdtext += """
**Figure:** Percentages of exon (E) and intron (I) regions for the prediction set.
If --eia-n is set, also include regions not covered by introns or exons (N).

&nbsp;

## Exon-intron region statistics ### {#eia-stats}

**Table:** Exon-intron region statistics for the prediction set.
If --eia-ib is set, also include statistics for sites containing intron
5' (F) and intron 3' (T) ends.

"""
        # EIA stats.
        if "F" in test_eia_stats_dic:
            test_perc_f_sites = "%.2f" % ((test_eia_stats_dic['F'] / c_test_out)*100) + " %"
            test_perc_t_sites = "%.2f" % ((test_eia_stats_dic['T'] / c_test_out)*100) + " %"
        test_perc_e = "%.2f" % ((test_eia_stats_dic['E'] / test_eia_stats_dic['total_pos'])*100)
        test_perc_i = "%.2f" % ((test_eia_stats_dic['I'] / test_eia_stats_dic['total_pos'])*100)
        if "N" in test_eia_stats_dic:
            test_perc_n = "%.2f" % ((test_eia_stats_dic['N'] / test_eia_stats_dic['total_pos'])*100)
        mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Prediction set &nbsp; | \n"
        mdtext += "| :-: | :-: |\n"
        mdtext += '| % E |' + " %s |\n" %(test_perc_e)
        mdtext += '| % I |' + " %s |\n" %(test_perc_i)
        if "N" in test_eia_stats_dic:
            mdtext += '| % N |' + " %s |\n" %(test_perc_n)
        if "F" in test_eia_stats_dic:
            mdtext += "| F sites | %i (%s) |\n" %(test_eia_stats_dic['F'], test_perc_f_sites)
            mdtext += "| T sites | %i (%s) |\n" %(test_eia_stats_dic['T'], test_perc_t_sites)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Transcript region plots and stats.
    if test_tra_stats_dic:
        mdtext += """
## Transcript region distribution ### {#tra-plot}

Distribution of transcript regions for the prediction set.

"""
        # TRA plot.
        create_test_set_region_annot_plot(test_tra_stats_dic, tra_plot_out,
                                          ["F", "C", "T", "N"],
                                          perc=True, theme=theme)
        tra_plot_path = plots_folder + "/" + tra_plot
        mdtext += '<img src="' + tra_plot_path + '" alt="Transcript region distribution"' + "\n"
        mdtext += 'title="Transcript region distribution" width="400" />' + "\n"
        mdtext += """
**Figure:** Percentages of 5'UTR (F), CDS (C), and 3'UTR (T) positions as well as
positions not covered by these transcript regions (N) for the prediction set.

&nbsp;

## Transcript region statistics ### {#tra-stats}

**Table:** Transcript region statistics for the prediction set.
Percentages of positions covered by 5'UTR (F), CDS (C), 3'UTR (T), or non
of these regions (N) are given.
If --tra-codons is set, also include statistics for start codons (S) and
stop codons (E) (sites which contain these).
If --tra-borders is set, also include statistics for transcript starts (A),
 transcript ends (Z), exon borders (B) (sites which contain these).

"""
        # TRA stats.
        test_perc_f = "%.2f" % ((test_tra_stats_dic['F'] / test_tra_stats_dic['total_pos'])*100)
        test_perc_c = "%.2f" % ((test_tra_stats_dic['C'] / test_tra_stats_dic['total_pos'])*100)
        test_perc_t = "%.2f" % ((test_tra_stats_dic['T'] / test_tra_stats_dic['total_pos'])*100)
        test_perc_n = "%.2f" % ((test_tra_stats_dic['N'] / test_tra_stats_dic['total_pos'])*100)


        mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Prediction set &nbsp; | \n"
        mdtext += "| :-: | :-: |\n"
        mdtext += '| % F |' + " %s |\n" %(test_perc_f)
        mdtext += '| % C |' + " %s |\n" %(test_perc_c)
        mdtext += '| % T |' + " %s |\n" %(test_perc_t)
        mdtext += '| % N |' + " %s |\n" %(test_perc_n)
        # Start stop codon annotations.
        if "S" in test_tra_stats_dic:
            test_perc_s_sites = "%.2f" % ((test_tra_stats_dic['S'] / c_test_out)*100) + " %"
            test_perc_e_sites = "%.2f" % ((test_tra_stats_dic['E'] / c_test_out)*100) + " %"
            mdtext += "| S sites | %i (%s) |\n" %(test_tra_stats_dic['S'], test_perc_s_sites)
            mdtext += "| E sites | %i (%s) |\n" %(test_tra_stats_dic['E'], test_perc_e_sites)
        # Border annotations.
        if "A" in test_tra_stats_dic:
            test_perc_a_sites = "%.2f" % ((test_tra_stats_dic['A'] / c_test_out)*100) + " %"
            test_perc_b_sites = "%.2f" % ((test_tra_stats_dic['B'] / c_test_out)*100) + " %"
            test_perc_z_sites = "%.2f" % ((test_tra_stats_dic['Z'] / c_test_out)*100) + " %"
            mdtext += "| A sites | %i (%s) |\n" %(test_tra_stats_dic['A'], test_perc_a_sites)
            mdtext += "| B sites | %i (%s) |\n" %(test_tra_stats_dic['B'], test_perc_b_sites)
            mdtext += "| Z sites | %i (%s) |\n" %(test_tra_stats_dic['Z'], test_perc_z_sites)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Repeat region plots and stats.
    if test_rra_stats_dic:
        mdtext += """
## Repeat region distribution ### {#rra-plot}

Distribution of repeat regions for the prediction set. Repeat
regions are annotated in the .2bit genomic sequences file as lowercase
sequences. These regions were identified by RepeatMasker and Tandem Repeats
Finder (with period of 12 or less).

"""
        # RRA plot.
        create_test_set_region_annot_plot(test_rra_stats_dic, rra_plot_out,
                                          ["R", "N"],
                                          perc=True, theme=theme)
        rra_plot_path = plots_folder + "/" + rra_plot
        mdtext += '<img src="' + rra_plot_path + '" alt="Repeat region distribution"' + "\n"
        mdtext += 'title="Repeat region distribution" width="400" />' + "\n"
        mdtext += """
**Figure:** Percentages of repeat (R) and no-repeat (N) regions for the
prediction set.

&nbsp;

## Repeat region statistics ### {#rra-stats}

**Table:** Repeat region statistics for the prediction set.
Percentages of prediction set regions covered by repeat (R)
 and non-repeat (N) regions are given.

"""
        # RRA stats.
        test_perc_r = "%.2f" % ((test_rra_stats_dic['R'] / test_rra_stats_dic['total_pos'])*100)
        test_perc_n = "%.2f" % ((test_rra_stats_dic['N'] / test_rra_stats_dic['total_pos'])*100)

        mdtext += "| &nbsp; Attribute &nbsp; | &nbsp; Prediction set &nbsp; |\n"
        mdtext += "| :-: | :-: |\n"
        mdtext += '| % R |' + " %s |\n" %(test_perc_r)
        mdtext += '| % N |' + " %s |\n" %(test_perc_n)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    # Target gene biotype count stats.
    if target_gbtc_dic and all_gbtc_dic:
        mdtext += """
## Target gene biotype statistics ### {#gbt-stats}

**Table:** Target gene biotype counts for the prediction set and their percentages
(count normalized by total count for the respective gene biotype).

"""
        mdtext += "| &nbsp; Gene biotype &nbsp; | &nbsp; Target count &nbsp; | &nbsp; Total count &nbsp; | &nbsp; Percentage &nbsp; | \n"
        mdtext += "| :-: | :-: | :-: | :-: |\n"
        unit = " %"
        for bt, target_c in sorted(target_gbtc_dic.items(), key=lambda item: item[1], reverse=True):
            all_c = all_gbtc_dic[bt]
            perc_c = "%.2f" % ((target_c / all_c) * 100)
            mdtext += "| %s | %i | %i | %s%s |\n" %(bt, target_c, all_c, perc_c, unit)
        mdtext += "\n&nbsp;\n&nbsp;\n"

    if t2hc_dic and t2i_dic:
        mdtext += """
## Target region overlap statistics ### {#tro-stats}

**Table:** Target region overlap statistics, showing the top %i targeted
regions (transcript or genes), with the # overlaps == # of sites
overlapping with the region.

""" %(target_top)

        if dataset_type == "t":
            mdtext += "| &nbsp; # overlaps &nbsp; | &nbsp; Transcript ID &nbsp; | &nbsp; &nbsp; Transcript biotype &nbsp; &nbsp; | &nbsp; Gene ID &nbsp; | &nbsp; Gene name &nbsp; | &nbsp; &nbsp; Gene biotype &nbsp; &nbsp; | \n"
            mdtext += "| :-: | :-: | :-: | :-: | :-: | :-: |\n"
            i = 0
            for tr_id, ol_c in sorted(t2hc_dic.items(), key=lambda item: item[1], reverse=True):
                i += 1
                if i > target_top:
                    break
                tr_bt = t2i_dic[tr_id][0]
                gene_id = t2i_dic[tr_id][1]
                gene_name = t2i_dic[tr_id][2]
                gene_bt = t2i_dic[tr_id][3]
                mdtext += "| %i | %s | %s |  %s | %s | %s |\n" %(ol_c, tr_id, tr_bt, gene_id, gene_name, gene_bt)
            mdtext += "| ... | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |\n"
            mdtext += "\n&nbsp;\n&nbsp;\n"

        elif dataset_type == "g":
            mdtext += "| &nbsp; # overlaps &nbsp; | &nbsp; Gene ID &nbsp; | &nbsp; Gene name &nbsp; | &nbsp; &nbsp; Gene biotype &nbsp; &nbsp; | \n"
            mdtext += "| :-: | :-: | :-: | :-: |\n"
            i = 0
            for gene_id, ol_c in sorted(t2hc_dic.items(), key=lambda item: item[1], reverse=True):
                i += 1
                if i > target_top:
                    break
                gene_name = t2i_dic[gene_id][0]
                gene_bt = t2i_dic[gene_id][1]
                mdtext += "| %i | %s | %s |  %s |\n" %(ol_c, gene_id, gene_name, gene_bt)
            mdtext += "| ... | &nbsp; | &nbsp; |  &nbsp; |\n"
            mdtext += "\n&nbsp;\n&nbsp;\n"


    print("Generate HTML report ... ")

    # Convert mdtext to html.
    md2html = markdown(mdtext, extensions=['attr_list', 'tables'])

    #OUTMD = open(md_out,"w")
    #OUTMD.write("%s\n" %(mdtext))
    #OUTMD.close()

    OUTHTML = open(html_out,"w")
    OUTHTML.write("%s\n" %(md2html))
    OUTHTML.close()


################################################################################

def get_ext_site_parts(id2bedrow_dic, chr_len_dic,
                       str_ext=150,
                       id2ucr_dic=False,
                       refid_dic=None,
                       extlen_dic=None,
                       id2extrow_dic=None,
                       id2newvp_dic=None):
    """
    Get extended site part lengths:
    [upstream structure calculation extension,
    upstream context extension,
    center region,
    downstream context extension,
    downstream structure calculation extension]

    id2bedrow_dic:
        Site ID to site BED row (region on reference), e.g.
        'sid1': 'id1\t950\t990\tsid1\t0\t+'
    chr_len_dic:
        reference (chromosome) ID -> reference length dic
        To check and prune extended sites at borders.
    str_ext:
        Amount of structure extension added on both sides,
        should be set to plfold_w.
    id2ucr_dic:
        viewpoint (uppercase) center region start and end
        coordinates for each site.
        Site ID -> [site_vp_start, site_vp_end]
    refid_dic:
        Store reference IDs from id2bedrow_dic.
    extlen_dic:
        Extended site lengths dic.
    id2extrow_dic:
        Extended reference BED row dictionary.
    id2newvp_dic:
        Stores new site viewpoint coordinates.
        Site ID -> [new_vp_start, new_vp_end]

    >>> id2bedrow_dic = {'sid1': 'id1\\t950\\t990\\tsid1\\t0\\t+', 'sid2': 'id1\\t500\\t540\\tsid2\\t0\\t+', 'sid3': 'id1\\t0\\t40\\tsid3\\t0\\t+', 'sid4': 'id1\\t10\\t50\\tsid4\\t0\\t-'}
    >>> id2ucr_dic = {'sid1': [11,30], 'sid2': [11,30], 'sid3': [11,30], 'sid4': [6,30]}
    >>> chr_len_dic = {'id1' : 1000}
    >>> get_ext_site_parts(id2bedrow_dic, chr_len_dic, str_ext=100, id2ucr_dic=id2ucr_dic)
    {'sid1': [100, 10, 20, 10, 10], 'sid2': [100, 10, 20, 10, 100], 'sid3': [0, 10, 20, 10, 100], 'sid4': [100, 5, 25, 10, 10]}

    """
    # Checks.
    assert id2bedrow_dic, "given id2bedrow_dic empty"
    assert str_ext > 0 and str_ext <= 500, "provide reasonable str_ext"
    # Part lengths dictionary.
    site2newl_dic = {}

    # Get new part lengths for each site.
    for site_id in id2bedrow_dic:
        cols = id2bedrow_dic[site_id].split("\t")
        seq_id = cols[0]
        assert seq_id in chr_len_dic, "sequence ID %s not in chr_len_dic" %(seq_id)

        ref_s = int(cols[1])
        ref_e = int(cols[2])
        site_id = cols[3]
        site_sc = cols[4]
        ref_pol = cols[5]
        site_len = ref_e - ref_s

        site_vp_s = 1
        site_vp_e = site_len
        if id2ucr_dic:
            assert site_id in id2ucr_dic, "site ID %s not in id2ucr_dic" %(site_id)
            site_vp_s = id2ucr_dic[site_id][0]
            site_vp_e = id2ucr_dic[site_id][1]
        site_vp_len = site_vp_e - site_vp_s + 1

        us_con_ext = site_vp_s - 1
        ds_con_ext = site_len - site_vp_e

        us_site_ext = str_ext
        ds_site_ext = str_ext

        ref_ext_s = ref_s - us_site_ext
        ref_ext_e = ref_e + ds_site_ext

        # Check for ends.
        ref_len = chr_len_dic[seq_id]

        if ref_ext_s < 0:
            us_site_ext += ref_ext_s
            ref_ext_s = 0
        if ref_ext_e > ref_len:
            diff = ref_ext_e - ref_len
            ds_site_ext = ds_site_ext - diff
            ref_ext_e = ref_len

        # Minus case, flip lengths.
        if ref_pol == "-":
            us_ext = ds_site_ext
            ds_ext = us_site_ext
            us_site_ext = us_ext
            ds_site_ext = ds_ext

        # Checks.
        ref_ext_l = ref_ext_e - ref_ext_s
        new_site_l = us_site_ext + us_con_ext + site_vp_len + ds_con_ext + ds_site_ext
        assert ref_ext_l == new_site_l, "ref_ext_l != new_site_l (%i != %i) for site ID %s" %(ref_ext_l, new_site_l, site_id)

        # Store extended BED rows.
        if id2extrow_dic is not None:
            id2extrow_dic[site_id] = "%s\t%i\t%i\t%s\t0\t%s" %(seq_id, ref_ext_s, ref_ext_e, site_id, ref_pol)

        # Store new site viewpoint start+end.
        if id2newvp_dic is not None:
            new_vp_s = site_vp_s + us_site_ext
            new_vp_e = site_vp_e + us_site_ext
            id2newvp_dic[site_id] = [new_vp_s, new_vp_e]

        # Store reference IDs.
        if refid_dic is not None:
            refid_dic[seq_id] = 1

        if extlen_dic is not None:
            extlen_dic[site_id] = ref_ext_l

        # Store new part lengths.
        site2newl_dic[site_id] = [us_site_ext, us_con_ext, site_vp_len, ds_con_ext, ds_site_ext]

    assert site2newl_dic, "nothing stored inside site2newl_dic"
    return site2newl_dic


################################################################################

def calc_ext_str_features(id2bedrow_dic, chr_len_dic,
                          out_bpp, out_str, args,
                          id2ucr_dic=False,
                          stats_dic=None,
                          bp_check_seqs_dic=False,
                          tr_regions=False,
                          tr_seqs_dic=False):
    """
    Calculate structure features (base pairs and structural element
    probabilities) by using extended sequences, and then prune them to
    viewpoint + context parts to match remaining feature lists.

    id2bedrow_dic:
        Site ID to BED region (tab separated)
    id2ucr_dic:
        Site ID -> [viewpoint_start, viewpoint_end]
        with 1-based coords.
    chr_len_dic:
        Reference sequence lengths dictionary.
    out_bpp:
        output .bpp.str file.
    out_str:
        Output .elem_p.str file.
    args:
        Arguments from graphprot2 gt / gp.

    """
    assert id2bedrow_dic, "id2bedrow_dic empty"
    assert chr_len_dic, "chr_len_dic empty"

    print("Extend sequences by --plfold-w for structure calculations ... ")

    # Get extended parts and infos.
    id2newvp_dic = {}
    id2extrow_dic = {}
    extlen_dic = {}
    refid_dic = {}
    id2newl_dic = get_ext_site_parts(id2bedrow_dic, chr_len_dic,
                                     str_ext=args.plfold_w,
                                     id2ucr_dic=id2ucr_dic,
                                     refid_dic=refid_dic,
                                     extlen_dic=extlen_dic,
                                     id2extrow_dic=id2extrow_dic,
                                     id2newvp_dic=id2newvp_dic)

    # Checks.
    assert id2extrow_dic, "id2extrow_dic empty"

    # tmp files.
    random_id = uuid.uuid1()
    tmp_fa = str(random_id) + ".tmp.fa"
    random_id = uuid.uuid1()
    tmp_bed = str(random_id) + ".tmp.bed"
    random_id = uuid.uuid1()
    tmp_bpp_out = str(random_id) + ".tmp.bpp.str"
    random_id = uuid.uuid1()
    tmp_elem_p_out = str(random_id) + ".tmp.elem_p.str"

    # If transcript regions.
    if tr_regions:
        # Checks.
        assert tr_seqs_dic, "tr_seqs_dic empty"
        for ref_id in refid_dic:
            assert ref_id in tr_seqs_dic, "reference ID %s not in tr_seqs_dic" %(ref_id)
        # Get extended sequences.
        seqs_dic = extract_transcript_sequences(id2extrow_dic, tr_seqs_dic)
        # Write sequences to FASTA.
        fasta_output_dic(seqs_dic, tmp_fa)
    else:
        # Genomic regions.
        bed_write_row_dic_into_file(id2extrow_dic, tmp_bed)
        # Extract sequences.
        bed_extract_sequences_from_2bit(tmp_bed, tmp_fa, args.in_2bit)

    # Check extracted sequences, replace N's with random nucleotides.
    polish_fasta_seqs(tmp_fa, extlen_dic,
                      vp_check_seqs_dic=bp_check_seqs_dic,
                      vp_dic=id2newvp_dic)
    calc_str_elem_up_bpp(tmp_fa, tmp_bpp_out, tmp_elem_p_out,
                                stats_dic=stats_dic,
                                id2ucr_dic=id2newvp_dic,
                                plfold_u=args.plfold_u,
                                plfold_l=args.plfold_l,
                                plfold_w=args.plfold_w)

    print("Post-process structure files ... ")

    # Refine elem_p.str.
    str_elem_p_dic = read_str_elem_p_into_dic(tmp_elem_p_out,
                                              p_to_str=True)
    assert str_elem_p_dic, "str_elem_p_dic empty"

    SEPOUT = open(out_str,"w")
    for site_id in str_elem_p_dic:
        us_ext = id2newl_dic[site_id][0]
        ds_ext = id2newl_dic[site_id][4]
        # Checks.
        len_ll = len(str_elem_p_dic[site_id])
        total_ext = us_ext + ds_ext
        assert len_ll > total_ext, "len_ll <= total_ext for site ID %s" %(site_id)
        if ds_ext:
            new_ll = str_elem_p_dic[site_id][us_ext:-ds_ext]
        else:
            # If ds_ext == 0.
            new_ll = str_elem_p_dic[site_id][us_ext:]
        assert new_ll, "new_ll empty for site ID %s (us_ext = %i, ds_ext = %i, len_ll = %i)" %(site_id, us_ext, ds_ext, len_ll)
        SEPOUT.write(">%s\n" %(site_id))
        for l in new_ll:
            s = "\t".join(l)
            SEPOUT.write("%s\n" %(s))
    SEPOUT.close()

    # Refine .bpp.str.
    seq_id = ""
    us_ext = 0
    ds_ext = 0
    max_len = 0

    # Pairing rules for sanity checking new base pair coords.
    bp_nts_dic = {
        "A" : ["U"],
        "C" : ["G"],
        "G" : ["C","U"],
        "U" : ["A","G"]
    }

    BPPOUT = open(out_bpp,"w")
    with open(tmp_bpp_out) as f:
        for line in f:
            if re.search(">.+", line):
                m = re.search(">(.+)", line)
                seq_id = m.group(1)
                us_ext = id2newl_dic[seq_id][0]
                ds_ext = id2newl_dic[seq_id][4]
                # Original length.
                max_len = extlen_dic[seq_id] - us_ext - ds_ext
                BPPOUT.write(">%s\n" %(seq_id))
            else:
                m = re.search("(\d+)\t(\d+)\t(.+)", line)
                s = int(m.group(1))
                e = int(m.group(2))
                bpp = m.group(3)
                new_s = s - us_ext
                new_e = e - us_ext
                # Filter.
                if new_s < 1:
                    continue
                if new_e < 1:
                    continue
                if new_s > max_len:
                    continue
                if new_e > max_len:
                    continue
                if bp_check_seqs_dic:
                    seq = bp_check_seqs_dic[seq_id].upper()
                    nt1 = seq[new_s-1]
                    nt2 = seq[new_e-1]
                    nt2_list = bp_nts_dic[nt1]
                    assert nt2 in nt2_list, "invalid base pair coordinates encountered for site ID %s (%s cannot pair with %s)" %(seq_id, nt1, nt2)
                BPPOUT.write("%i\t%i\t%s\n" %(new_s, new_e, bpp))
    BPPOUT.close()

    # Remove tmp files.
    if os.path.exists(tmp_fa):
        os.remove(tmp_fa)
    if os.path.exists(tmp_bed):
        os.remove(tmp_bed)
    if os.path.exists(tmp_bpp_out):
        os.remove(tmp_bpp_out)
    if os.path.exists(tmp_elem_p_out):
        os.remove(tmp_elem_p_out)


################################################################################

def polish_fasta_seqs(in_fa, len_dic,
                      vp_dic=False,
                      vp_check_seqs_dic=False,
                      report=False,
                      repl_alphabet=["A","C","G","U"]):
    """
    Read in FASTA sequences, check lengths, and replace N's with
    nucleotide characters from repl_alphabet. Overwrite original in_fa
    with polished content.

    """
    import random
    assert len_dic, "len_dic empty"
    # Read in FASTA (do not skip N containing extensions here).
    seqs_dic = read_fasta_into_dic(in_fa,
                                   skip_n_seqs=False)
    assert len(seqs_dic) == len(len_dic), "len(seqs_dic) != len(len_dic)"
    FAOUT = open(in_fa,"w")
    for seq_id in seqs_dic:
        seq = seqs_dic[seq_id].upper()
        assert seq_id in len_dic, "sequence ID %s not in seqs_dic" %(seq_id)
        assert len(seq) == len_dic[seq_id], "sequence length != len_dic length (%i != %i)" %(len(seq), len_dic[seq_id])
        new_seq = seq
        if re.search("N", seq):
            if report:
                print("WARNING: N nucleotides encountered for sequence ID %s. Apply polishing ... " %(seq_id))
            new_seq = ""
            for c in seq:
                new_c = c
                if c == "N":
                    new_c = random.choice(repl_alphabet)
                new_seq += new_c
        if vp_dic:
            assert seq_id in vp_dic, "sequence ID %s not in vp_dic" %(seq_id)
            vp_s = vp_dic[seq_id][0]
            vp_e = vp_dic[seq_id][1]
            new_seq = update_sequence_viewpoint(new_seq, vp_s, vp_e)
            if vp_check_seqs_dic:
                assert seq_id in vp_check_seqs_dic, "sequence ID %s not in vp_check_seqs_dic" %(seq_id)
                vp_seq1 = seq_get_vp_region(new_seq)
                vp_seq2 = seq_get_vp_region(vp_check_seqs_dic[seq_id])
                assert vp_seq1, "uppercase sequence region extraction failed for vp_seq1 (ID: %s, seq: %s)" %(seq_id, new_seq)
                assert vp_seq2, "uppercase sequence region extraction failed for vp_seq2 (ID: %s, seq: %s)" %(seq_id, vp_check_seqs_dic[seq_id])
                assert vp_seq1 == vp_seq2, "vp_seq1 != vp_seq2 for ID %s (\"%s\" != \"%s\")" %(seq_id, vp_seq1, vp_seq2)
        FAOUT.write(">%s\n%s\n" %(seq_id,new_seq))
    FAOUT.close()


################################################################################

def get_major_lc_len_from_seqs_dic(seqs_dic):
    """
    Go through sequences dictionary (sequence iD -> sequence) and extract
    longest lowercase part for each sequence. Count how many times each
    length appears and return most frequent length.

    >>> seqs_dic = {'id1': 'acgACGUac', 'id2': 'ACGU', 'id3': 'GUacguaa', 'id4': 'cguACGU'}
    >>> get_major_lc_len_from_seqs_dic(seqs_dic)
    3
    >>> seqs_dic = {'id1': 'CCAA', 'id2': 'ACGU'}
    >>> get_major_lc_len_from_seqs_dic(seqs_dic)
    False

    """
    assert seqs_dic, "given seqs_dic empty"
    # Lowercase part length counts dictionary.
    lcl_dic = {}
    for seq_id in seqs_dic:
        m = re.search("([acgtun]+)", seqs_dic[seq_id])
        if m:
            lc_len = len(m.group(1))
            if lc_len not in lcl_dic:
                lcl_dic[lc_len] = 1
            else:
                lcl_dic[lc_len] += 1
    major_len = False
    if lcl_dic:
        for lcl, lcc in sorted(lcl_dic.items(), key=lambda item: item[1], reverse=True):
            major_len = lcl
            break
    return major_len


################################################################################

def seq_get_vp_region(seq):
    """
    Get viewpoint (uppercase region) from a sequence.

    >>> seq = "acguAACCGGacgu"
    >>> seq_get_vp_region(seq)
    'AACCGG'

    """
    assert seq, "seq empty"
    vp_seq = False
    m = re.search("[acgun]*([ACGUN]+)", seq)
    if m:
        vp_seq = m.group(1)
    return vp_seq


################################################################################

def drop_a_line():
    """
    Drop a line.

    """
    lines = []
    a = """
       \"Of all the RBP-BSP tools in the world,
           this is definitely one of them.\"
"""
    b = """
                     \"Let's Party!\"
"""
    c = """
          \"I eat Green Berets for breakfast.\"
"""
    d = """
           \"There's always barber college.\"
"""

    #lines.append(a)
    lines.append(b)
    lines.append(c)
    lines.append(d)
    return(random.choice(lines))


################################################################################
################################################################################

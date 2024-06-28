# This script is used to design a library of guides and targets to test the terminal adjacent mismatch hypothesis.
# 4 viral species are randomly selected, and a site in each virus is mutated to generate the target library. The guide library is generated to introduce or avoid the introductin of this mismatch.

import matplotlib.pyplot as plt
import fastaparser
import os
import numpy as np
import pandas as pd
from badgers.utils import prepare_sequences as prep_seqs 
from random import sample
import random

def DNA(length):
    return ''.join(random.choice('CGTA') for _ in range(length))

data_dir = './data/random-search/'

# Randomly sample viruses from a list of all vertebrate infecting viruses. This list was
# previously made by the authors of the ADAPT manuscript.
virus_df = pd.read_csv(data_dir + 'all-vertebrate.tsv', sep = '\t')
virus_df = virus_df[virus_df['neighbor-count'] > 100]
virus_df

virus_df.groupby(['family'], sort=False).apply(pd.DataFrame.sample, n = 1).sample(n = 4)

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

def edit_seq(seq, pos, new_nt):
    seq = list(seq)
    seq[pos] = new_nt
    return "".join(seq)

order_seqs_df = pd.DataFrame()
order_guides_df = pd.DataFrame()

pos27_PFS = ['AG', 'CG', 'GG', 'TG']
for i, virus_file in enumerate([x for x in os.listdir('../alignments/tam_library/') if '.fasta' in x]):
    with open('../alignments/tam_library/' + virus_file) as fasta_file:           
        parser = fastaparser.Reader(fasta_file)
        for ix, seq in enumerate(parser):
            ref_seq = seq.sequence_as_string()
            
    position = sample(list(find_all(ref_seq, pos27_PFS[i])), 1)
    position = int(position[0])
    
    context_nt = 150
    starting_seq = ref_seq[position-37 - context_nt:position+11 + context_nt]
    curr_seqs = []
    
    for new_nt in list('ACGT'):
        curr_seqs.append(edit_seq(starting_seq, context_nt + 39, new_nt))
    
    for new_nt in list('ACT'):
        curr_seqs.append(edit_seq(starting_seq, context_nt + 38, new_nt))
                        
    t7_promoter = "gaaatTAATACGACTCACTATAgggCACTATAGGGGCTCTAGCGACTTCTTTAAATAGTGGCTTAAAATAAC"
    
    for tt, tseq in enumerate(curr_seqs):
        target_seq_to_order = (t7_promoter + tseq).lower()

        order_seqs_df = order_seqs_df.append(
            pd.DataFrame({'virus name': [virus_file.split('.fasta')[0]], 'starting_pos': [-context_nt + position-37], 
                          'target group': [pos27_PFS[i]], 'PFS-adjacent allele' : [tseq[context_nt + 37]], 'PFS allele' : [tseq[context_nt + 38]], 
                          'target_name': [f'target_gPFS_group{i}_{tt}'],  'target_seq (no t7 promoter)': [tseq], 'target_seq_to_order': [ 
    target_seq_to_order]}))
        
    curr_guides = []
    guide_starting = starting_seq[context_nt+10:context_nt + 38]
    for new_nt in list('ACGT'):
        curr_guides.append(edit_seq(guide_starting, 27, new_nt))
        
    direct_repeat_rna = "GAUUUAGACUACCCCAAAAACGAAGGGGACUAAAAC" 
    for xx, guide_seq in enumerate(curr_guides):
        assert len(guide_seq) == 28
        order_guide_seq = prep_seqs.sub_t_u(direct_repeat_rna + prep_seqs.revcomp(guide_seq))
            
        order_guides_df = order_guides_df.append(
            pd.DataFrame({'virus name': [virus_file.split('.fasta')[0]], 'starting_pos': [position-37], 
                          'target group': [pos27_PFS[i]], 'PFS-adjacent allele' : [guide_seq[27]], 
                          'guide_name': [f'guide_gPFS_group{i}_{xx}'],
                          'guide_seq (protospacer; not revcomped)': [guide_seq], 'guide_seq_to_order': [
    order_guide_seq]}))
        
# These two dataframes contain the guide and the target sequences needed to test the hypothesis
order_seqs_df = order_seqs_df.reset_index(drop = True)
order_seqs_df.to_csv('tam_library_targets.csv', index = False)
order_guides_df = order_guides_df.reset_index(drop = True)
order_seqs_df.to_csv('tam_library_guides.csv', index = False)


#!/bin/bash
# Script to benchmark the generative design algorithms on designing guides for SNP identification
# The input file is a pickled dataframe which contains pairs of 100 random sequences that have one nucleotide different between them
# The pickled dataframe has 2800 rows, each row containing a pair of sequences. There are 100 unique pairs of sequences, with 28 possible positionings of the SNP within the guide.

python design_guides.py diff all ./processed_sites/validation_100seqs_synmismatch.pkl ./gen_results/ext_validation/ --num_cpu 80 --verbose_results --processed_sites_path --save_pickled_results

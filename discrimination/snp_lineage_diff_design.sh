#!/bin/sh
# Script to run the algorithms in BADGERS to design guides that perform SNP identification and differentiate between viral lineages
 
# Here, we design guides for six key clinically relevant SNPs:
# Each .fasta file has two sequence: the WT sequence and the sequence with the variant, so we use the option  --all_targets_one_file
for file in ./alignments/SNP-design/*
do   
    files_list+=($file)
    read -a virus_name <<< "$file"
    echo $file
    python design_guides.py diff both ./alignments/SNP-design/$file/ ./gen_results/SNP-design/${virus_name[0]}/ --num_cpu 10 --verbose_results --all_targets_one_file --save_to_pickle --output_to_single_directory --diff_n_top_guides_per_site 5
done 
 
# Here, we design guides to differentiate between the dengue virus lineages:
# The dengue virus sequences were downlaoded from ViPR and aligned with MAFFT (between serotypes) and then split out into separate .fasta files for each serotype
python design_guides.py diff both ./alignments/full-genome-DENV/ ./gen_results/DENV-design/ --num_cpu 90 --verbose_results --save_to_pickle --output_to_single_directory 

# Here, we design guides to differentiate between the SARS-CoV-2 lineages:
# The SARS-CoV-2 sequences were downloaded from GISAID and aligned with MAFFT
# All the sequences for the different lineages are in the same .fasta file, so we use the option  --all_targets_one_file
python design_guides.py diff both './alignments/full-genome-COV2/final alignment VOI.fasta' ./gen_results/full-genome-COV2/ --num_cpu 80 --verbose_results --save_to_pickle --output_to_single_directory
#!/bin/sh

#Use the model-guided exploration algorithms to design guides for all 
# valid genomic sites in the five viral species considered

IFS=".fasta.0"

for file in ./alignments/final-full-genome-design/*
do   
    files_list+=($file)
    read -a virus_name <<< "$file"
    echo $file
    python design_guides.py multi all ./processed_sites/final-full-genome-design/${virus_name[0]}_df.pkl ./gen_results/final-full-genome-design/${virus_name[0]}/ --num_cpu 80 --verbose_results --benchmarking --save_pickled_results --output_to_single_directory 
done  
 
# Previously, used the 'parallel' utility to run one parallel job for each genomic site, but decided to use
# python's multiprocessing library (as implemented above) instead.
"""
This script houses functions that are used to process genomic sites for multi-target detection.

The functions enable the creation of the multi-target random search dataset and the
processing of the full-genome-design benchmark datasets.
"""

import fastaparser
import os
import pandas as pd
import numpy as np
from collections import Counter
from operator import itemgetter
import sys
from scipy import stats


from adapt import alignment
from adapt.prepare import cluster
from adapt.utils import seq_io


def find_test_targets(target_start, target_end, aln_path, num_representative_targets = False):
    """
    This function is taken from ADAPT's pick_test_targets.py file
    (https://github.com/broadinstitute/adapt/blob/b8464e7bb2dda44cb9a24325fef9ce0c93951c5e/bin/pick_test_targets.py)

    This function is located within a file that isn't a part of the standard ADAPT package, so it couldn't be imported.
    Here, we use it to find a set of targets sequences that represent the sequence diversity encountered at a genomic site
    within the alignment.

    Args:
        target_start: The start position of the target site
        target_end: The end position of the target site
        aln_path: The path to the alignment file
    Returns:
        rep_seqs: A list of representative sequences
        rep_seqs_frac: The fraction of sequence diversity represented by each of the representative sequences
        target_start: The start position of the representative sequences
        target_end: The end position of the representative sequences
    """
    min_seq_len_to_consider = 80 
    min_target_len = 250
    min_frac_to_cover_with_rep_seqs = 0.95
    max_cluster_distance = 0.1

    print('Reading in Fasta...')
    seqs = seq_io.read_fasta(aln_path)
    aln = alignment.Alignment.from_list_of_seqs(list(seqs.values()))

    # Expand the extracted range to the minimum
    # Note that this extracts to get a target length of args.min_target_len
    # *in the alignment*; if there are gaps in the alignment within the region, the actual length of the sequences could be shorter
    target_len = target_end - target_start
    nt_to_add = int(max(0, min_target_len - target_len) / 2)
    target_start = max(0, target_start - nt_to_add)
    target_end = min(aln.seq_length, target_end + nt_to_add)
    if target_end - target_start == min_target_len - 1:
        # Fix off-by-1 edge case
        target_end = min(aln.seq_length, target_end + 1)

    # Extract the alignment where this design target (amplicon) is
    aln_extract = aln.extract_range(target_start, target_end)
    # Pull out the sequences, without gaps
    aln_extract_seqs = aln_extract.make_list_of_seqs(remove_gaps=True)

    # Remove target sequences that are too short
    # This can happen due to gaps in the alignment; a sequence can have
    # length 0, for example, if is it all '-' in the amplicon (extracted
    # range)
    # If they are extremely short (shorter than minhash_k, below), then
    # this will cause an error downstream
    aln_extract_seqs = [s for s in aln_extract_seqs
            if len(s) >= min_seq_len_to_consider]

    # Add indices for each sequence so it can be used as a dict
    aln_extract_seqs_dict = {i: s for i, s in enumerate(aln_extract_seqs)}

    # The number of k-mers to use with MinHash for clustering cannot be
    # more than the number of k-mers in a sequence
    minhash_k = 12
    min_seq_len = min(len(s) for s in aln_extract_seqs)
    minhash_N = min(50, min_seq_len - minhash_k - 1)

    # Find representative sequences
    if num_representative_targets: 
        # Use a maximum number of clusters, ignoring the inter-cluster
        # distance threshold
        threshold = None
        num_clusters = num_representative_targets

    else:
        # Use an inter-cluster distance threhsold
        threshold = max_cluster_distance
        num_clusters = None

    print('Finding Representative Seqs...')
    rep_seqs_idx, rep_seqs_frac = cluster.find_representative_sequences(
            aln_extract_seqs_dict,
            k=minhash_k, N=minhash_N, threshold=threshold,
            num_clusters=num_clusters,
            frac_to_cover=min_frac_to_cover_with_rep_seqs)
    rep_seqs = [aln_extract_seqs[i] for i in rep_seqs_idx]

    return rep_seqs, rep_seqs_frac, target_start, target_end


def run_adapt(exp_name, virus_name, start_pos, seqs):
    """
    This function runs ADAPT on a given set of sequences,
    and is used in our case to run ADAPT on a randomly-chosen genomic site
    Args:
        exp_name: The name of the experiment
        virus_name: The name of the virus
        start_pos: The start position of the target site
        seqs: The sequences in the alignment at the selected genomic position
    Returns:
        adapt_guide: The ADAPT guide that was designed, or returns 0 if design failed
    """

    fasta_path = f"../../detection/processed_sites/{exp_name}/{virus_name}.{start_pos} adapt_random_sites alignment.fasta"

    with open(fasta_path, 'w') as fasta_file:
        writer = fastaparser.Writer(fasta_file)
        for idx, seq in enumerate(seqs):
            writer.writefasta((f"{virus_name} site {start_pos} sequence#{idx}", seq))

    outpath = f'../../detection/adapt_results/{exp_name}/{virus_name}.{start_pos} random_sites guides.tsv' 

    cmd = f"design.py sliding-window fasta '{fasta_path}' -o '{outpath}' "
    cmd = cmd + "-w 48 -gl 28 --predict-cas13a-activity-model --obj maximize-activity -hgc 1 --maximization-algorithm greedy --verbose --debug"
    print('Running... ' + cmd)
    os.system(cmd)
    
    if(not os.path.exists(outpath)):
        print('ADAPT FILE DOES NOT EXIST')
        return 0

    if(len(pd.read_csv(outpath, sep = '\t')['target-sequences'].values) > 0):
        adapt_guide = pd.read_csv(outpath, sep = '\t')['target-sequences'].values[0]
        return adapt_guide
    
    else:
        return 0
    
def select_random_site(genome_sections, section, seqs, G_PFS):
    """
    Selects a random site from a given section of the genome
    Args:
        genome_sections: A list of lists, where each list contains the indices of the genome that are in that section
        section: The section of the genome to select a site from
        seqs: The sequences in the alignment
        G_PFS: If True, then G PFS sites are allowed
    """

    found_site = False

    print(f'Finding Site in Section {section}')

    num_sites = 0
    while(found_site == False):
        start_pos = np.random.choice(genome_sections[section], 1)[0]

        targets = [seq[start_pos:start_pos+48] for seq in seqs 
        if all(char in ['A', 'C', 'T', 'G'] for char in seq[start_pos:start_pos+48])]
        
        num_sites += 1

        found_site = True

        if(len(targets) < 10):
            found_site = False

        if(not G_PFS):
            PFS = Counter(map(itemgetter(38), targets))
            PFS_nt = max(PFS, key=PFS.get)

            if(PFS_nt == 'G'):
                found_site = False

        if(num_sites > 10):
            print(f'Tried {num_sites} sites in section {section}') 

        if(num_sites > 400):
            print(f'Failed to Find Site in Section {section}')
            if(section + 2 <= len(genome_sections)):
                return select_random_site(genome_sections, section + 1, seqs, G_PFS)
            else:
                return select_random_site(genome_sections, section - 1, seqs, G_PFS)

    return start_pos

def detection_import_random_sites(exp_name, num_sites, G_PFS = False, alt_output = False, seed = 1):
    """
    This function takes a folder of input genomes, selects random sites from each of the genomes,
    runs ADAPT on these random sites, and then saves the results to a pickled sites_df file. 

    The genome is split into num_sites sections linearly, and one site is chose from each genome section.

    Args:
        exp_name: The name of the experiment
        num_sites: The number of sites to select from each genome
        G_PFS: If True, then G PFS sites are allowed
        alt_output: If you want to save the results to a different folder, specify it here
        seed: The random seed to use
    """

    np.random.seed(seed)
    df_list = []
    directory = '../../detection/alignments/{}/'.format(exp_name)

    print(f"List of Files to Import {os.listdir('../../detection/alignments/{}/'.format(exp_name))}")

    for alignment_file in os.listdir('../../detection/alignments/{}/'.format(exp_name)):       
        
        seqs = []
        items = ['target_set', 'start_pos', 'seq_id', 'virus_name', 'adapt_guide', 'test_targets', 'site_num', 'target_start', 'target_end',
        'adapt_test_targets', 'gen_test_targets', 'test_targets_frac_covered', 'adapt_start_pos']
        seqs_dict = {item: [] for item in items}
        
        
        with open('../../detection/alignments/{}/'.format(exp_name) + alignment_file) as fasta_file:           
            parser = fastaparser.Reader(fasta_file)
            for seq in parser:
                seqs.append(seq.sequence_as_string())

        min_length = min([len(seq) for seq in seqs])
        genome_sections = list(np.array_split(range(min_length), num_sites))

        virus_name = alignment_file.split(' alignment.fasta')[0]
        print('Importing.... ' + virus_name)

        start_pos_list = [] 
        section_list = []
        for section in range(num_sites):
            start_pos_list.append(select_random_site(genome_sections, section, seqs, G_PFS))
            section_list.append(section)

        for section, start_pos in zip(section_list, start_pos_list):
            
            test_targets, frac_covered, target_start, target_end = find_test_targets(start_pos-125, start_pos+125 + 48, directory + alignment_file)

            seqs_chars = np.array([list(seq) for seq in seqs])
            mode, count = stats.mode(seqs_chars)
            gaps = list(mode[0][int(target_start):start_pos]).count('-')
            
            targets = [seq[start_pos:start_pos+48] for seq in seqs 
            if all(char in ['A', 'C', 'T', 'G'] for char in seq[start_pos:start_pos+48])]

            print(f'Running ADAPT For Section {section}')
            adapt_guide = run_adapt(exp_name, virus_name, start_pos, targets)

            if(len(section_list) > 7 * num_sites):
                print('SUITABLE SITES COULD NOT BE FOUND')
                print(f'ONLY FOUND {len(seqs_dict)} sites')
                break

            if(adapt_guide == 0):
                # ADAPT Design FAILED, selecting another site
                print(f'Section {section}')
                start_pos_list.append(select_random_site(genome_sections, section, seqs, G_PFS))
                section_list.append(section)
                continue
            
            # Saving the randomly chosen sites, the ADAPT results, and the test targets to the dataframe
            seqs_dict['target_set'].append(targets)
            seqs_dict['start_pos'].append(start_pos)           
            seqs_dict['seq_id'].append(virus_name + '.' + str(start_pos))
            seqs_dict['virus_name'].append(virus_name)
            seqs_dict['adapt_start_pos'].append(start_pos)
            seqs_dict['adapt_guide'].append(adapt_guide)
            seqs_dict['test_targets'].append(test_targets)
            seqs_dict['test_targets_frac_covered'].append(frac_covered)
            seqs_dict['site_num'].append(section)
            seqs_dict['target_start'].append(target_start)
            seqs_dict['target_end'].append(target_end)
            seqs_dict['adapt_test_targets'].append([target[start_pos - target_start - gaps: start_pos + 48 - target_start - gaps] for target in test_targets])
            seqs_dict['gen_test_targets'].append([target[start_pos - target_start - gaps: start_pos + 48 - target_start - gaps] for target in test_targets])

        virus_df = pd.DataFrame(seqs_dict)

        if(alt_output != False):
            virus_df.to_pickle('./detection/processed_sites/{}/'.format(alt_output) + '{}_df.pkl'.format(virus_name))
        else:

            if not os.path.exists('./detection/processed_sites/{}/'.format(exp_name)):
                os.makedirs('./detection/processed_sites/{}/'.format(exp_name))

            virus_df.to_pickle('./detection/processed_sites/{}/'.format(exp_name) + '{}_df.pkl'.format(virus_name))

        df_list.append(virus_df)
        print(df_list)
        print(f'Finished Importing {virus_name}')
            
    master_df = pd.concat(df_list).reset_index(drop = True)

    if(alt_output != False):
        master_df.to_pickle('../../detection/processed_sites/{}/'.format(alt_output) + 'seqs_df.pkl')
    else:
        master_df.to_pickle('../../detection/processed_sites/{}/'.format(exp_name) + 'seqs_df.pkl')

    return master_df

def detection_import_adapt_full_genome(exp_name, specific_virus = False):
    """
    Takes a folder of viral alignments and then parses them into Pandas 
    dataframes for the generative models. Within the exported dataframe,
    each row represents a genomic site. The design results for ADAPT are included for each
    genomic site so we can benchmark the performance of them compared to the generative design algorithms.

    Args:
        exp_name: The name of the experiment
        specific_virus: If you only want to run the script on a specific virus, specify it here
    """

    directory = '../alignments/{}/'.format(exp_name)
    adapt_directory = '../adapt_results/{}/'.format(exp_name)
    out_dir = '../processed_sites/{}/'.format(exp_name) 

    print('Importing ADAPT Results')

    df_list = []
    adapt_files = sorted([x for x in os.listdir(adapt_directory) if '.tsv.0' in x])

    if(specific_virus):
        adapt_files = [x for x in adapt_files if specific_virus in x]
        
    # Reading in the ADAPT results
    print(adapt_files)
    for file in adapt_files:       
        adapt_df = pd.read_csv(adapt_directory + file, sep = '\t').dropna()
        virus_name = file.split('.tsv.0')[0].split('guides')[1]
        print(virus_name)
        print(adapt_df)
        alignment_file = [x for x in os.listdir(directory) if virus_name in x and '.fasta' in x][0] 
        
        seqs = []
        with open(directory + alignment_file) as fasta_file:           
                parser = fastaparser.Reader(fasta_file)
                for seq in parser:
                    seqs.append(seq.sequence_as_string())
     
        items = ['target_set', 'start_pos', 'seq_id', 'virus_name', 'adapt_guide', 'adapt_start_pos']
        seqs_dict = {item: [] for item in items}

        z = 0
        for start_pos, end_pos, adapt_guide, adapt_guide_start in zip(adapt_df['window-start'].values, adapt_df['window-end'].values, adapt_df['target-sequences'].values, adapt_df['target-sequence-positions'].values):
                
                # start_pos and end_pos are the start and end of the amplicon which adapt provided, so we need to add context
                start_pos = int(start_pos) - 10
                end_pos = int(end_pos) + 10

                # Only consider valid target sets
                targets = [seq[start_pos:start_pos+48] for seq in seqs 
                if all(char in ['A', 'C', 'T', 'G'] for char in seq[start_pos:start_pos+48])]

                # If less than 80% of the sequences are valid, then don't considser this site
                if(len(targets) <= len(seqs) * 0.8):
                    continue
        
                # Saving the target set parameters and the adapt results to the dataframe          
                seqs_dict['target_set'].append(targets)
                seqs_dict['start_pos'].append(start_pos)           
                seqs_dict['seq_id'].append(virus_name + '.' + str(start_pos))
                seqs_dict['virus_name'].append(virus_name)
                seqs_dict['adapt_start_pos'].append(start_pos)
                seqs_dict['adapt_guide'].append(adapt_guide)
 
                z += 1

        virus_df = pd.DataFrame(seqs_dict)
        print(virus_df)
        virus_df.to_pickle(out_dir + '{}_df.pkl'.format(virus_name))
        df_list.append(virus_df)
            
    master_df = pd.concat(df_list).reset_index(drop = True)
    master_df.to_pickle(out_dir + 'seqs_df.pkl')


if __name__ == '__main__':

    # This serves as the random_search dataset
    # Since this is for random search, we don't actually end up using the adapt guides or the test targets exported from this function
    # detection_import_random_sites('random_search', 10, G_PFS = True, alt_output = '100random_sites_withG')

    # Validation dataset
    detection_import_random_sites('random_search', 10, G_PFS = True, alt_output = '100random_sites_withG_validation_set')  

    # This enables us to generate the processed sites dataframes for the full-genome-design benchmarks
    # detection_import_adapt_full_genome('final-full-genome-design')
    
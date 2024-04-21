# This notebook generates the main-text and supplementary figures describing the performance of different methods in designing guides for the multi-target detection objective 
# across the five viruses considered in the manuscript

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns 
import random
from scipy import stats
from dragon.utils import prepare_sequences as prep_seqs
sys.path.append('../detection/alignments/')
import process_genomic_sites as import_fasta  
from matplotlib.colors import ListedColormap
import random
from random import sample

sns.set_theme(font="Helvetica", style='ticks')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

from matplotlib import font_manager
font_manager.fontManager.addfont('/home/ubuntu/Helvetica.ttf')

plt.rcParams['font.size'] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams["legend.frameon"] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.top'] = False

plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.family'] = 'Helvetica'

cmap = plt.cm.get_cmap('viridis')
gcmap = plt.cm.get_cmap('gray')
base_col = "#bbbbbb"
adapt_col =  "#555555" 
evolutionary_col = "#55ad70"
wgan_col =  (.35, .09, .35) 
adalead_col = '#5661E9'
cbas_col = '#D66B16'

exp = 'final-full-genome-design'
fig_dir ='./detection/{}/'.format(exp)
data_dict = {}

virus_names = [x for x in os.listdir("./detection/gen_results/{}/".format(exp)) if '.' not in x]

# Reading in all the files for each of the genomic sites across the viruses considered
for virus in sorted(virus_names):
    print(virus)
    nfiles = len(os.listdir('./detection/gen_results/{}/{}/'.format(exp, virus)))
    print(nfiles)
    
    data_dict[virus] = extract_df(virus)
        
print(len(data_dict.keys()))


# Main Text Figures

# Dictionary for renaming shortened viral name to real name
renaming = {'FLUA': 'influenza A segment 2', 'LASVsegS': 'Lassa virus segment S', 'SARSCOV2': 'SARS-CoV-2'
           , 'DENV': 'dengue virus', 'EVB': 'Enterovirus B'} 

# Plotting the percentage of sequences the guides from each method can detect across different genomic windows

# Function that goes through all the input windows and determines the maximum percentage of sequence diversity the method's guides 
# can detect within that window
def windowMax(kk, x, k):
        
    output = []
    for i in range(len(x)-1): 
        output.append(kk[(kk.start_pos).isin(range(x[i], x[i+1]))].perc_highly_active.max() * 100)
    return output

# Plotting the results by window
a = 1
lw = 55
fig, ax = plt.subplots(nrows = len(data_dict.keys()), ncols = 1, figsize = (10, 4 * len(data_dict.keys()))) 

# Defining the length of each of the five viruses considered
virus_length = [11352, 8216, 2455, 3490, 40974]

for i, virus in enumerate(data_dict.keys()):
    df_dict = data_dict[virus]
    wsize = int(virus_length[i]/15)
    
    x = np.arange(0, virus_length[i], wsize)
    x_plot = x[:-1]
   
    its = 1/7
    ax[i].scatter(x_plot+ its*wsize,windowMax(df_dict['consensus'], x, wsize), label = 'Consensus guides', alpha = a, c = base_col, s = lw)
    ax[i].scatter(x_plot + 2*its*wsize,windowMax(df_dict['adapt'], x, wsize), label = 'ADAPT guides', alpha = a, c = adapt_col, s = lw)
    ax[i].scatter(x_plot + 3*its*wsize,windowMax(df_dict['wgan'], x, wsize), label = 'WGAN-AM guides', alpha = a, c = wgan_col, s = lw)
    ax[i].scatter(x_plot + 4*its*wsize,windowMax(df_dict['genetic'], x, wsize), label = 'Evolutionary guides', alpha = a, c = evolutionary_col, s = lw)
    ax[i].scatter(x_plot + 5*its*wsize,windowMax(df_dict['adalead'], x, wsize), label = 'AdaLead guides', alpha = a, c = adalead_col, s = lw)
    ax[i].scatter(x_plot + 6*its*wsize,windowMax(df_dict['cbas_original'], x, wsize), label = 'CbAS guides', alpha = a, c = cbas_col, s = lw)

    for y in x[::2]:
        ax[i].axvspan(y, y + wsize, facecolor='grey', alpha=0.075)
    
    ax[i].set_xlabel('{} genome position'.format(renaming[virus]))
    ax[i].set_ylabel('Detected sequences (%)')

    ax[i].set_xlim(0, max(x))
    if(virus == 'SARSCOV2'):
        ax[i].set_ylim(90, 101)
    
fig.tight_layout()
fig.savefig(fig_dir + 'ALL_dots_coverage_graph_rev.pdf'.format(wsize), dpi = 500)
()
plt.close('all')

# Plots the same information as the above graph, but on a per-gene level
# Determines the percentage of sequence diversity detected within each gene by each design method
def within_pos(start, end, method):
    src = data_dict['DENV'][method]
    src = src[(src.start_pos > start) & (src.start_pos < end)]

    return np.max(src.perc_highly_active.values)*100 

# The reference sequence https://www.ncbi.nlm.nih.gov/nuccore/NC_001477.1 was used to determine the positions in the alignment
# at which the different genes start and end
# 3' UTR is last 500 nt
# 5' UTR ends at 100

denv_genes = ["5' UTR", 'C', 'prM', 'E', 'NS1', 'NS2A', 'NS2B', 'NS3', 'NS4A', 'NS4B', 'NS5',
             "3' UTR"]
denv_gene_start = [0, 342, 
                   624, 1079, 2670, 3726, 4383, 4773, 6652, 7106, 7854, 10556, max(data_dict['DENV']['consensus'].start_pos.values)]

dat = {x: [] for x in ['Gene', 'Algorithm', 'Detected']}

for method in ['consensus', 'adapt', 'wgan', 'evolutionary']:
    for i, gene in enumerate(denv_genes):
        dat['Algorithm'].append(method)
        dat['Gene'].append(gene)
        dat['Detected'].append(within_pos(denv_gene_start[i], denv_gene_start[i+1], method)) 

dat = pd.DataFrame(dat)
fig, ax = plt.subplots(figsize = (10, 4))
#lw = 50
x = np.arange(0, len(denv_genes))
ax.scatter(x+ .2,dat[dat.Algorithm == 'consensus'].Detected, label = 'Consensus guides', alpha = a, c = base_col, s = lw)
ax.scatter(x + .4,dat[dat.Algorithm == 'adapt'].Detected, label = 'ADAPT guides', alpha = a, c = adapt_col, s = lw)
ax.scatter(x + .6,dat[dat.Algorithm == 'wgan'].Detected, label = 'WGAN guides', alpha = a, c = wgan_col, s = lw)
ax.scatter(x + .8,dat[dat.Algorithm == 'evolutionary'].Detected, label = 'Evolutionary guides', alpha = a, c = evolutionary_col, s = lw)
ax.set_xticks(x+.5) 
ax.set_xticklabels(denv_genes)
for y in x[::2]:
    ax.axvspan(y, y + 1, facecolor='grey', alpha=0.075)
    
ax.set_xlim(0, len(x))
ax.set_xlabel('Dengue virus gene')
ax.set_ylabel('Detected sequences (%)')

fig.tight_layout()
fig.savefig(fig_dir + 'DENV_genes_coverage_graph.pdf'.format(wsize), dpi = 500)
plt.close('all')


#Plots the distribution of the fitness values across the different methods
a = 1
feature = 'full_model_score' #Previous versions of the method exported the guide fitness in a column called 'true model score'
# The feature being plotted is simply just the guide fitness
fig, ax = plt.subplots(nrows = len(data_dict.keys()), ncols = 1, figsize = (10, 4 * len(data_dict.keys()))) 

for i, virus in ['DENV']:
    df_dict = data_dict[virus]
    filterc = df_dict['adapt']['full_model_score'] + df_dict['consensus']['full_model_score'] > -5

    mod_data2 = []
    for algo in ['consensus', 'adapt', 'wgan', 'evolutionary']:
        mod_data2.append(df_dict[algo][feature][filterc].values - df_dict['consensus'][feature][filterc].values)
    
    sns.violinplot(data = mod_data2, palette = [base_col, adapt_col, wgan_col, evolutionary_col], inner = None, ax = ax[i])
        
    ax[i].set_ylabel('Relative fitness of probes')
    ax[i].set_xlabel('Algorithm')
    ax[i].set_title('{}'.format(virus))
    
    for index, color in enumerate(["#e9e9e9", "#cdcdcd", "#ccbccc", "#cee6d4"]):
        ax[i].boxplot(mod_data2[index], whis='range', positions=np.array([index]),
            showcaps=False,widths=0.04, patch_artist=True,
            boxprops=dict(color=color, facecolor= color),
            whiskerprops=dict(color=color, linewidth=1),
            medianprops=dict(color="black", linewidth=2 ))
    
    ax[i].set_xticks(np.arange(4)) 
    
    ax[i].set_xticklabels(['Consensus', 'MBC', 'WGAN-AM', 'Evolutionary'])
    
    if('DENV' in virus):
        denv_data = mod_data2
        

fig.tight_layout()
fig.savefig(fig_dir + 'ALL_fitness_violin_active_sites_filtered.pdf', dpi = 500)
plt.close('all')

# Running wilcoxon rank-sum tests to determine significance in the dengue virus case
print(stats.wilcoxon(mod_data2[0], mod_data2[1], alternative = 'less'))
print(stats.wilcoxon(mod_data2[0], mod_data2[2], alternative = 'less'))
print(stats.wilcoxon(mod_data2[0], mod_data2[3], alternative = 'less'))
print(stats.wilcoxon(mod_data2[1], mod_data2[2], alternative = 'less'))
print(stats.wilcoxon(mod_data2[1], mod_data2[3], alternative = 'less'))


# Plotting the minimum hamming distance from the guides generated by the different methods to all targets in the target_set
fig, ax = plt.subplots(nrows = len(data_dict.keys()), ncols = 1, figsize = (10, 4 * len(data_dict.keys()))) 

for i, virus in enumerate(data_dict.keys()):
    df_dict = data_dict[virus]
    filterc = df_dict['adapt']['full_model_score'] + df_dict['consensus']['full_model_score'] > -5
 
    plt.figure(figsize=(10,4))
    print(virus)
    if('wgan' in df_dict.keys()):
        ax[i].hist([list(df_dict['consensus'][filterc].hd_min_targets.values), 
                    list(df_dict['adapt'][filterc].hd_min_targets.values), 
                    list(df_dict['wgan'][filterc].hd_min_targets.values),
                      list(df_dict['evolutionary'][filterc].hd_min_targets.values),
                      list(df_dict['adalead'][filterc].hd_min_targets.values),
                      list(df_dict['cbas'][filterc].hd_min_targets.values)]
                      , color = [base_col, adapt_col, wgan_col, evolutionary_col, adalead_col, cbas_col],
              label = ['Consensus', 'MBC', 'WGAN-AM', 'Evolutionary', 'AdaLead', 'CbAS'], histtype ='bar', edgecolor='black', bins = np.arange(10)-0.5,
                  align = 'mid')

    ax[i].set_xlabel('Minimum Hamming distance between guide and all target sequences - {}'.format(renaming[virus]))
    ax[i].set_ylabel('Number of sites in genome')
    ax[i].legend(loc = 'upper right', bbox_to_anchor=(1, 1))
    ax[i].set_xlim(-.5, 6.5)

fig.tight_layout()
fig.savefig(fig_dir + 'ALL_histogram_hamming_cons_MIN_filtered.pdf', dpi = 400)
plt.close('all')


# Supplement Figs


# Plot comparing all methods performance
feature = 'full_model_score'
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 8)) 

sumn_dict = {x: [] for x in ['algo', 'virus', 'score']}
for idx, algo in enumerate(['consensus', 'adapt', 'wgan', 'genetic', 'adalead', 'cbas']):
    all_virus_dt = []
    for virus in data_dict.keys():
        df_dict = data_dict[virus]
        filterc = df_dict['adapt']['full_model_score'] + df_dict['consensus']['full_model_score'] > -5
        ssx = df_dict[algo][feature][filterc].values - df_dict['consensus'][feature][filterc].values
        
        for val in ssx:
            sumn_dict['algo'].append(algo)
            sumn_dict['virus'].append(virus)
            sumn_dict['score'].append(val)
        
color = [adapt_col, wgan_col, evolutionary_col, adalead_col, cbas_col]

# creating a dictionary composed of species as keys and colors as values
my_pal = {'Evolutionary': evolutionary_col, 'ADAPT': adapt_col, 'consensus': base_col,
          'WGAN-AM': wgan_col, 'AdaLead': adalead_col, 'cbas': cbas_col, 'adapt': adapt_col}
sumn_df = pd.DataFrame(sumn_dict)

# Getting unique categories and algorithms
viruses = sumn_df['virus'].unique()
algorithms = sumn_df['algo'].unique()

# Position tracking for ticks
pos_counter = 0
ticks = []  # This will keep track of where to put the labels
tick_labels = []  # This will keep track of what the labels will be

# Plot each group
for virus in viruses:
    group_positions = []  # To store positions for each group for central tick placement
    for algo in algorithms:
        # Filter data for each virus and algorithm
        subset = sumn_df[(sumn_df['virus'] == virus) & (sumn_df['algo'] == algo)]
        
        # Calculate percentiles and median
        median = subset['score'].median()
        perc_10 = np.percentile(subset['score'], 10)
        perc_90 = np.percentile(subset['score'], 90)

        # Plot 10th and 90th percentiles as lines
        ax.hlines(perc_10, pos_counter - 0.37, pos_counter + 0.37, color=my_pal[algo], linestyles='solid', lw=2)
        ax.hlines(perc_90, pos_counter - 0.37, pos_counter + 0.37, color=my_pal[algo], linestyles='solid', lw=2)
        
        # Add a vertical line connecting 10th and 90th percentiles
        ax.vlines(pos_counter, perc_10, perc_90, color=my_pal[algo], linestyles='solid', lw=2, label = algo)
        
        # Plot median as a dot
        ax.plot(pos_counter, median, 'o', color=my_pal[algo], markersize=7)

        # Store position
        group_positions.append(pos_counter)

        # Move to next position
        pos_counter += 1

    # Calculate middle position for the virus group tick mark
    if group_positions:
        middle_position = (group_positions[0] + group_positions[-1]) / 2
        ticks.append(middle_position)
        tick_labels.append(virus)

    # Add a little extra space after each virus grouping
    pos_counter += 1

# Remove the last extra increment for correct plot appearance
if pos_counter > 0:
    pos_counter -= 1

# Customizing plot
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, rotation=0, fontsize=15)
ax.set_ylabel('Relative fitness of probes', fontsize = 20)
ax.set_xlabel('Virus', fontsize = 20)
ax.legend()

#fig.tight_layout()
fig.savefig(fig_dir + f'ALL_summarized_fitness_by_virus.pdf', bbox_inches='tight')
plt.show()
plt.close('all')



# Plotting where there are mismatches between the DRAGON-designed guides and consensus for G PFS sites
mismatch_posw = []
mismatch_posg = []

for i, virus in enumerate(data_dict.keys()):
    df_dict = data_dict[virus]
    
    in1 = df_dict['wgan']['guide_sequence'][df_dict['wgan'].G_PFS == True].values
    in2 = df_dict['consensus']['guide_sequence'][df_dict['consensus'].G_PFS == True].values
    in3 = df_dict['evolutionary']['guide_sequence'][df_dict['evolutionary'].G_PFS == True].values

    for w, c, g in zip(in1, in2, in3):
        mismatch_posw.append(prep_seqs.mismatch_positions(c, w))
        mismatch_posg.append(prep_seqs.mismatch_positions(c, g))
        
plt.figure(figsize=(12,5))
plt.xticks(range(1, 29))
width = 0.44
space = 0.22
plt.bar(np.arange(1-space, 29 - space), np.mean(mismatch_posw, axis = 0), color = wgan_col, width = width, edgecolor = 'black', label = 'WGAN-AM')

plt.bar(np.arange(1+space, 29 +space), np.mean(mismatch_posg, axis = 0), color = evolutionary_col, width = width, edgecolor = 'black', label = 'Evolutionary')
plt.xlim(0.25, 29)
plt.ylabel('% of guides with mismatch against consensus')
plt.xlabel('Position in guide')
plt.legend()
fig.tight_layout()
plt.savefig(fig_dir + 'mismatch-bargraphs_GPFS_mult_mismatch_cons.pdf', dpi = 400)
plt.close('all')

# Plotting where there are mismatches between the MEA-designed guides and consensus for all genomic sites
mismatch_posw = []
mismatch_posg = []

for i, virus in enumerate(data_dict.keys()):
    df_dict = data_dict[virus]
    
    in1 = df_dict['wgan']['guide_sequence'].values
    in2 = df_dict['consensus']['guide_sequence'].values
    in3 = df_dict['evolutionary']['guide_sequence'].values
        
    for w, c, g in zip(in1, in2, in3):
        mismatch_posw.append(prep_seqs.mismatch_positions(c, w))
        mismatch_posg.append(prep_seqs.mismatch_positions(c, g))
        
        
plt.figure(figsize=(12,5))
plt.xticks(range(1, 29))
width = 0.44
space = 0.22
plt.bar(np.arange(1-space, 29 - space), np.mean(mismatch_posw, axis = 0), color = wgan_col, width = width, edgecolor = 'black', label = 'WGAN-AM')

plt.bar(np.arange(1+space, 29 +space), np.mean(mismatch_posg, axis = 0), color = evolutionary_col, width = width, edgecolor = 'black', label = 'Evolutionary')
plt.xlim(0.25, 29)
plt.ylabel('% of guides with mismatch against consensus')
plt.xlabel('Position in guide')
plt.legend()
fig.tight_layout()
plt.savefig(fig_dir + 'mismatch-bargraphs_mult_mismatch_cons.pdf', dpi = 400)
plt.close('all')

def non_G_PFS(df):
    return df[(df.G_PFS == False)]

# Processing Seqs

def extract_df(virus_name):
    # Import all the guide designs across the different viruses and sites
    
    curr_dir = './detection/gen_results/{}/{}/'.format(exp, virus_name)
    files_list = [x for x in os.listdir(curr_dir) if '.pkl' in x]

    methods = list(np.unique([x.split('_guides')[0] for x in files_list]))
    methods

    df_dict = {}
    all_results = []
    adapt_results = []
    cons_results = []

    for idx, method in enumerate(methods):
        df_list = []
        for file in [x for x in files_list if method == x.split('_')[0]]:
            xf = pd.read_pickle(curr_dir + file).reset_index(drop = True)
            method_xf = xf[xf.algo == method]
            max_pos = method_xf.full_model_score.idxmax()

            rowi = method_xf.loc[[max_pos]] #.to_frame().transpose()
            df_list.append(rowi)
            all_results.append(rowi)

            if(idx == 0):
                adapt_results.append(xf[xf.algo == 'ADAPT'].drop_duplicates())
                cons_results.append(xf[xf.algo == 'consensus'].drop_duplicates())


        results = pd.concat(df_list).sort_values(['virus_name', 'start_pos']).reset_index()
        df_dict[method] = pd.DataFrame(results)

    all_results = pd.concat(all_results).reset_index()
    df_dict['adapt'] = pd.concat(adapt_results).sort_values(['virus_name', 'start_pos']).reset_index()
    df_dict['consensus'] = pd.concat(cons_results).sort_values(['virus_name', 'start_pos']).reset_index()

    methods = df_dict.keys()

    return df_dict

def plot_nuc_compare(seq_labels, nuc_labels, diff_list, virus):
    # Plot nucleotide schematics for sites
    
    fig = plt.figure(figsize=(15*1,int(len(seq_labels)/2.6)))
    ax60 = sns.heatmap(np.array(diff_list), cmap=ListedColormap(['black', 'blue', 'red', 'green', 'yellow', 'purple']), cbar = False
                       , yticklabels = seq_labels, annot = np.array(nuc_labels), xticklabels = np.arange(1, 29), fmt = '', square = True,
                      annot_kws={"fontsize":16}) 
    
    ax60.tick_params(axis='x', which='major', labelsize=13)
    algo = seq_labels[0].split('.')[0]
    
    spaces = []
    for i, x in enumerate(seq_labels):
        if('guide' in x):
            spaces.append(i)
            
    for x in spaces:
        ax60.axhline(x, linewidth = 2.1, color = 'white')
    
    plt.xlabel('Position in guide')
    plt.savefig(fig_dir +'diff_heatmap_{}.pdf'.format(virus), bbox_inches = 'tight',
    pad_inches = 1)
    ()
    
    return fig


def compP(data_dict, curr_virus):
    # Compute P

    g = data_dict[curr_virus]['evolutionary'][data_dict[curr_virus]['evolutionary'].G_PFS == False].full_model_score
    c = data_dict[curr_virus]['consensus'][data_dict[curr_virus]['consensus'].G_PFS == False].full_model_score
    a = data_dict[curr_virus]['adapt'][data_dict[curr_virus]['adapt'].G_PFS == False].full_model_score
    w = data_dict[curr_virus]['wgan'][data_dict[curr_virus]['wgan'].G_PFS == False].full_model_score
            
    return (g+w - (c+a)).sort_values(), (g+w - (c+a)).reset_index(drop = True)
        
def find_substring(gene, probe):
    #Find the location of the guide binding region within the test target
    best_score = -np.inf
    best_pos = np.inf
    
    gene = list(gene)
    probe = list(probe)
    
    for pos in range(0, len(gene) - 28):
        seq = gene[pos:pos+28]
        score = sum(np.array(seq) == np.array(probe))

        if(score > best_score):
            best_score = score
            best_pos = pos
            
    return best_pos

guides_df = pd.DataFrame()
targets_df = pd.DataFrame()
s = data_dict.keys()

for vidx, curr_virus in enumerate(s):
    
    splist, plist = compP(data_dict, curr_virus)
    tidx = int(len(splist) * 0.75)
    bidx = int(len(splist) * 0.25)

    methods = ['genetic', 'wgan', 'adapt', 'consensus']
    rmethods = {'genetic': 'Evolutionary', 'wgan': 'WGAN-AM', 'adapt': 'MBC', 'consensus': 'Consensus'}

    aln = './detection/alignments/{}/{}.fasta.0'.format(exp, curr_virus)
    print('Processing virus w/ alignment at:')
    print(aln)

    seq_labels = []
    nuc_labels = []
    diff_list = []
    built_list = []
    
    # Test that find targets doesn't fail at positions at very beginning/end of genome
    for tidx in plist.index.to_list()[:50][::-1] + plist.index.to_list()[-50:][::-1]:
        start_pos = data_dict[curr_virus]['adapt'].iloc[tidx].start_pos
        test_targets, frac_covered, target_start, target_end = import_fasta.find_test_targets(start_pos-125, 
        start_pos+125 + 48, aln)
        
        if((len(test_targets) != len(frac_covered)) or (len(test_targets) < 1) or (len(test_targets[0]) < 100)):
            print(f'ERROR: finding targets failed')
            continue
            
    #For target sequences, the T7 promoter needs to be added at the front
    t7_promoter = "gaaatTAATACGACTCACTATAgggCACTATAGGGGCTCTAGCGACTTCTTTAAATAGTGGCTTAAAATAAC"
        
    # Preparing crRNA sequence
    direct_repeat_rna = "GAUUUAGACUACCCCAAAAACGAAGGGGACUAAAAC"
                
    ts = random.sample(splist.index.to_list()[tidx:], 2)
    bs = random.sample(splist.index.to_list()[:bidx], 1)
    built_list = ts + bs

    for qidx in built_list:
        
        # Extracting seqs
        gen_guide_row = data_dict[curr_virus]['adapt'].loc[qidx]
        start_pos = gen_guide_row.start_pos
                
        test_targets, frac_covered, target_start, target_end = import_fasta.find_test_targets(start_pos-125, 
        start_pos+125 + 48, aln)
        
        # Finds the location of the guide-binding region within the test target
        string_pos = find_substring(test_targets[0], gen_guide_row.guide_sequence)
        perf_test_targets = [target[string_pos-10: string_pos + 38] for target in test_targets]
        t_lens = [len(t) for t in perf_test_targets]
        
            
        # Construct target df
        z = 0
        for frac_div, target in zip(frac_covered, test_targets):
            targets_df = targets_df.append(pd.DataFrame({'seq_id': [f"{curr_virus}.{z}.{target_start}.{target_end}"],
                                          'target_seq': [target],
                                          'target_seq_to_order': [(t7_promoter + target).lower()],
                                          'frac_div': [frac_div],
                                           'target_len': [len(t7_promoter + target)]})).reset_index(drop = True)
            z += 1
        
        for method in methods:
            gen_guide_row = data_dict[curr_virus][method].loc[qidx]
            
            # Building data structure for plotting             
            nuc_labels.append(gen_guide_row['guide_sequence'])
            nuc_labels.extend([x[10:-10] for x in perf_test_targets])
           
            seq_labels.append(f"{rmethods[method]} guide")
            diff_list.append(prep_seqs.mismatch_nucleotides(gen_guide_row['guide_sequence'], gen_guide_row['guide_sequence']))
            
            z = 0
            for frac_div, target in zip(frac_covered, perf_test_targets):
                seq_labels.append(f"Target {z+1} ({round(frac_div* 100,1) }%)")
                diff_list.append(prep_seqs.mismatch_nucleotides(gen_guide_row['guide_sequence'], target[10:-10]))
                z += 1

            order_seq = prep_seqs.sub_t_u(direct_repeat_rna + prep_seqs.revcomp(gen_guide_row['guide_sequence']))

            # Construct guide df
            guides_df = guides_df.append(pd.DataFrame({'seq_id': [f"{method}.{curr_virus}.{gen_guide_row['start_pos']}"],
                                          'algo': [method],
                                          'guide_seq': [gen_guide_row['guide_sequence']],
                                          'guide_seq_to_order': [order_seq],
                                          'guide_seq_len': [len(order_seq)]})).reset_index(drop = True)

    nuc_labels = [list(x) for x in nuc_labels]
    
    print('Plotting Results')
    plot_nuc_compare(seq_labels, nuc_labels, diff_list, curr_virus)
    

targets_df.to_csv(fig_dir + f'{"_".join(s)}_targets.csv')
guides_df.to_csv(fig_dir + f'{"_".join(s)}_guides.csv')
# Generates the figures describing the performance of different methods in designing guides for the variant identification objective. 
# As described in the manuscript, these methods were run on a synthetic dataset of 100 pairs of on-target and off-target sequences, each differing by one nucleotide.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from badgers.models.cas13_diff import Cas13Diff 
from badgers.utils.cas13_landscape import Cas13Landscape
from badgers.utils import prepare_sequences as prep_seqs
from badgers.utils import import_fasta as import_fasta

sns.set_theme(font="Helvetica", style='ticks')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

from matplotlib import font_manager
font_manager.fontManager.addfont('/home/ubuntu/Helvetica.ttf')

plt.rcParams['text.color'] = 'black'
plt.rcParams['font.size'] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams["legend.frameon"] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.family'] = 'Helvetica'

plt.rcParams['font.family'] = 'Helvetica'

cmap = plt.cm.get_cmap('viridis')
gcmap = plt.cm.get_cmap('gray')
base_col = "#bbbbbb"
adapt_col =  "#555555" 
evolutionary_col = "#55ad70"
wgan_col =  (.35, .09, .35) 
adalead_col = '#5661E9'
cbas_col = '#D66B16'


## External Validation Results
results_dir = './discrimination/gen_results/ext_validation' 
base_dir = results_dir + '/'
wganfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and 'wgan' + '_' in x]
evolutionaryfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and 'evolutionary' + '_' in x]
cbasfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and 'cbas' + '_' in x]
adaleadfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and 'adalead' + '_' in x]


# This function generates the baseline guide with a synthetic misamtch
def gen_baseline_guide(target_set1_nt, target_set2_nt):
    
    baseline_guide_list_nt = list(target_set1_nt[0][10:-10])
    
    syn_mis_pos = 23
    
    # Randomly selecting a replacement that will lead to a TRUE mismatch, rather than G-U pairing
    if(baseline_guide_list_nt[syn_mis_pos] == 'G'):
        replace_nt = np.random.choice(list('CTG'.replace(baseline_guide_list_nt[syn_mis_pos], '')))
    elif(baseline_guide_list_nt[syn_mis_pos] == 'T'): 
        replace_nt = np.random.choice(list('ATG'.replace(baseline_guide_list_nt[syn_mis_pos], '')))
    else:
        replace_nt = np.random.choice(list('ATCG'.replace(baseline_guide_list_nt[syn_mis_pos], '')))

    baseline_guide_list_nt[syn_mis_pos] = replace_nt
    baseline_guide_nt = "".join(baseline_guide_list_nt)
    
    return baseline_guide_nt


# Compiling the results for both the wgan and evolutionary algorithm
# This reads in all the per-site files from both MEAs and builds a dataframe that can be used for analysis

cas13landscape = Cas13Landscape()
# These grid hyperparams don't matter b/c we are just using the model to predict activity
grid = {'c': 1.0, 'a' : 3.769183, 'k':  -3.833902, 'o': -2.134395, 't2w' : 2.973052} 

def compile_results(rfiles, top_n = 1):
    compiled_r = pd.read_pickle(results_dir + '/' + rfiles[0]).reset_index(drop = True)[0:0]

    costm = []
    snp_posm = []
    for rfile in rfiles:
        r_df = pd.read_pickle(results_dir + '/' + rfile)
        
        snp_pos_list = []
        for tidx, site in r_df.iterrows():
            target_set1 = site.target_set1_nt
            target_set2 = site.target_set2_nt
            try:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[0][i]][0] -10
            except Exception:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[1][i]][0] -10
                
            snp_pos_list.append(snp_pos)
            costm.append(site.fitness)
            snp_posm.append(snp_pos)
            
        r_df['snp_pos'] = snp_pos_list

        gen_df = r_df.dropna().reset_index(drop = True).sort_values(by = ['fitness'], ascending = False)
        indices = prep_seqs.edit_distance_index(gen_df, top_n)
        
        if(len(gen_df[gen_df.snp_pos == 25]) > 0):
            # The baseline guide places the SNP at position 25
            baseline = gen_df[gen_df.snp_pos == 25].iloc[0]
        else:
            print('MISSING')
            continue

        baseline_guide = gen_baseline_guide(baseline.target_set1_nt, baseline.target_set2_nt)
        
        for q in indices:
            
            gen_r = gen_df.iloc[q]
            
            # Computing perf of baseline guide
            model = Cas13Diff(cas13landscape, baseline.target_set1_nt, baseline.target_set2_nt, grid)
            evals = model._fitness_function([baseline_guide], output_type = 'eval')[0]
            baseline_perf = pd.Series({'pred_perf_target_set1': evals[0],'pred_perf_target_set2': evals[1]})
            
            # Computing perf of MEA guide
            model = Cas13Diff(cas13landscape, gen_r.target_set1_nt, gen_r.target_set2_nt, grid)
            evals = model._fitness_function([gen_r.guide_sequence], output_type = 'eval')[0]
            perf = pd.Series({'pred_perf_target_set1': evals[0],'pred_perf_target_set2': evals[1]})
            
            # Building the dataframe for the baseline guide
            gen_r['target_set1_name'] = baseline.seq_id.split('_synseq2')[0].split('synseq1_')[1]
            gen_r['target_set2_name'] = baseline.seq_id.split('_synseq2_')[1]
            gen_r['baseline_guide'] = baseline_guide
            gen_r['fitness'] = gen_r.fitness

            
            for key in perf.keys():
                gen_r[key] = perf[key]

            for key in perf.keys():
                gen_r['base_' + key] = baseline_perf[key]

            compiled_r = compiled_r.append(gen_r, ignore_index = True)

    results = compiled_r.sort_values(by = ['seq_id', 'fitness'], ascending = False).reset_index(drop = True).rename({'guide_sequence': 'generated_guide'}, axis =1)
    return results, snp_posm, costm

w_results, w_snp_pos_list, w_cost_list = compile_results(wganfiles)
g_results, g_snp_pos_list, g_cost_list = compile_results(evolutionaryfiles)
c_results, c_snp_pos_list, c_cost_list = compile_results(cbasfiles)
a_results, a_snp_pos_list, a_cost_list = compile_results(adaleadfiles)

# Main Figs 
fig_dir = './figures/discrimination/'

def flat(x):
    x = [[x[0]] if len(x) > 1 else x for x in x]
    return np.concatenate(x)

cmap = plt.cm.get_cmap('viridis')

# Plotting the distribution of on-target, off-target, and the difference on on/off target activity
# for guides designed by the different methods
lw = 5
wt_strategy = 'Baseline'
plt.figure(figsize=(10,4))
sns.kdeplot(flat(w_results.base_pred_perf_target_set1) - flat(w_results.base_pred_perf_target_set2), color = base_col, label = 'Baseline'.format(wt_strategy), alpha = 1, lw = lw)
sns.kdeplot(flat(w_results.pred_perf_target_set1) - flat(w_results.pred_perf_target_set2), color = wgan_col , label = 'WGAN-AM', alpha = 1, lw = lw)
sns.kdeplot(flat(g_results.pred_perf_target_set1) - flat(g_results.pred_perf_target_set2), color = evolutionary_col, label = 'Evolutionary', alpha = 1, lw = lw)
sns.kdeplot(flat(a_results.pred_perf_target_set1) - flat(a_results.pred_perf_target_set2), color = adalead_col, label = 'Adalead guides', alpha = 1, lw = lw)
sns.kdeplot(flat(c_results.pred_perf_target_set1) - flat(c_results.pred_perf_target_set2), color = cbas_col, label = 'CbAS guides', alpha = 1, lw = lw)
#plt.title('Difference in Predicted On-Target and Off-Target Activity'.format(wt_strategy), fontsize = 14)
plt.ylabel('Density')
plt.xlabel('Predicted on-target activity – predicted off-target activity')
plt.legend(loc = 'upper right', bbox_to_anchor=(1.1, 1.1))

plt.tight_layout()
plt.savefig(fig_dir + 'ext_kde_delta_activity.pdf', dpi = 200)

plt.close('all')

plt.figure(figsize=(5,4))
sns.kdeplot(flat(w_results.base_pred_perf_target_set1), color = base_col, label = 'Baseline'.format(wt_strategy), alpha = 1, lw = lw)
sns.kdeplot(flat(w_results.pred_perf_target_set1), color = wgan_col , label = 'WGAN-AM', alpha = 1, lw = lw)
sns.kdeplot(flat(g_results.pred_perf_target_set1), color = evolutionary_col, label = 'Evolutionary', alpha = 1, lw = lw)
sns.kdeplot(flat(a_results.pred_perf_target_set1), color = adalead_col, label = 'Adalead guides', alpha = 1, lw = lw)
sns.kdeplot(flat(c_results.pred_perf_target_set1), color = cbas_col, label = 'CbAS guides', alpha = 1, lw = lw)
#plt.title('Difference in Predicted On-Target and Off-Target Activity'.format(wt_strategy), fontsize = 14)
plt.ylabel('Density')
plt.xlabel('Predicted on-target activity')
#plt.legend(loc = 'upper right', bbox_to_anchor=(1.1, 1.1))
plt.xlim(-4, -0.3)
plt.ylim(0,3.2)
plt.yticks([0, 1, 2, 3])
plt.tight_layout()
plt.savefig(fig_dir + 'ext_kde_ontarget_activity.pdf', dpi = 200)

plt.close('all')


plt.figure(figsize=(5,4))
sns.kdeplot(flat(w_results.base_pred_perf_target_set2), color = base_col, label = 'Baseline'.format(wt_strategy), alpha = 1, lw = lw)
sns.kdeplot(flat(w_results.pred_perf_target_set2), color = wgan_col , label = 'WGAN-AM', alpha = 1, lw = lw)
sns.kdeplot(flat(g_results.pred_perf_target_set2), color = evolutionary_col, label = 'Evolutionary', alpha = 1, lw = lw)
sns.kdeplot(flat(a_results.pred_perf_target_set2), color = adalead_col, label = 'Adalead guides', alpha = 1, lw = lw)
sns.kdeplot(flat(c_results.pred_perf_target_set2), color = cbas_col, label = 'CbAS guides', alpha = 1, lw = lw)
#plt.title('Difference in Predicted On-Target and Off-Target Activity'.format(wt_strategy), fontsize = 14) 
plt.ylabel('Density')
plt.xlabel('Predicted off-target activity')
plt.xlim(-4, -0.3)
plt.ylim(0,3.2)
plt.yticks([0, 1, 2, 3])
plt.tight_layout()
plt.savefig(fig_dir + 'ext_kde_offtarget_activity.pdf', dpi = 200)
plt.close('all')

# Plots the location of where the SNP was placed in the guide for both methods
counts = []
snp_pos = np.array(w_results.snp_pos.values)
for x in range(28):
    counts.append(len(snp_pos[snp_pos == x]))
    
counts2 = []
snp_pos2 = np.array(g_results.snp_pos.values)
for x in range(28):
    counts2.append(len(snp_pos2[snp_pos2 == x]))
    
ind = np.arange(28)  # the x locations for the groups
width = .4     # the width of the bars

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
rects1 = ax.bar(1 + ind - width/2, counts, width, color=wgan_col, edgecolor='black', label = 'WGAN-AM')

womenMeans = (25, 32, 34, 20, 25)
womenStd =   (3, 5, 2, 3, 3)
rects2 = ax.bar(1 + ind+width/2, counts2, width, color= evolutionary_col, edgecolor='black', label = 'Evolutionary')
plt.xlim(1, 28)
plt.xlabel('Position of SNP on guides')
plt.ylabel('% of guides')
plt.xticks(1 + np.arange(28))
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig(fig_dir + 'snp_guide_positions.pdf', dpi = 200)


# Computing Mismatch Positions Relative to SNP
# The below code finds the position synthetic mismatches relative to the SNP 
# within each of the best-generated guides for each of the target sets.

def plt_mismatch(rfiles, top_n = 1):
    compiled_r = {'best_snp_pos' : [], 'gen_guide_nt' : [], 'target1_no_context_nt': []}

    for rfile in rfiles:
        r_df = pd.read_pickle(results_dir + '/' + rfile)
        
        snp_pos_list = []
        for tidx, site in r_df.iterrows():
            target_set1 = site.target_set1_nt
            target_set2 = site.target_set2_nt
            try:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[0][i]][0] -10
            except Exception:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[1][i]][0] -10
                
            snp_pos_list.append(snp_pos)

        r_df['snp_pos'] = snp_pos_list

        gen_df = r_df.dropna().reset_index(drop = True).sort_values(by = ['fitness'], ascending = False)
        indices = prep_seqs.edit_distance_index(gen_df, top_n)

        
        best_guide = gen_df.iloc[indices[0]]
        
        compiled_r['best_snp_pos'].append(best_guide.snp_pos)
        compiled_r['target1_no_context_nt'].append(best_guide.target_set1_nt[0][10:-10])
        compiled_r['gen_guide_nt'].append(best_guide.guide_sequence)
        
    return pd.DataFrame(compiled_r)

def plt_mismatch_freqs(bb, gan):
    data = np.zeros([len(gan), 28])
    count = 0

    nuc_data = np.zeros([len(gan), 28])
    for index in range(len(gan)):
        data[index] = prep_seqs.mismatch_positions(bb[index], gan[index])
        
        nuc_data[index] = prep_seqs.mismatch_nucleotides(bb[index], gan[index])

        if((data[index] == np.zeros(28)).all()):
            count += 1

    print('Length of Dataframe: ' + str(len(data)))
    print('Number of WT Guides Generated: ' + str(count))

    nuc_heat = pd.DataFrame.from_dict({'A' : np.count_nonzero(nuc_data == 1, axis = 0), 'C' : np.count_nonzero(nuc_data == 2, axis = 0), 'G': np.count_nonzero(nuc_data == 3, axis = 0)
    , 'T':np.count_nonzero(nuc_data == 4, axis = 0)}, orient = 'index')

    nuc_heat = nuc_heat/len(gan)
    
    nuc_heat_sum = nuc_heat.copy()
    
    nuc_heat_sum.loc['Sum'] = nuc_heat_sum.loc['A'] + nuc_heat_sum.loc['G'] + nuc_heat_sum.loc['T'] + nuc_heat_sum.loc['C']

    return nuc_data, nuc_heat, nuc_heat_sum
            
df = plt_mismatch(wganfiles)
z = df['best_snp_pos'].unique()
r = []
for i in z:
    pld = df[df['best_snp_pos'] == i].reset_index()
    q = plt_mismatch_freqs(pld['target1_no_context_nt'], pld['gen_guide_nt'])[2].iloc[4:].iloc[0][i-4:i+5] * len(pld)
    print('\n')
    r.append(q.to_list())
    
print(np.sum(r, axis = 0))
pltfreqs1 = np.sum(r, axis = 0)

df = plt_mismatch(evolutionaryfiles)
z = df['best_snp_pos'].unique()
r = []
for i in z:
    pld = df[df['best_snp_pos'] == i].reset_index()
    q = plt_mismatch_freqs(pld['target1_no_context_nt'], pld['gen_guide_nt'])[2].iloc[4:].iloc[0][i-4:i+5] * len(pld)
    print('\n')
    r.append(q.to_list())

pltfreqs2 = np.sum(r, axis = 0)



# Plots the frequency and position of the synthetic mismatches introduced by both methods
# The lists of frequencies come from the above analysis
ind = np.arange(-4, 5)  # the x locations for the groups
width = .5     # the width of the bars

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
rects1 = ax.scatter([-4, -3, -2, -1, 1, 2, 3, 4], np.array(pltfreqs1), color=wgan_col, label = 'WGAN-AM', s = 65)

rects2 = ax.scatter([-4, -3, -2, -1, 1, 2, 3, 4] , np.array(pltfreqs2), color= evolutionary_col,label = 'Evolutionary', s = 65)

plt.xlabel('Position on guide relative to SNP')
plt.ylabel('% of guides with mismatch')
plt.xticks(np.arange(-4, 5))
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir + 'mismatch_positions.pdf', dpi = 200)

sum(pltfreqs1)
sum(pltfreqs2)

# Exploratory Figures
# The below figures plot the fitness of the MEA-generated guides across the different SNP placements. 

# Create a dataframe that summarizes the fitness of the guides by their SNP placement
def compile_results_costs(rfiles):
    compiled_r = pd.read_pickle(results_dir + '/' + rfiles[0]).reset_index(drop = True)[0:0]
    
    final = pd.DataFrame()
    
    for rfile in rfiles:
        snp_pos_list = []
        r_df = pd.read_pickle(results_dir + '/' + rfile)
        
        for tidx, site in r_df.iterrows():
            target_set1 = site.target_set1_nt
            target_set2 = site.target_set2_nt
            try:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[0][i]][0] -10
            except Exception:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[1][i]][0] -10
                
          
            snp_pos_list.append(snp_pos)
                    
        r_df['snp_pos'] = snp_pos_list

        largest = r_df.groupby(['snp_pos']).apply(lambda group: group.nlargest(1, columns='fitness')).reset_index(drop=True)

        final = final.append(largest)

    return final.fitness, final.snp_pos

w_cost_list, w_snp_pos_list = compile_results(wganfiles)
g_cost_list, g_snp_pos_list = compile_results(evolutionaryfiles)


plt.figure(figsize=(10,4))
ax16 = sns.violinplot(x=np.array(w_snp_pos_list)+1, y=w_cost_list, palette = 'turbo', scale='width')
ax16.set_xlabel('Position of SNP in guide')
ax16.set_ylabel('Fitness')
plt.tight_layout()
plt.ylim(-1.5, 0)
plt.savefig(fig_dir + 'wgan_SNP_pos_vs_cost.pdf', dpi = 200)

plt.figure(figsize=(10,4))
ax16 = sns.violinplot(x=np.array(g_snp_pos_list)+1, y=g_cost_list, palette = 'turbo', scale='width')

ax16.set_xlabel('Position of SNP in guide')
ax16.set_ylabel('Fitness')
plt.tight_layout()

plt.savefig(fig_dir + 'evolutionary_SNP_pos_vs_cost.pdf', dpi = 200)


# Processing lineage identification files

def save_df(input_df, method, fitness_col):
    # Saves diff files
    
    guides_df = pd.DataFrame()
    targets_df = pd.DataFrame()

    t7_promoter = "gaaatTAATACGACTCACTATAgggCACTATAGGGGCTCTAGCGACTTCTTTAAATAGTGGCTTAAAATAAC"
    direct_repeat_rna = "GAUUUAGACUACCCCAAAAACGAAGGGGACUAAAAC"

    for index, row in input_df.iterrows():
        #Adding direct repeat and features for crRNA 
        guide = row.guide_sequence
        order_seq = prep_seqs.sub_t_u(direct_repeat_rna + prep_seqs.revcomp(guide))
        guides_df = guides_df.append(pd.DataFrame({'seq_id': [f"{method}.{row.target1_name}.{int(row.start_pos)}"],
                                      'fitness': [row[fitness_col]],
                                      'guide_seq': [guide],
                                      'guide_seq_to_order': [order_seq],
                                      'guide_seq_len': [len(order_seq)]})).reset_index(drop = True)

    guides_df.to_csv(f'./figures/discrimination/{rname}/{method}_guides.csv')

def find_best_amplicon(input_df, fitness_col):
    # It is necessary to find an amplicon in which all of the baseline guides could bind
    # Otherwise, these guides would not be cumbersome for diagnostic purposes and would require one primer set per lineage-specific guide

    mean_perf_list = []
    window_df_list = []

    # Iterate through all possible ~2000nt amplicons and identify the amplicon with the best
    # mean performance
    for pos in range(0, max(input_df.start_pos) - 2000, 50):
        current_window = input_df[(input_df.start_pos > pos) & (input_df.start_pos < pos + 2000)]
        
        if(len(current_window) < 1):
            continue
            
        max_window = current_window.groupby('target1_name').apply(lambda group: group.nlargest(1, columns=fitness_col)).reset_index(drop=True)

        mean_perf = np.mean(max_window.fitness_col)
        mean_perf_list.append(mean_perf)
        window_df_list.append(max_window)
        
    return mean_perf_list, window_df_list

rname = 'full-genome-DENV'
results_dir = f'./discrimination/gen_results/full-genome-DENV/results_by_site/'
base_dir = results_dir + '/'

# Selected algorithm to import results from
salgo = 'wgan'
rfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and salgo + '_' in x]

results = pd.DataFrame()
for file in rfiles:
    try:
        results = results.append(pd.read_pickle(results_dir + file))
    except Exception:
        continue
        
seq_grouped = results.groupby('seq_id').apply(lambda group: group.nlargest(1, columns='fitness')).reset_index(drop=True)

# Finding the best amplicon and saving
mean_perf_list, window_df_list = find_best_amplicon(seq_grouped)
save_df(window_df_list[np.argmax(mean_perf_list)], salgo, 'fitness') 

# Baseline method for designing guides that do lineage discrimination
seqs_df = pd.read_pickle('./discrimination/alignments/full-genome-DENV/seqs_df.pkl') 
cons_nt = 10
base_df = pd.DataFrame()
for target1_name, group in seqs_df.groupby('target1_name'):
    print(target1_name)
    
    hd_list = []
    index_list = []
    
    all_t2 = group.target_set2_nt
    cons_t2 = [prep_seqs.consensus_seq([x[10:-10] for x in t2], nt = True) for t2 in all_t2]
    flat_list = [item for sublist in all_t2 for item in sublist]
    
    for index, row in group.iterrows():
        target_set1_nt = [x[cons_nt:-cons_nt] for x in row.target_set1_nt]
        target_set2_nt = [x[cons_nt:-cons_nt] for x in row.target_set2_nt]
        
        # If the average nucleotide identity of the on-target or off-target sets 
        # is less than a threshold, then don't consider this site
        if(prep_seqs.compute_ani(target_set1_nt) < 0.9 or prep_seqs.compute_ani(target_set2_nt) < 0.9):
            continue
            
        # Computing the consensus at this site and computing its minimum hamming dist
        # to the off-target set
        t1_cons = prep_seqs.consensus_seq(target_set1_nt, nt = True)
        hd = min([prep_seqs.hamming_dist(t1_cons, t2) for t2 in cons_t2])
        
        # If the consensus is identical to a sequence in the off-target set, then ignore this site
        if(hd<1):
            continue
                    
        row_save = row.copy()
        row_save['hd'] = hd
        base_df = base_df.append(row_save)

base_mean_perf_list, base_window_df_list = find_best_amplicon(base_df)

# Construct the baseline DF and save the guides
base_results_df = base_window_df_list[np.argmax(base_mean_perf_list)]
base_results_df['guide_sequence'] = [x[0][cons_nt:-cons_nt] for x in base_results_df.target_set1_nt]
save_df(base_results_df, 'baseline', 'hd')


# Processing SNP files
# The order_compile_results is used to read in all the files and create a dataframe with guides for the SNP tasks

results_dir = './discrimination/gen_results/SNP-design/results_by_site/' 
wganfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and 'wgan' + '_' in x]
evolutionaryfiles = [x for x in os.listdir(results_dir) if '.pkl' in x and 'evolutionary' + '_' in x]

# This is distinct from the old compile_results b/c there is a new row for the baseline df
def order_compile_results(rfiles, top_n = 3):
    compiled_r = pd.read_pickle(results_dir + '/' + rfiles[0]).reset_index(drop = True)[0:0]

    for rfile in rfiles:
        print('Reading in {}'.format(rfile))
        r_df = pd.read_pickle(results_dir + '/' + rfile)
        snp_pos_list = []
        
        for tidx, site in r_df.iterrows():
            target_set1 = site.target_set1_nt
            target_set2 = site.target_set2_nt
            try:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[0][i]][0] -10
            except Exception:
                snp_pos = [i for i in range(len(target_set1[0])) if target_set1[0][i] != target_set2[1][i]][0] -10
                
            snp_pos_list.append(snp_pos)
            
        r_df['snp_pos'] = snp_pos_list

        gen_df = r_df.dropna().reset_index(drop = True).sort_values(by = ['fitness'], ascending = False)
        indices = prep_seqs.edit_distance_index(gen_df, top_n)

        if(len(gen_df[gen_df.snp_pos == 25]) < 1):
            print('SITE MISSING')
            continue
        
        # The baseline guide places the SNP at position 25
        baseline = gen_df[gen_df.snp_pos == 25].iloc[0]
        baseline_guide = gen_baseline_guide(baseline.target_set1_nt, baseline.target_set2_nt)
        
        for q in indices:
            
            gen_r = gen_df.iloc[q]
            base_r = baseline

             # Computing perf of baseline guide
            model = Cas13Diff(cas13landscape, baseline.target_set1_nt, baseline.target_set2_nt, grid)
            evals = model._fitness_function([baseline_guide], output_type = 'eval')[0]
            baseline_perf = pd.Series({'pred_perf_target_set1': evals[0],'pred_perf_target_set2': evals[1]})
            
            # Computing perf of MEA guide
            model = Cas13Diff(cas13landscape, gen_r.target_set1_nt, gen_r.target_set2_nt, grid)
            evals = model._fitness_function([gen_r.guide_sequence], output_type = 'eval')[0]
            perf = pd.Series({'pred_perf_target_set1': evals[0],'pred_perf_target_set2': evals[1]})
        
            
            # Summarizing the Cas13 guide sequences for both the baseline and MEA guides
            direct_repeat_rna = "GAUUUAGACUACCCCAAAAACGAAGGGGACUAAAAC"
            
            gen_r['target_set1_name'] = baseline.seq_id.split('_vs_')[0]
            gen_r['target_set2_name'] = baseline.seq_id.split('_vs_')[1]
            gen_r['guide_seq_to_order'] = prep_seqs.sub_t_u(direct_repeat_rna + prep_seqs.revcomp(gen_r.guide_sequence))
            gen_r['guide_seq_len'] = len(gen_r['guide_seq_to_order'])
            
            for key in perf.keys():
                gen_r[key] = perf[key]
            
            for key in perf.keys():
                base_r[key] = baseline_perf[key]
            
            base_r['algo'] = 'synmismatch'
            base_r['snp_pos'] = 25
            base_r['guide_sequence'] = baseline_guide
            base_r['target_set1_name'] = baseline.seq_id.split('_vs_')[0]
            base_r['target_set2_name'] = baseline.seq_id.split('_vs_')[1]
            base_r['guide_seq_to_order'] = prep_seqs.sub_t_u(direct_repeat_rna + prep_seqs.revcomp(baseline_guide))
            base_r['guide_seq_len'] = len(base_r['guide_seq_to_order'])

            compiled_r = compiled_r.append(gen_r, ignore_index = True)
        compiled_r = compiled_r.append(base_r, ignore_index = True)

    results = compiled_r.sort_values(by = ['seq_id'], ascending = False).reset_index(drop = True).rename({'guide_sequence': 'guide_seq'}, axis =1)
    return results

ws_results = order_compile_results(wganfiles)
gs_results = order_compile_results(evolutionaryfiles)   
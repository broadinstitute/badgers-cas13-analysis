"""CbAS explorer
This is a modified version of FLEXS implementation of CbAS
Citation to original code: https://github.com/samsinai/FLEXS/tree/master

We encountered difficulties training the original VAE implemented in FLEXS,
so instead used the code from the VAE implemented in the original CbAS publication"""

import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import flexs
from badgers.utils import prepare_sequences as prep_seqs

import os
import time
 
def random_dna(length):
    return ''.join(random.choice('CGTA') for i in range(length))

class CbASOriginal(flexs.Explorer):
    """CbAS explorer."""

    def __init__(
        self,
        model: flexs.Model,
        rounds: int,
        starting_sequence: str,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        alphabet: str,
        latent_dim: int = 20,
        per_train_epochs: int = 10,
        algo: str = "cbas",
        Q: float = 0.7,
        cycle_batch_size: int = 100,
        mutation_rate: float = .1/28,
        log_file: Optional[str] = None,
    ):
        """
        Explorer which implements Conditioning by Adaptive Sampling (CbAS)
        and DbAS.

        Paper: https://arxiv.org/pdf/1901.10060.pdf 
        """
        name = f"{algo}_Q={Q}"
        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size, 
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        if algo not in ["cbas", "dbas"]:
            raise ValueError("`algo` must be one of 'cbas' or 'dbas'")
        self.algo = algo

        np.random.seed(seed=int(time.time()))
        filepath = f'/home/ubuntu/cbas_tmp_files3/{np.random.random()}-{random_dna(5)}' 
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        hyperparams = pd.Series({'cycle_batch_size': cycle_batch_size, 
                                 'per_train_epochs': per_train_epochs,
                                 'latent_dim': latent_dim})  
        
        print(hyperparams)
        hyperparams.to_pickle(f'{filepath}/hyperparams.pkl')  

        self.filepath = filepath
        self.latent_dim = latent_dim 
        self.per_train_epochs = per_train_epochs
        self.alphabet = alphabet
        self.Q = Q  
        self.cycle_batch_size = cycle_batch_size
        self.mutation_rate = mutation_rate

    def _extend_samples(self, samples, weights):
        # generate random seqs around the input seq if the sample size is too small
        
        samples = list(samples)
        weights = list(weights)
        sequences = set(samples)

        print('Extending samples...')
        print(len(sequences))
        print(len(samples))
        while len(sequences) < self.cycle_batch_size:
            
            sample = random.choice(samples)
            sample = prep_seqs.generate_random_mutant(
                sample, self.mutation_rate, alphabet=self.alphabet
            )

            if sample not in sequences:
                samples.append(sample)
                weights.append(1)
                sequences.add(sample)

        return np.array(samples), np.array(weights)
    
    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath) 

        """Propose top `sequences_batch_size` sequences for evaluation."""
        # If we are on the first round, our model has no data yet, so the
        # best policy is to propose random sequences in a small neighborhood.
        last_round = measured_sequences_data["round"].max()
        if last_round == 0:
            print('Creating starting set of sequences')
            sequences = set()
            while len(sequences) < self.sequences_batch_size:
                sequences.add(
                    prep_seqs.generate_random_mutant(
                        self.starting_sequence,
                        2 / len(self.starting_sequence),
                        self.alphabet,
                    )
                )

            sequences = np.array(list(sequences))
            print(len(sequences))
            print(np.mean([prep_seqs.hamming_dist(self.starting_sequence, target) for target in sequences]))
            return sequences, self.model.get_fitness(sequences)
        
        last_round_sequences = measured_sequences_data[
            measured_sequences_data["round"] == last_round
        ]

        gamma = np.percentile(last_round_sequences["true_score"], 100 * self.Q)
        initial_batch = last_round_sequences["sequence"][
            last_round_sequences["true_score"] >= gamma
        ].to_numpy()
        initial_weights = np.ones(len(initial_batch))

        initial_batch, initial_weights = self._extend_samples(
            initial_batch, initial_weights
        )

        M = len(initial_batch[0])
        N = len(initial_batch)
        Xtrain = np.zeros((N, M, 4))

        for i in range(len(initial_batch)):
            Xtrain[i] = prep_seqs.one_hot_encode(initial_batch[i])
        
        # Train VAE_0
        np.save(f'{self.filepath}/X_train.npy', Xtrain)  
        os.system(f"python /home/ubuntu/troubleshoot-cbas/cbas_run_funcs.py 1 {self.filepath}")  

        sequences = {}
        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_batch:

            # Get samples
            os.system(f"python /home/ubuntu/troubleshoot-cbas/cbas_run_funcs.py 2 {self.filepath}")  
            Xt = np.load(f"{self.filepath}/proposals_onehot.npy")
            Xt_nuc = [prep_seqs.convert_to_nt(x) for x in Xt]
            
            # Score samples
            scores = self.model.get_fitness(Xt_nuc)

            gamma = max(np.percentile(scores, self.Q * 100), gamma)
            time.sleep(np.random.choice(np.arange(0.01, 2)))
            X0_p = np.load(f"{self.filepath}/X0_p.npy")
            Xt_p = np.load(f"{self.filepath}/Xt_p.npy") 
        
            log_pxt = np.sum(np.log(Xt_p) * Xt, axis=(1, 2))
            log_px0 = np.sum(np.log(X0_p) * Xt, axis=(1, 2))
            weights = np.exp(log_px0-log_pxt)
            weights = np.nan_to_num(weights)
            weights[scores < gamma] = 0
             
            # Train VAE   
            np.save(f"{self.filepath}/X_cont_train.npy", Xt)
            np.save(f"{self.filepath}/weights.npy", weights)

            os.system(f"python /home/ubuntu/troubleshoot-cbas/cbas_run_funcs.py 3 {self.filepath}")  
            sequences.update(zip(Xt_nuc, scores))

        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]
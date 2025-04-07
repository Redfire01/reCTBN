//! Module containing Hiton's algorithm

use crate::params::Params;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::ParallelExtend;
use std::collections::{BTreeSet, HashMap};
use std::mem;
use std::usize;

use super::hypothesis_test::*;
use crate::parameter_learning::ParameterLearning;
use crate::process;
use crate::structure_learning::StructuralLearningAlgorithm;
use crate::tools::Dataset;
use crate::structure_learning::constraint_based_algorithm::Cache;
use crate::params::ParamsTrait;


pub struct Hiton <P: ParameterLearning>{
    parameter_learning: P,
    Ftest: F,
    Chi2test: ChiSquare,
}

impl<P: ParameterLearning> Hiton <P> {
    pub fn new(parameter_learning: P, Ftest: F, Chi2test: ChiSquare) -> Hiton<P> {
        Hiton {
            parameter_learning,
            Ftest,
            Chi2test,
        }
    }
}

impl<P: ParameterLearning> StructuralLearningAlgorithm for Hiton<P> { // Structural learning algorithm for hiton??? o altro??
    fn fit_transform<T>(&self, net: T, dataset: &Dataset) -> T
    where
        T: process::NetworkProcess,
    {
        if net.get_number_of_nodes() != dataset.get_trajectories()[0].get_events().shape()[1] {
            panic!("Dataset and Network must have the same number of variables.")
        }

        let mut net = net;

        net.initialize_adj_matrix();
        
        let mut learned_parent_sets: Vec<(usize, BTreeSet::<usize>)> = vec![];
        let mut separation_set = BTreeSet::<usize>::new();

        learned_parent_sets.par_extend(net.get_node_indices().into_par_iter().map(|child_node| {
            let mut cache = Cache::new(&self.parameter_learning);
            let mut candidate_parent_set: Vec<(usize, f64)> = Vec::new();
            // Da qui cambia 
            let mut all_parents: BTreeSet<usize> = net
                .get_node_indices()
                .into_iter()
                .filter(|x| x != &child_node)
                .collect();
            //println!("child_node: {}, all_nodes: {:?}", child_node, all_parents);
            //1:  Forall nodes in the network it evaluate the independence test for a separation set = empty set
            for parent in all_parents.iter(){
                let n_tests: usize = net.get_node(child_node.clone()).get_reserved_space_as_parent() 
                            * net.get_node(parent.clone()).get_reserved_space_as_parent();

                let p_value_f = self.Ftest.compute_pvalues(
                                    &net,
                                    child_node,
                                    *parent,
                                    &separation_set,
                                    dataset,
                                    &mut cache,
                                );
                let p_value_chi = self.Chi2test.compute_pvalues(
                                    &net,
                                    child_node,
                                    *parent,
                                    &separation_set,
                                    dataset,
                                    &mut cache,
                                );
                // F s < limsx || s > limdx --> dependent || Chi-squared < 1-self-alpha independent --> > 1-self.alpha --> dependent
                //2: IF: the nodes are not independent they are added to candidate_parent_set ELSE: not considered
                //   eliminate all the nodes independent from T.
                if !(p_value_f.2 && p_value_chi.2) {  
                    // lista di tuple (parent, mean_p_value)
                    candidate_parent_set.push((*parent, ((p_value_f.0 + p_value_chi.0)/(2.0 * n_tests as f64))));
                }           
                
            }
            //3: Sorting of the candidate parent set

            candidate_parent_set.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            //4: Initialization of the currentPS ... 
            let mut currentPC_keys = BTreeSet::<usize>::new();
            let mut candidate_parent_set_keys: BTreeSet<usize> = candidate_parent_set.iter().map(|(key, _)| *key).collect();
            //5: Foreach node in candidate parent set... 
            for X in candidate_parent_set_keys.clone().iter() { 

                //6: IF T (child_node) not indep from X | separation set \subseteq currentPS

                let max_size = candidate_parent_set_keys.len();

                for separation_set_size in 1..=max_size {  // <-- Ciclo esterno che varia `separation_set_size`
                    for separation_set in candidate_parent_set_keys
                        .clone()
                        .iter()
                        .filter(|&&x| x != *X)
                        .copied() // Copia i valori per ottenere un iteratore su `usize`
                        .combinations(separation_set_size) 
                    {
                        let separation_set: BTreeSet<usize> = separation_set.into_iter().collect();
                        let mut sep_set_size = 1;

                        for i in separation_set.iter(){
                            sep_set_size *= net.get_node(i.clone()).get_reserved_space_as_parent();
                        }

                        let n_tests: usize = net.get_node(child_node.clone()).get_reserved_space_as_parent() 
                                    * net.get_node(X.clone()).get_reserved_space_as_parent() 
                                    * sep_set_size;

                        let p_value_f = self.Ftest.call( // False if s < lim_sx (alpha/2) or s > lim_dx (1 - (alpha/2)) => M1 and M2 dependent
                                            &net,                   // True if lim_sx < s < lim_dx => M1 and M2 independent
                                            child_node,
                                            *X,
                                            &separation_set,
                                            dataset,
                                            &mut cache,
                                        );
                        let p_value_chi = self.Chi2test.call(        // True if < 1-self.alpha => M1 and M2 independent
                                            &net,                               // usare call.
                                            child_node,
                                            *X,
                                            &separation_set,
                                            dataset,
                                            &mut cache,
                                        );

                        // 7: add X to currentPS
                        if p_value_chi && p_value_f {  
                            candidate_parent_set_keys.remove(&X);
                        }

                    }
                }

            }

                (child_node, candidate_parent_set_keys)
        }));
        

        for (child_node, currentPC) in learned_parent_sets {
            let mut i = 0;
            for parent_node in currentPC.iter() {
                i = i+1;
                println!("child node: {}, parent node: {}, it = {}", child_node, parent_node, i);
                net.add_edge(*parent_node, child_node);
            }
        }
        
        net
    }   
}


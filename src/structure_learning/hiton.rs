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
        
        let mut learned_parent_sets: Vec<(usize, Vec<(usize, f64)>)> = vec![];

        learned_parent_sets.par_extend(net.get_node_indices().into_par_iter().map(|child_node| {
            let mut cache = Cache::new(&self.parameter_learning);
            let mut candidate_parent_set: BTreeSet<usize> = net
                .get_node_indices()
                .into_iter()
                .filter(|x| x != &child_node)
                .collect();

            // Da qui cambia 

            let mut currentPC: Vec<(usize, f64)> = Vec::new();
            //let currentPC_keys: BTreeSet<usize> = BTreeSet::new();
            let currentPC_keys = BTreeSet::<usize>::new();
            let mut separation_set = BTreeSet::<usize>::new(); // 0 at the moment --> then we'll consider also the possible separation sets.
            for parent in candidate_parent_set
                .difference(&currentPC_keys){ 
                /* 
                println!("parent = {}", parent);
                
                for i in currentPC_keys.iter(){
                    print!("Parent {}, Current PC = {}, dim = {}", parent, i, currentPC.len());
                }
                println!();*/
                
                let n_tests = net.get_node(child_node.clone()).get_reserved_space_as_parent() 
                                * net.get_node(parent.clone()).get_reserved_space_as_parent() ;
    //            println!("numero test = {}", n_tests);
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
                if !(p_value_f.2 && p_value_chi.2) {  // maximizes association
                    // lista di tuple (parent, mean_p_value)
                    currentPC.push((*parent, ((p_value_f.0 + p_value_chi.0)/(2.0 * n_tests as f64))));
                } 
                // Now let's evaluate if there is a variable X and a subset of CurrentPC s.t. X ind T | S
                // Separato?? più efficiente, non mi serve tenerlo unito(?)
                let mut separation_set_size = 0;
                let currentPC_keys: BTreeSet<usize> = currentPC.iter().map(|(key, _)| *key).collect();
                for X in currentPC_keys.clone().iter() {
                    for separation_set in currentPC_keys.clone()
                        .iter()
                        .filter(|x| x != &X)
                        .map(|x| *x)
                        .combinations(separation_set_size){
                        let separation_set:BTreeSet::<usize> = separation_set.into_iter().collect();
                        // cardinalità della variabile target X cardinalità del separation set  X cardinalità del candidato genitore
                        /*let sep_set_size: usize = separation_set.iter()
                                                    .map(|x| net.get_node(x.clone()).get_reserved_space_as_parent())
                                                    .product();*/
                        let mut sep_set_size = 1;
                        for i in separation_set.iter(){
    //                        println!("nodo = {}, reserved_space = {}", i, net.get_node(i.clone()).get_reserved_space_as_parent());
                            sep_set_size *= net.get_node(i.clone()).get_reserved_space_as_parent();
                        }
                        let n_tests: usize = net.get_node(child_node.clone()).get_reserved_space_as_parent() 
                                    * net.get_node(X.clone()).get_reserved_space_as_parent() 
                                    * sep_set_size;
                        /*
                        println!("separation set size = {}", sep_set_size);

                        println!("p_value_f = {}, p_value_chi = {}, n_test = {}", p_value_f.0 / (n_tests as f64), p_value_chi.0 / (n_tests as f64), n_tests);
                        */
                        let p_value_f = self.Ftest.compute_pvalues( // False if s < lim_sx (alpha/2) or s > lim_dx (1 - (alpha/2)) => M1 and M2 dependent
                                            &net,                   // True if lim_sx < s < lim_dx => M1 and M2 independent
                                            child_node,
                                            *X,
                                            &separation_set,
                                            dataset,
                                            &mut cache,
                                        );
                        let p_value_chi = self.Chi2test.compute_pvalues(        // True if < 1-self.alpha => M1 and M2 independent
                                            &net,
                                            child_node,
                                            *X,
                                            &separation_set,
                                            dataset,
                                            &mut cache,
                                        );

                        

                        if p_value_chi.2 && p_value_f.2 {  
                            currentPC.retain(|x| x.0 != *X);
                        }
                        separation_set_size += 1;
                    }
                    

                }
                let currentPC_keys: BTreeSet<usize> = currentPC.iter().map(|(key, _)| *key).collect();
            }

                (child_node, currentPC)
        }));
        for (child_node, mut currentPC) in learned_parent_sets {

            currentPC.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        }
        
/*
        for (child_node, currentPC) in learned_parent_sets {
            for parent_node in currentPC.iter() {
                net.add_edge(*parent_node, child_node);
            }
        }*/
/*        
        for i in net.get_node_indices() {
            for j in net.get_parent_set(i).iter() {
                println!("i = {}, j = {}", i, j);
            }
            
        }*/
        
        net
    }   
}


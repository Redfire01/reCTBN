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
use crate::structure_learning::constraint_based_algorithm::{Cache};
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
        
        let mut learned_parent_sets: Vec<(usize, Vec::<usize>)> = vec![];
        let mut separation_set = BTreeSet::<usize>::new();

        learned_parent_sets.par_extend(net.get_node_indices().into_par_iter().map(|child_node| {
            let mut cache = Cache::new(&self.parameter_learning);
            let mut candidate_parent_set: Vec<(usize, f64)> = Vec::new();
            let mut all_parents: BTreeSet<usize> = net
                .get_node_indices()
                .into_iter()
                .filter(|x| x != &child_node)
                .collect();
            //println!("child_node: {}, all_nodes: {:?}", child_node, all_parents);
            //1:  Forall nodes in the network it evaluate the independence test for a separation set = empty set
            for parent in all_parents.iter(){
                /*let n_tests: usize = net.get_node(child_node.clone()).get_reserved_space_as_parent() 
                            * net.get_node(parent.clone()).get_reserved_space_as_parent();*/

                let p_value_f = self.Ftest.compute_pvalues(
                                    &net,
                                    child_node,
                                    *parent,
                                    &separation_set,
                                    dataset,
                                    &mut cache,
                                );
                if p_value_f.2 {
                    let p_value_chi = self.Chi2test.compute_pvalues(
                                                &net,
                                                child_node,
                                                *parent,
                                                &separation_set,
                                                dataset,
                                                &mut cache,
                                            );
                
                if !(p_value_chi.2) {  
                    // lista di tuple (parent, mean_p_value)
                        //candidate_parent_set.push((*parent, (((p_value_chi.0)) / (n_tests as f64))));
                        //candidate_parent_set.push((*parent, (((p_value_chi.0)/(1e-6)))));
                        //println!("p_value chi: {}", p_value_chi.0);
                        candidate_parent_set.push((*parent, (((p_value_chi.0)))));
                    }
                }
                else{
                    //candidate_parent_set.push((*parent, (((p_value_f.0)) / (n_tests as f64))));
                    //println!("p_value f: {}", p_value_f.0);
                    candidate_parent_set.push((*parent, (((p_value_f.0)))));
                }

                // F s < limsx || s > limdx --> dependent || Chi-squared < 1-self-alpha independent --> > 1-self.alpha --> dependent
                //2: IF: the nodes are not independent they are added to candidate_parent_set ELSE: not considered
                //   eliminate all the nodes independent from T.
                /*println!("Child_node: {}, possibleParent: {}, p_value f:{}, p_value chi: {}", child_node, parent, 
                        (p_value_f.0/ (2.0 * n_tests as f64)), (p_value_chi.0/ (2.0 * n_tests as f64)));*/
                
                
            }
            //3: Sorting of the candidate parent set
            //println!("child_node {}, CPS: {:?}", child_node, candidate_parent_set);
            candidate_parent_set.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            //println!("child_node {}, CPS: {:?}", child_node, candidate_parent_set);
            //4: Initialization of the currentPS ... 
            let mut candidate_parent_set_keys: Vec<usize> = candidate_parent_set.iter().map(|(key, _)| *key).collect();

            //5: Foreach node in candidate parent set... 
            
                //6: IF T (child_node) not indep from X | separation set \subseteq currentPS
/*
            for separation_set_size in 1..=candidate_parent_set_keys.len() {  // <-- Ciclo esterno che varia `separation_set_size`
                let mut candidate_parent_set_keys_2= candidate_parent_set_keys.clone();
                for X in candidate_parent_set_keys.iter() { 

                    for separation_set in candidate_parent_set_keys
                        .iter()
                        .filter(|x| x != &X)
                        .map(|x| *x)
                        .combinations(separation_set_size)
                    {
                        let separation_set = separation_set.into_iter().collect();
                        /*
                        let mut sep_set_size = 1;

                        for i in separation_set.iter(){
                            sep_set_size *= net.get_node(i.clone()).get_reserved_space_as_parent();
                        }*/

                        
                        // 7: add X to currentPS
                        if self.Ftest.call( 
                                            &net,               
                                            child_node,
                                            *X,
                                            &separation_set,
                                            dataset,
                                            &mut cache,
                                        ) && self.Chi2test.call(        
                                            &net,                            
                                            child_node,
                                            *X,
                                            &separation_set,
                                            dataset,
                                            &mut cache,
                                        ) {  
                            candidate_parent_set_keys_2.remove(X);
                            break;
                        }

                    }
                }
                candidate_parent_set_keys = candidate_parent_set_keys_2;

            }*/
        
            /*
                1 - prima cosa prendo il primo elemento da candidate_parent_set_keys e lo metto in CurrentPC
                2 - fintantoché candidate_parent_set_keys non vuoto
                    2.1 - inserisco e faccio pop
                    2.2 - creo separation set 
                    2.3 - controllo che o il nodo è X o nel separation set ci sia un elemento che è X. (X ultimo nodo preso in considerazione)
                    2.4 - faccio il test d'ipotesi
            */
/*
            println!("{:?}", candidate_parent_set_keys);
            
            for separation_set_size in 1..=candidate_parent_set_keys.len() {  // <-- Ciclo esterno che varia `separation_set_size`
                      
                println!("Child node: {}, PS: {:?}", child_node, candidate_parent_set_keys);
                        for X in candidate_parent_set_keys.clone().iter() {
                            for separation_set in candidate_parent_set_keys
                                .clone()
                                .iter()
                                .filter(|&&x| x != *X)
                                .copied() // Copia i valori per ottenere un iteratore su `usize`
                                .combinations(separation_set_size) 
                            {
                                let separation_set: BTreeSet<usize> = separation_set.into_iter().collect();
                                
                                // 7: add X to currentPS

                                //if X == candidate_parent_set_keys.last().unwrap() || separation_set.contains(candidate_parent_set_keys.last().unwrap()){
                                    if self.Ftest.call(
                                                        &net,                   
                                                        child_node,
                                                        *X,
                                                        &separation_set,
                                                        dataset,
                                                        &mut cache,
                                                    ) && self.Chi2test.call(        
                                                        &net,                            
                                                        child_node,
                                                        *X,
                                                        &separation_set,
                                                        dataset,
                                                        &mut cache,
                                                    ){  
                                        candidate_parent_set_keys.retain(|&x| x != *X);
                                        break;
                                    }
                                //}


                            }
                        }

                    }*/
            let mut CurrentPC: Vec<usize> = Vec::new();
            if candidate_parent_set_keys.len() > 0{
                CurrentPC.push(candidate_parent_set_keys.pop().unwrap());
                while candidate_parent_set_keys.len() > 0 {

                    CurrentPC.push(candidate_parent_set_keys.pop().unwrap());
                    for separation_set_size in 1..= (CurrentPC.len() - 1){  // <-- Ciclo esterno che varia `separation_set_size`
                        //println!("Child node: {}, PS: {:?}", child_node, candidate_parent_set_keys);
                        for X in CurrentPC.clone().iter() {
                            for separation_set in CurrentPC
                                .clone()
                                .iter()
                                .filter(|&&x| x != *X)
                                .copied() // Copia i valori per ottenere un iteratore su `usize`
                                .combinations(separation_set_size) 
                            {
                                let separation_set: BTreeSet<usize> = separation_set.into_iter().collect();
                                
                                // 7: add X to currentPS

                                if X == CurrentPC.last().unwrap() || separation_set.contains(CurrentPC.last().unwrap()){
                                    if self.Ftest.call(
                                                        &net,                   
                                                        child_node,
                                                        *X,
                                                        &separation_set,
                                                        dataset,
                                                        &mut cache,
                                                    ) && self.Chi2test.call(        
                                                        &net,                            
                                                        child_node,
                                                        *X,
                                                        &separation_set,
                                                        dataset,
                                                        &mut cache,
                                                    ){  
                                        CurrentPC.retain(|&x| x != *X);
                                        break;
                                    }
                                }


                            }

                        }

                    }
                }
            }  

                (child_node, CurrentPC)
        }));
        

        for (child_node, currentPC) in learned_parent_sets {
            for parent_node in currentPC.iter() {
                net.add_edge(*parent_node, child_node);
            }
        }
        
        net
    }   
}


#![allow(non_snake_case)]

mod utils;
use std::collections::BTreeSet;

use ndarray::{arr1, arr2, arr3};
use reCTBN::parameter_learning::BayesianApproach;
use reCTBN::params;
use reCTBN::process::ctbn::*;
use reCTBN::process::NetworkProcess;
use reCTBN::structure_learning::constraint_based_algorithm::*;
use reCTBN::structure_learning::hypothesis_test::*;
use reCTBN::structure_learning::score_based_algorithm::*;
use reCTBN::structure_learning::score_function::*;
use reCTBN::structure_learning::StructuralLearningAlgorithm;
use reCTBN::structure_learning::hiton::Hiton;
use reCTBN::structure_learning::hiton;
use reCTBN::structure_learning::hiton2::Hiton2;
use reCTBN::structure_learning::hiton2;
use std::io;
use reCTBN::tools::*;
use utils::*;
use std::time::Instant;
use std::error::Error;
use std::fs::File;
use csv::Writer;
use std::time::Duration;
use serde::Serialize;
use rand::Rng;


#[macro_use]
extern crate approx;



fn generate_nodes(
    net: &mut CtbnNetwork,
    nodes_cardinality: usize,
    nodes_domain_cardinality: usize,
) {
    for node_label in 0..nodes_cardinality {
        net.add_node(generate_discrete_time_continous_node(
            node_label.to_string(),
            nodes_domain_cardinality,
        ))
        .unwrap();
    }
}


#[test]
#[ignore] 
fn global_test_density_first() -> Result<(), Box<dyn Error>>{
    //test_random_all_algorithms(0.1, 10);
    //test_random_all_algorithms(0.2, 10);
    //test_random_all_algorithms(0.3, 10);
    //test_random_all_algorithms(0.4, 10);
    //test_random_all_algorithms(0.3, 10);
    //test_random_all_algorithms(0.5, 20);
    //test_random_all_algorithms(0.05, 20);
    //test_random_all_algorithms(0.1, 20);
    //test_random_all_algorithms(0.15, 20);
    //test_random_all_algorithms(0.2, 20);
    //test_random_all_algorithms(0.20, 15);
    //test_random_all_algorithms(0.4, 20);
    //test_random_all_algorithms(0.5, 20);
    //test_random_all_algorithms(0.4, 30);
    //test_random_all_algorithms(0.2, 20);
    //test_random_all_algorithms(0.1, 30);
    //test_random_all_algorithms(0.15, 30);
    //test_random_all_algorithms(0.2, 30);
    //test_random_all_algorithms(0.3, 30);
    //test_random_all_algorithms(0.4, 30);
    println!("Enter the number of nodes:");

    let mut usize_input = String::new();
    io::stdin().read_line(&mut usize_input).expect("Failed to read usize");
    let n_node: usize = usize_input.trim().parse().expect("Please enter a valid usize");
    println!("Enter the density of the network:");
    let mut float_input = String::new();
    io::stdin().read_line(&mut float_input).expect("Failed to read float");
    let density: f64 = float_input.trim().parse().expect("Please enter a valid float");
    test_random_all_algorithms(density, n_node);   
    test_all_in_one(n_node);
    //test_random_all_algorithms(0.1, 40);
    //let (net1, real_net2, real_net3, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(20, 0.2); 
    //let data = trajectory_generator_2(&net1, 100, 10, Some(6347747169756259));
    //println!("{:?}", data);
    
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Costraint-based_random".to_string(), 10);
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Score-based_random".to_string(), 10);
    Ok(())
}

fn test_all_in_one(n_node:usize) -> Result<(), Box<dyn Error>> { 
    let filename = format!("Algorithms_{}_nodes_all_in_one.csv", n_node, );
    let file = File::create(filename.to_string())?;
    let mut wtr = Writer::from_writer(file);
    //for n in 0..100{
        let mut f;
        let mut chi_sq;
        f = F::new(1e-1);
        chi_sq = ChiSquare::new(1e-1);

        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        let hiton = Hiton2::new(parameter_learning, f, chi_sq);

        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        let ctpc = CTPC::new(parameter_learning, f, chi_sq);

        let bic = BIC::new(1, 1.0);      
        let ll = LogLikelihood::new(1, 1.0);
        let ctss = HillClimbing::new(bic, None);
        
        let (real_net1, real_net2, real_net3, data1, data2, data3) = all_in_one(n_node);
        
        learn_mixed_discrete_net_gen(&mut wtr, &hiton, 1, n_node, real_net1, data1.clone())?;
        learn_mixed_discrete_net_gen(&mut wtr, &ctpc, 1, n_node, real_net2, data1.clone())?;
        learn_mixed_discrete_net_gen(&mut wtr, &ctss, 1, n_node, real_net3, data1.clone())?;
    //}
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())


}

fn all_in_one(n_node:usize) ->(CtbnNetwork, CtbnNetwork, CtbnNetwork, Dataset, Dataset, Dataset) {

    let mut net1 = CtbnNetwork::new();
    let mut net2 = CtbnNetwork::new();
    let mut net3 = CtbnNetwork::new();
    generate_nodes(&mut net1, n_node, 3);
    generate_nodes(&mut net2, n_node, 3);
    generate_nodes(&mut net3, n_node, 3);
    
    
    let mut i = 0;

    while i < n_node-2{
        net1.add_edge(i, n_node -2);
        net2.add_edge(i, n_node -2);
        net3.add_edge(i, n_node -2);
        i = i + 1;
    }
    net1.add_edge(n_node -2, n_node -1);
    net2.add_edge(n_node -2, n_node -1);
    net3.add_edge(n_node -2, n_node -1);
    //Tree structure
    let mut i = 2;
    /*
    while i < n_node-2{
        net1.add_edge(i, i+1);
        net2.add_edge(i, i+1);
        net3.add_edge(i, i+1);
        net1.add_edge(i, i-1);
        net2.add_edge(i, i-1);
        net3.add_edge(i, i-1);
        /*
        net1.add_edge(i, i+2);
        net2.add_edge(i, i+2);
        net3.add_edge(i, i+2);
        net1.add_edge(i, i-2);
        net2.add_edge(i, i-2);
        net3.add_edge(i, i-2);*/
        i = i + 1;
    }*/


    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(10.0..30.0, Some(6813071588535822));

    cim_generator.generate_parameters(&mut net1);

    cim_generator.generate_parameters(&mut net2);

    cim_generator.generate_parameters(&mut net3);

    //let number = 30* n_node.pow(2);
    let number = 10000;
    println!("numero traj: {}", number);

    let data1 = trajectory_generator(&net1, number.try_into().unwrap(), 10.0, Some(6347747169756259)); // modificare 
    let data2 = trajectory_generator(&net1, number.try_into().unwrap(), 10.0, Some(6347747169756259));
    let data3 = trajectory_generator(&net1, number.try_into().unwrap(), 10.0, Some(6347747169756259));
    

    return (net1, net2, net3, data1, data2, data3);
}

fn average_bic_score(net: CtbnNetwork, data: Dataset) -> f64{
    let mut avg_score = 0.0;
    let ll = LogLikelihood::new(1, 1.0); 
    for i in net.get_node_indices(){
        let parent_set = net.get_parent_set(i);
        let score = ll.call(&net, i, &parent_set, &data);
        //println!("Node: {}, ParentSet: {:?}, Score: {}", i, parent_set, score);
        avg_score += score;
    }
    return (avg_score/(net.get_node_indices().len() as f64));
}

fn test_random_all_algorithms(density: f64, n_node:usize) -> Result<(), Box<dyn Error>> { 
    
    let filename = format!("New_Tests_Algorithms_{}_{}_nodes_hiton_cardinality3_2.csv", density, n_node);
    let file = File::create(filename.to_string())?;
    let mut wtr = Writer::from_writer(file);
    for n in 0..100{
        let mut f:F;
        let mut chi_sq:ChiSquare;
        

        if n_node == 10{
            if density == 0.1{
                f = F::new(5e-5);
                chi_sq = ChiSquare::new(5e-3);
            }
            else if density == 0.2{
                f = F::new(1e-3);
                chi_sq = ChiSquare::new(1e-2);
            }
            else{
                f = F::new(5e-2);
                chi_sq = ChiSquare::new(5e-1);
            }
        }
        else if n_node == 20{
            if density == 0.1{
                f = F::new(5e-5);
                chi_sq = ChiSquare::new(5e-3);
            }
            else if density == 0.05{
                f = F::new(1e-2);
                chi_sq = ChiSquare::new(1e-2);
            }
            else if density == 0.15{
                f = F::new(1e-5);
                chi_sq = ChiSquare::new(1e-3);
            }
            else if density == 0.2{
                f = F::new(8e-1);
                chi_sq = ChiSquare::new(8e-1);
            }
            else{
                f = F::new(8e-1);
                chi_sq = ChiSquare::new(8e-1);
            }
        }
        else {
            if density == 0.1{
                f = F::new(5e-5);
                chi_sq = ChiSquare::new(5e-3);
            }
            else if density == 0.15{
                f = F::new(5e-3);
                chi_sq = ChiSquare::new(5e-2);
            }
            else if density == 0.2{
                f = F::new(1e-1);
                chi_sq = ChiSquare::new(1e-1);
            }
            else{
                f = F::new(8e-1);
                chi_sq = ChiSquare::new(8e-1);
            }
        }
        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        

        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        

        let bic = BIC::new(1, 1.0);      
        let ll = LogLikelihood::new(1, 1.0);


        let (real_net1, real_net2, real_net3, data1, data2, data3) = 
                    get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, density);

        let likelihood = average_bic_score(real_net1.clone(), data1.clone());
        println!("Avg bic: {}", likelihood);

        /*let hiton = Hiton::new(parameter_learning, f, chi_sq); 
        //learn_mixed_discrete_net_gen(&mut wtr, &hiton, 1, n_node, real_net1.clone(), data1.clone())?;
        learn_mixed_discrete_net_gen(&mut wtr, &hiton, 1, n_node, real_net1.clone(), data2.clone())?;*/
        let hiton2 = Hiton2::new(parameter_learning, f, chi_sq); 
        //learn_mixed_discrete_net_gen(&mut wtr, &hiton, 1, n_node, real_net1.clone(), data1.clone())?;
        learn_mixed_discrete_net_gen(&mut wtr, &hiton2, 1, n_node, real_net1.clone(), data2.clone())?;
        //learn_mixed_discrete_net_gen(&mut wtr, &hiton, 1, n_node, real_net1.clone(), data3.clone())?;
        let ctpc = CTPC::new(parameter_learning, f, chi_sq);
        //learn_mixed_discrete_net_gen(&mut wtr, &ctpc, 1, n_node, real_net1.clone(), data1.clone())?;
        learn_mixed_discrete_net_gen(&mut wtr, &ctpc, 1, n_node, real_net1.clone(), data2.clone())?;
        //learn_mixed_discrete_net_gen(&mut wtr, &ctpc, 1, n_node, real_net1.clone(), data3.clone())?;
        let ctss = HillClimbing::new(ll, None);
        //learn_mixed_discrete_net_gen(&mut wtr, &ctss, 1, n_node, real_net1.clone(), data1.clone())?;
        learn_mixed_discrete_net_gen(&mut wtr, &ctss, 1, n_node, real_net1.clone(), data2.clone())?;
        //learn_mixed_discrete_net_gen(&mut wtr, &ctss, 1, n_node, real_net1.clone(), data3.clone())?;

 

    }
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())

}

fn learn_mixed_discrete_net_gen<T: StructuralLearningAlgorithm>(wtr: &mut Writer<File>, sl: &T, n_test: usize, n_node:usize, 
                                                                real_net:CtbnNetwork, data: Dataset) -> Result<(), Box<dyn Error>> {


    let n = real_net.get_node_indices().len();
    let original_mat = create_adj_matrix(&real_net);

    let start = Instant::now();
    let net = sl.fit_transform(real_net, &data);
    let duration = start.elapsed();
    

    let avg_bic = average_bic_score(net.clone(), data.clone());
    println!("Avg bic: {}", avg_bic);
    

    let computed_mat = create_adj_matrix(&net);
    let mut tp = 0; 
    let mut fp = 0; 
    let mut fn_ = 0; 

    for i in 0..n {
        for j in 0..n {
            if i != j {
                match (original_mat[i][j], computed_mat[i][j]) {
                    (1, 1) => tp += 1,
                    (0, 1) => fp += 1,
                    (1, 0) => fn_ += 1,
                    _ => {} 
                }
            }
        }
    }

    for row in &original_mat {
        for val in row {
            print!("{} ", val);
        }
        println!(); 
    }
    println!(); 
    println!(); 
    for row in &computed_mat {
        for val in row {
            print!("{} ", val);
        }
        println!();
    }

    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64/(tp + fp)as f64
    };
    
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64/(tp + fn_)as f64
    };
  
    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        (2.0 * precision * recall) / (precision + recall)
    };
    println!("Duration: {:?}, Precision: {}, Recall: {}, f1_score:{}", duration, precision, recall, f1_score);
    save_csv(wtr, duration, f1_score, precision, recall, avg_bic)?;

    Ok(())
}

/*
#[test]
//#[ignore] 
fn global_test() -> Result<(), Box<dyn Error>>{
    learn_mixed_discrete_net_10_nodes_hiton_gen_2("Hiton_random".to_string(), 10);
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Costraint-based_random".to_string(), 10);
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Score-based_random".to_string(), 10);
    Ok(())
}

#[test]
//#[ignore] 
fn global_test_20_nodes() -> Result<(), Box<dyn Error>>{
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Hiton_random".to_string(), 10);
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Costraint-based_random".to_string(), 10);
    //learn_mixed_discrete_net_10_nodes_hiton_gen_2("Score-based_random".to_string(), 10);
    Ok(())
}*/
/*
fn learn_mixed_discrete_net_10_nodes_hiton_gen_2(model: String, n_node:usize) -> Result<(), Box<dyn Error>> { 
    let filename = format!("{}_{}_nodes.csv", model, n_node);
    let file = File::create(filename.to_string())?;
    let mut wtr = Writer::from_writer(file);
   

    if model == "Hiton_random" {
        let f = F::new(1e-6);
        let chi_sq = ChiSquare::new(1e-4);
        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        let ctpc = Hiton::new(parameter_learning, f, chi_sq);
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 1, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 2, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 3, n_node)?;
    }
    else if model == "Costraint-based_random"{
        let f = F::new(1e-6);
        let chi_sq = ChiSquare::new(1e-4);
        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        let ctpc = CTPC::new(parameter_learning, f, chi_sq);
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 1, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 2, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 3, n_node)?;
    }
    else if model == "Score-based_random"{
        let bic = BIC::new(1, 1.0);                              // BIC with None Duration: 16.9263123s // with Some(1) Duration: 17.6956868s
        let ll = LogLikelihood::new(1, 1.0);
        let ctpc = HillClimbing::new(bic, None);
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 1, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 2, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 3, n_node)?;
    }


    
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())

}*/


/*
#[test]
//#[ignore] 
fn learn_mixed_discrete_net_10_nodes_constraint_gen_2() -> Result<(), Box<dyn Error>> { // Duration: 10.2790371s
    let file = File::create("Constraint-based_algorithm.csv".to_string())?;
    let mut wtr = Writer::from_writer(file);
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = CTPC::new(parameter_learning, f, chi_sq);
    learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc)?;
    learn_mixed_discrete_net_10_nodes_gen_hiton_2(&mut wtr, &ctpc)?;
    learn_mixed_discrete_net_10_nodes_gen_hiton_3(&mut wtr, &ctpc)?;
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())

}


#[test]
//#[ignore]
fn learn_mixed_discrete_net_10_nodes_hill_climbing_gen_2() -> Result<(), Box<dyn Error>> { // LogLikelihood with None Duration: 17.320026s // with Some(1) Duration: 17.0288328s
    let file = File::create("Score-based_algorithm.csv".to_string())?;
    let mut wtr = Writer::from_writer(file);
    let bic = BIC::new(1, 1.0);                              // BIC with None Duration: 16.9263123s // with Some(1) Duration: 17.6956868s
    let ll = LogLikelihood::new(1, 1.0);
    let ctpc = HillClimbing::new(bic, None);
    learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc)?;
    learn_mixed_discrete_net_10_nodes_gen_hiton_2(&mut wtr, &ctpc)?;
    learn_mixed_discrete_net_10_nodes_gen_hiton_3(&mut wtr, &ctpc)?;
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())
}
*/
/*
fn learn_mixed_discrete_net_10_nodes_gen_hiton<T: StructuralLearningAlgorithm>(wtr: &mut Writer<File>, sl: &T, n_test: usize, n_node:usize) -> Result<(), Box<dyn Error>> {
    
    let (real_net, data);
    if n_node == 10{
        if n_test == 1{
            (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, 0.1);
        }
        else if n_test == 2{
            (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, 0.2);
        }
        else{
            (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, 0.3);
        }
    }
    else {
        if n_test == 1{
            (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, 0.1);
        }
        else if n_test == 2{
            (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, 0.2);
        }
        else{
            (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node, 0.3);
        }
    }

    let n = real_net.get_node_indices().len();
    let original_mat = create_adj_matrix(&real_net);

    let start = Instant::now();
    let net = sl.fit_transform(real_net, &data);
    let duration = start.elapsed();

    println!("Duration: {:?}", duration);

    let computed_mat = create_adj_matrix(&net);
    let mut tp = 0; 
    let mut fp = 0; 
    let mut fn_ = 0; 

    for i in 0..n {
        for j in 0..n {
            if i != j {


                match (original_mat[i][j], computed_mat[i][j]) {
                    (1, 1) => tp += 1,
                    (0, 1) => fp += 1,
                    (1, 0) => fn_ += 1,
                    _ => {} 
                }
            }
        }
    }
    let precision = tp as f64/(tp + fp)as f64;
    let recall = tp as f64/(tp + fn_) as f64;
  
    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        (2.0 * precision * recall) / (precision + recall)
    };

    save_csv(wtr, duration, f1_score)?;

    Ok(())
}*/

/*
#[test]
#[ignore]
fn test_prova(){
    let (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_random(10, 0.1);
    let edges = create_adj_matrix(&real_net);
    for row in &edges {
        for val in row {
            print!("{}\t", val);
        }
        println!();
    }
}*/

fn get_mixed_discrete_net_10_nodes_with_data_gen_random(n_node:usize, density:f64) ->(CtbnNetwork, CtbnNetwork, CtbnNetwork, Dataset, Dataset, Dataset) {
    let mut net1 = CtbnNetwork::new();
    let mut net2 = CtbnNetwork::new();
    let mut net3 = CtbnNetwork::new();
    let max_edges = n_node as f64 * (n_node-1) as f64 * density;
    generate_nodes(&mut net1, n_node, 3);
    generate_nodes(&mut net2, n_node, 3);
    generate_nodes(&mut net3, n_node, 3);
    
    let mut i = 0.0;
    while i < (max_edges){
        let mut rng = rand::thread_rng();
        let node_1 = rng.gen_range(0..=(n_node-1));
        let node_2 = rng.gen_range(0..=(n_node-1));
        if node_1 == node_2{
            continue;
        }
        i = i + 1.0;
        net1.add_edge(node_1, node_2);
        net2.add_edge(node_1, node_2);
        net3.add_edge(node_1, node_2);
    }
    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(10.0..30.0, Some(6813071588535822)); 
    cim_generator.generate_parameters(&mut net1);
    cim_generator.generate_parameters(&mut net2);
    cim_generator.generate_parameters(&mut net3);

    let number = (30 * n_node);
    //println!("numero traj: {}", number);
    let cont = ((n_node * n_node) as f64 * density) as usize / n_node;
    //let data = trajectory_generator(&net1, number.try_into().unwrap(), 30.0, Some(6347747169756259)); // modificare 
    
    let data1 = trajectory_generator_2(&net1, number.try_into().unwrap(), 
                                                (3usize.pow(cont as u32 + 2) as f64 * 0.5) as usize, Some(6347747169756259), n_node);
    let data2 = trajectory_generator_2(&net1, number.try_into().unwrap(), 
                                                3usize.pow(cont as u32 + 2), Some(6347747169756259), n_node);
    let data3 = trajectory_generator_2(&net1, number.try_into().unwrap(), 
                                                (3usize.pow(cont as u32 + 2) as f64 * 1.5) as usize, Some(6347747169756259), n_node);
    /*let data1 = trajectory_generator_2(&net1, number.try_into().unwrap(), 
                                                40, Some(6347747169756259), n_node);
    let data2 = trajectory_generator_2(&net1, number.try_into().unwrap(), 
                                               40, Some(6347747169756259), n_node);
    let data3 = trajectory_generator_2(&net1, number.try_into().unwrap(), 
                                                40, Some(6347747169756259), n_node);*/

    return (net1, net2, net3, data1, data2, data3);
    //let data = trajectory_generator(&net1, number.try_into().unwrap(), 10.0, Some(6347747169756259));
    //let data = trajectory_generator_2(&net1, 1, 729, Some(6347747169756259)); 
    //return (net1, net2, net3, data);

}



fn get_mixed_discrete_net_10_nodes_with_data_gen() -> (CtbnNetwork, Dataset) { // 23 edges > 20% coverage
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 100, 3);
    //net.add_node(generate_discrete_time_continous_node(String::from("9"), 4));
        //.unwrap();

        net.add_edge(3, 0);
        net.add_edge(6, 0);
    
        net.add_edge(0, 1);
        net.add_edge(5, 1);
    
        net.add_edge(4, 2);
        net.add_edge(1, 2);
    
        net.add_edge(1, 3);
        net.add_edge(8, 3);
    
        net.add_edge(7, 4);
        net.add_edge(2, 4);
    
        net.add_edge(2, 5);
        net.add_edge(9, 5);
    
        net.add_edge(3, 6);
        net.add_edge(5, 6);
        
        net.add_edge(6, 8);
        net.add_edge(9, 8);

        net.add_edge(3, 9);
        net.add_edge(7, 9);

    

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..50.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 1200, 60.0, Some(6347747169756259));
    
    return (net, data);
}



fn create_adj_matrix(net: &CtbnNetwork) -> Vec<Vec<usize>>{
    let n = net.get_node_indices().len();
    let mut adj_matrix = vec![vec![0; n]; n];
    for node in net.get_node_indices() {
        for parent in net.get_parent_set(node){
            adj_matrix[node][parent] = 1;
        }
    }
    return adj_matrix;
}

#[derive(Serialize)]
struct Record {
    f1_score: f64,
    precision: f64,
    recall: f64,
    duration: f64,
    avg_bic:f64,
}
/*
fn learn_mixed_discrete_net_10_nodes_gen_hiton_2<T: StructuralLearningAlgorithm>(wtr: &mut Writer<File>, sl: &T) -> Result<(), Box<dyn Error>>  {
    let (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_2();
    let n = real_net.get_node_indices().len();
    let original_mat = create_adj_matrix(&real_net);

    let start = Instant::now();
    let net = sl.fit_transform(real_net, &data);
    let duration = start.elapsed();

    println!("Duration: {:?}", duration);

    let computed_mat = create_adj_matrix(&net);
    let mut tp = 0; 
    let mut fp = 0; 
    let mut fn_ = 0; 

    for i in 0..n {
        for j in 0..n {
            if i != j {


                match (original_mat[i][j], computed_mat[i][j]) {
                    (1, 1) => tp += 1,
                    (0, 1) => fp += 1,
                    (1, 0) => fn_ += 1,
                    _ => {} 
                }
            }
        }
    }
    let precision = tp as f64/(tp + fp)as f64;
    let recall = tp as f64/(tp + fn_) as f64;
  
    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        (2.0 * precision * recall) / (precision + recall)
    };

    save_csv(wtr, duration, f1_score)?;

    Ok(())
    
}*/


fn get_mixed_discrete_net_10_nodes_with_data_gen_2() -> (CtbnNetwork, Dataset) { // 20%
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 9, 3);
    net.add_node(generate_discrete_time_continous_node(String::from("9"), 4))
        .unwrap();

        net.add_edge(3, 0);
        net.add_edge(6, 0);
    
        net.add_edge(0, 1);
        net.add_edge(5, 1);
    
        net.add_edge(4, 2);
        net.add_edge(1, 2);
    
        net.add_edge(1, 3);
        net.add_edge(8, 3);
    
        net.add_edge(7, 4);
        net.add_edge(2, 4);
    
        net.add_edge(2, 5);
        net.add_edge(9, 5);
    
        net.add_edge(3, 6);
        net.add_edge(5, 6);
        
        net.add_edge(6, 8);
        net.add_edge(9, 8);

        net.add_edge(3, 9);
        net.add_edge(7, 9);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..10.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259));
    return (net, data);

}

/*
fn learn_mixed_discrete_net_10_nodes_gen_hiton_3<T: StructuralLearningAlgorithm>(wtr: &mut Writer<File>, sl: &T) -> Result<(), Box<dyn Error>> {
    
    let (real_net, data) = get_mixed_discrete_net_10_nodes_with_data_gen_3();
    let n = real_net.get_node_indices().len();
    let original_mat = create_adj_matrix(&real_net);

    let start = Instant::now();
    let net = sl.fit_transform(real_net, &data);
    let duration = start.elapsed();

    println!("Duration: {:?}", duration);

    let computed_mat = create_adj_matrix(&net);
    let mut tp = 0; 
    let mut fp = 0; 
    let mut fn_ = 0; 

    for i in 0..n {
        for j in 0..n {
            if i != j {


                match (original_mat[i][j], computed_mat[i][j]) {
                    (1, 1) => tp += 1,
                    (0, 1) => fp += 1,
                    (1, 0) => fn_ += 1,
                    _ => {} 
                }
            }
        }
    }
    let precision = tp as f64/(tp + fp)as f64;
    let recall = tp as f64/(tp + fn_) as f64;
  
    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        (2.0 * precision * recall) / (precision + recall)
    };

    save_csv(wtr, duration, f1_score)?;

    Ok(())
}*/

fn get_mixed_discrete_net_10_nodes_with_data_gen_3() -> (CtbnNetwork, Dataset) { // 20%
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 9, 4);
    net.add_node(generate_discrete_time_continous_node(String::from("9"), 4))
        .unwrap();

        net.add_edge(3, 0);
        net.add_edge(6, 0);
    
        net.add_edge(0, 1);
        net.add_edge(5, 1);
    
        net.add_edge(4, 2);
        net.add_edge(1, 2);
    
        net.add_edge(1, 3);
        net.add_edge(8, 3);
    
        net.add_edge(7, 4);
        net.add_edge(2, 4);
    
        net.add_edge(2, 5);
        net.add_edge(9, 5);
    
        net.add_edge(3, 6);
        net.add_edge(5, 6);
        
        net.add_edge(6, 8);
        net.add_edge(9, 8);

        net.add_edge(3, 9);
        net.add_edge(7, 9);

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..50.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 1200, 60.0, Some(6347747169756259));
    return (net, data);
}


fn save_csv(wtr: &mut Writer<File>, duration: Duration, f1_score: f64, precision: f64, recall:f64, avg_bic:f64)  -> Result<(), Box<dyn Error>>{

    let secs = duration.as_secs();
    let nanos = duration.subsec_nanos(); 

    let fraction = nanos as f64 / 1_000_000_000.0;

    let dur = secs as f64 + fraction;
    

    let records = vec![
        Record {
            
            f1_score: f1_score,
            precision: precision,
            recall: recall,
            duration:  dur,
            avg_bic: avg_bic,

        },
    ];

    for record in records {
        wtr.serialize(record)?;
    }

    Ok(())
}



/*
#[test]

fn learn_mixed_discrete_net_10_nodes_hiton_gen() {
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = Hiton::new(parameter_learning, f, chi_sq);
    learn_mixed_discrete_net_10_nodes_gen_hiton(&ctpc);
}*/



/*
#[test]
//#[ignore]
fn learn_mixed_discrete_net_20_nodes_hiton_gen_second_simulation() -> Result<(), Box<dyn Error>> { // LogLikelihood with None Duration: 17.320026s // with Some(1) Duration: 17.0288328s
    let file = File::create("Hiton2.csv".to_string())?;
    let mut wtr = Writer::from_writer(file);
    let f = F::new(1e-6);
    let chi_sq = ChiSquare::new(1e-4);
    let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
    let ctpc = Hiton::new(parameter_learning, f, chi_sq);

    learn_mixed_discrete_net_20_nodes_gen_hiton_second_simul(&mut wtr, &ctpc)?;
    //learn_mixed_discrete_net_10_nodes_gen_hiton_2(&mut wtr, &ctpc)?;
    //learn_mixed_discrete_net_10_nodes_gen_hiton_3(&mut wtr, &ctpc)?;
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())
}

#[test]
#[ignore]
fn learn_mixed_discrete_net_20_nodes_score_based_gen_second_simulation() -> Result<(), Box<dyn Error>> { // LogLikelihood with None Duration: 17.320026s // with Some(1) Duration: 17.0288328s
    let file = File::create("Score-based_algorithm.csv".to_string())?;
    let mut wtr = Writer::from_writer(file);
    let bic = BIC::new(1, 1.0);                              // BIC with None Duration: 16.9263123s // with Some(1) Duration: 17.6956868s
    let ll = LogLikelihood::new(1, 1.0);
    let ctpc = HillClimbing::new(ll, None);

    learn_mixed_discrete_net_20_nodes_gen_hiton_second_simul(&mut wtr, &ctpc)?;
    //learn_mixed_discrete_net_10_nodes_gen_hiton_2(&mut wtr, &ctpc)?;
    //learn_mixed_discrete_net_10_nodes_gen_hiton_3(&mut wtr, &ctpc)?;
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())
}
*/



/*
fn learn_mixed_discrete_net_10_nodes_hiton_gen_2(model: String) -> Result<(), Box<dyn Error>> { // Duration: 9.9982139s
    let filename = format!("{}.csv", model);
    let file = File::create(filename.to_string())?;
    let mut wtr = Writer::from_writer(file);
    let n_node = 20;
    //let ctpc: CTPCType;

    if model == "Hiton_20_nodes" {
        let f = F::new(1e-6);
        let chi_sq = ChiSquare::new(1e-4);
        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        let ctpc = Hiton::new(parameter_learning, f, chi_sq);
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 1, n_node)?;
        /*learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 2, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 3, n_node)?;*/
    }
    else if model == "Costraint-based_20_nodes"{
        let f = F::new(1e-6);
        let chi_sq = ChiSquare::new(1e-4);
        let parameter_learning = BayesianApproach { alpha: 1, tau: 1.0 };
        let ctpc = CTPC::new(parameter_learning, f, chi_sq);
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 1, n_node)?;
        /*learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 2, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 3, n_node)?;*/
    }
    else if model == "Score-based_20_nodes"{
        let bic = BIC::new(1, 1.0);                              // BIC with None Duration: 16.9263123s // with Some(1) Duration: 17.6956868s
        let ll = LogLikelihood::new(1, 1.0);
        let ctpc = HillClimbing::new(bic, None);
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 1, n_node)?;
        /*learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 2, n_node)?;
        learn_mixed_discrete_net_10_nodes_gen_hiton(&mut wtr, &ctpc, 3, n_node)?;*/
    }


    
    wtr.flush()?;

    println!("CSV salvato con successo!");
    Ok(())

}*/

fn get_mixed_discrete_net_20_nodes_with_data_gen_first_simul() -> (CtbnNetwork, Dataset) { // 23 edges > 20% coverage
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 20, 3);
    //net.add_node(generate_discrete_time_continous_node(String::from("9"), 4));
        //.unwrap();

        net.add_edge(3, 0);
        net.add_edge(6, 0);
    
        net.add_edge(0, 1);
        net.add_edge(5, 1);
    
        net.add_edge(4, 2);
        net.add_edge(1, 2);
    
        net.add_edge(1, 3);
        net.add_edge(8, 3);
    
        net.add_edge(7, 4);
        net.add_edge(2, 4);
    
        net.add_edge(2, 5);
        net.add_edge(9, 5);
    
        net.add_edge(3, 6);
        net.add_edge(5, 6);
        
        net.add_edge(6, 8);
        net.add_edge(9, 8);

        net.add_edge(3, 9);
        net.add_edge(7, 9);

        net.add_edge(3, 10);
        net.add_edge(16, 10);
    
        net.add_edge(10, 11);
        net.add_edge(5, 11);
    
        net.add_edge(4, 12);
        net.add_edge(1, 2);
    
        net.add_edge(11, 13);
        net.add_edge(8, 13);
    
        net.add_edge(7, 14);
        net.add_edge(12, 14);
    
        net.add_edge(2, 15);
        net.add_edge(9, 15);
    
        net.add_edge(13, 16);
        net.add_edge(15, 16);
        
        net.add_edge(6, 18);
        net.add_edge(19, 18);

        net.add_edge(13, 19);
        net.add_edge(7, 19);

    

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..10.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259));
    return (net, data);
}

fn get_mixed_discrete_net_20_nodes_with_data_gen_second_simul() -> (CtbnNetwork, Dataset) { // 23 edges > 20% coverage
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 20, 4);
    //net.add_node(generate_discrete_time_continous_node(String::from("9"), 4));
        //.unwrap();

        net.add_edge(3, 0);
        net.add_edge(6, 0);
    
        net.add_edge(0, 1);
        net.add_edge(5, 1);
    
        net.add_edge(4, 2);
        net.add_edge(1, 2);
    
        net.add_edge(1, 3);
        net.add_edge(8, 3);
    
        net.add_edge(7, 4);
        net.add_edge(2, 4);
    
        net.add_edge(2, 5);
        net.add_edge(9, 5);
    
        net.add_edge(3, 6);
        net.add_edge(5, 6);
        
        net.add_edge(6, 8);
        net.add_edge(9, 8);

        net.add_edge(3, 9);
        net.add_edge(7, 9);

        net.add_edge(3, 10);
        net.add_edge(16, 10);
    
        net.add_edge(10, 11);
        net.add_edge(5, 11);
    
        net.add_edge(4, 12);
        net.add_edge(1, 2);
    
        net.add_edge(11, 13);
        net.add_edge(8, 13);
    
        net.add_edge(7, 14);
        net.add_edge(12, 14);
    
        net.add_edge(2, 15);
        net.add_edge(9, 15);
    
        net.add_edge(13, 16);
        net.add_edge(15, 16);
        
        net.add_edge(6, 18);
        net.add_edge(19, 18);

        net.add_edge(13, 19);
        net.add_edge(7, 19);

    

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..10.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259));
    return (net, data);
}

fn get_mixed_discrete_net_20_nodes_with_data_gen_third_simul() -> (CtbnNetwork, Dataset) { // 23 edges > 20% coverage
    let mut net = CtbnNetwork::new();
    generate_nodes(&mut net, 19, 3);
    net.add_node(generate_discrete_time_continous_node(String::from("20"), 4)).unwrap();

        net.add_edge(3, 0);
        net.add_edge(6, 0);
    
        net.add_edge(0, 1);
        net.add_edge(5, 1);
    
        net.add_edge(4, 2);
        net.add_edge(1, 2);
    
        net.add_edge(1, 3);
        net.add_edge(8, 3);
    
        net.add_edge(7, 4);
        net.add_edge(2, 4);
    
        net.add_edge(2, 5);
        net.add_edge(9, 5);
    
        net.add_edge(3, 6);
        net.add_edge(5, 6);
        
        net.add_edge(6, 8);
        net.add_edge(9, 8);

        net.add_edge(3, 9);
        net.add_edge(7, 9);

        net.add_edge(3, 10);
        net.add_edge(16, 10);
    
        net.add_edge(10, 11);
        net.add_edge(5, 11);
    
        net.add_edge(4, 12);
        net.add_edge(1, 2);
    
        net.add_edge(11, 13);
        net.add_edge(8, 13);
    
        net.add_edge(7, 14);
        net.add_edge(12, 14);
    
        net.add_edge(2, 15);
        net.add_edge(9, 15);
    
        net.add_edge(13, 16);
        net.add_edge(15, 16);
        
        net.add_edge(6, 18);
        net.add_edge(19, 18);

        net.add_edge(13, 19);
        net.add_edge(7, 19);

    

    let mut cim_generator: UniformParametersGenerator =
        RandomParametersGenerator::new(1.0..10.0, Some(6813071588535822));
    cim_generator.generate_parameters(&mut net);

    let data = trajectory_generator(&net, 300, 30.0, Some(6347747169756259));
    return (net, data);
}
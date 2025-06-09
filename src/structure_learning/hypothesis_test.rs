//! Module for constraint based algorithms containing hypothesis test algorithms like chi-squared test, F test, etc...

use std::collections::BTreeSet;

use ndarray::{Array3, Axis};
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor};

use crate::params::*;
use crate::structure_learning::constraint_based_algorithm::Cache;
use crate::{parameter_learning, process, tools::Dataset};

pub trait HypothesisTest {
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> bool//(f64, bool) //bool//
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning;

    fn compute_pvalues<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> (f64, f64, bool)
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning;
}

/// Does the chi-squared test (χ2 test).
///
/// Used to determine if a difference between two sets of data is due to chance, or if it is due to
/// a relationship (dependence) between the variables.
///
/// # Arguments
///
/// * `alpha` - is the significance level, the probability to reject a true null hypothesis;
///   in other words is the risk of concluding that an association between the variables exists
///   when there is no actual association.
#[derive(Copy, Clone)]
pub struct ChiSquare {
    alpha: f64,
}

/// Does the F-test.
///
/// Used to determine if a difference between two sets of data is due to chance, or if it is due to
/// a relationship (dependence) between the variables.
///
/// # Arguments
///
/// * `alpha` - is the significance level, the probability to reject a true null hypothesis;
///   in other words is the risk of concluding that an association between the variables exists
///   when there is no actual association.
#[derive(Copy, Clone)]
pub struct F {
    alpha: f64,
}


// try 1

impl F {
    pub fn new(alpha: f64) -> F {
        F { alpha }
    }

    /// Compare two matrices extracted from two 3rd-orer tensors.
    ///
    /// # Arguments
    ///
    /// * `i` - Position of the matrix of `M1` to compare with `M2`.
    /// * `M1` - 3rd-order tensor 1.
    /// * `j` - Position of the matrix of `M2` to compare with `M1`.
    /// * `M2` - 3rd-order tensor 2.
    ///
    /// # Returns
    ///
    /// * `true` - when the matrices `M1` and `M2` are very similar, then **independendent**.
    /// * `false` - when the matrices `M1` and `M2` are too different, then **dependent**.
/*
    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        cim_1: &Array3<f64>,
        j: usize,
        M2: &Array3<usize>,
        cim_2: &Array3<f64>,
    ) -> (f64, bool) {
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let cim_1 = cim_1.index_axis(Axis(0), i);
        let cim_2 = cim_2.index_axis(Axis(0), j);
        let r1 = M1.sum_axis(Axis(1));
        let r2 = M2.sum_axis(Axis(1));
        let q1 = cim_1.diag();
        let q2 = cim_2.diag();
        let mut counter = 0;

        /*let lim_sx = self.alpha / 2.0;
        let lim_dx = 1.0 - (self.alpha / 2.0);

        let (sum_s, all_within_limits) = (0..r1.len()).fold((0.0, true), |(sum, within), idx| { // TO-DO Chi-squared
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
    
            (
                sum + (s - 0.5).abs() * 2.0,             // NB: To check.
                within && (s >= lim_sx && s <= lim_dx),
            )
        });
        return (sum_s, self.alpha, all_within_limits)*/
        /*
        let mut cont = 0.0;
        let mut acc = 0.0;
        
        for idx in 0..r1.shape()[0] {
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
            let lim_sx = self.alpha / 2.0;
            let lim_dx = 1.0 - (self.alpha / 2.0);
            cont += 1.0;
            acc += (s - 0.5).abs()*2.0;
            if s < lim_sx || s > lim_dx { // lascio?
                return (r , false);
            }
        }*/

        
        for idx in 0..r1.shape()[0] {
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
            let lim_sx = self.alpha / 2.0;
            let lim_dx = 1.0 - (self.alpha / 2.0);
           
            if s < lim_sx || s > lim_dx {
                let s  = (s - 0.5).abs()*2.0; // lascio?
                return (s , false);
            }
        }

        (0.0, true)

    }
    */

    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        cim_1: &Array3<f64>,
        j: usize,
        M2: &Array3<usize>,
        cim_2: &Array3<f64>,
    ) -> bool {
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let cim_1 = cim_1.index_axis(Axis(0), i);
        let cim_2 = cim_2.index_axis(Axis(0), j);
        let r1 = M1.sum_axis(Axis(1));
        let r2 = M2.sum_axis(Axis(1));
        let q1 = cim_1.diag();
        let q2 = cim_2.diag();
        let mut counter = 0;
        let alpha_corrected = self.alpha / ((M2.shape()[0] * M2.shape()[1]) as f64);
        let lim_sx = alpha_corrected / 2.0;
        let lim_dx = 1.0 - (alpha_corrected / 2.0);

        /*let lim_sx = self.alpha / 2.0;
        let lim_dx = 1.0 - (self.alpha / 2.0);

        let (sum_s, all_within_limits) = (0..r1.len()).fold((0.0, true), |(sum, within), idx| { // TO-DO Chi-squared
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
    
            (
                sum + (s - 0.5).abs() * 2.0,             // NB: To check.
                within && (s >= lim_sx && s <= lim_dx),
            )
        });
        return (sum_s, self.alpha, all_within_limits)*/

        
        for idx in 0..r1.shape()[0] {
            if r1[idx] == 0.0 || r2[idx] == 0.0{
                continue;
            }
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
            
            if s < lim_sx || s > lim_dx {
                return false;
            }
        }

        true

    }
 
    
    pub fn compute_pvalue_F_test(
        &self,
        i: usize,
        M1: &Array3<usize>,
        cim_1: &Array3<f64>,
        j: usize,
        M2: &Array3<usize>,
        cim_2: &Array3<f64>,
    ) -> (f64, f64, bool) {
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let cim_1 = cim_1.index_axis(Axis(0), i);
        let cim_2 = cim_2.index_axis(Axis(0), j);
        let r1 = M1.sum_axis(Axis(1));
        let r2 = M2.sum_axis(Axis(1));
        let q1 = cim_1.diag();
        let q2 = cim_2.diag();
        let mut counter = 0;
        let alpha_corrected = self.alpha / ((M2.shape()[0] * M2.shape()[1]) as f64);
        let lim_sx = alpha_corrected / 2.0;
        let lim_dx = 1.0 - (alpha_corrected / 2.0);
        
        let (sum_s, all_within_limits) = (0..r1.len()).fold((0.0, true), |(sum, within), idx| { // TO-DO Chi-squared
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
    
            (
                sum + (s - 0.5).abs() * 2.0,
                within && (s >= lim_sx && s <= lim_dx), 
            )
        });
        return (sum_s, self.alpha, all_within_limits)
        
        /*let (min_s, all_within_limits) = (0..r1.len()).fold((f64::INFINITY, true), |(min_val, within), idx| {
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
            let val = (s - 0.5).abs() * 2.0;

            (
                min_val.min(val),
                within && (s >= lim_sx && s <= lim_dx), 
            )
        });
        return (min_s, self.alpha, all_within_limits)*/
        /*
        let (max_s, all_within_limits) = (0..r1.len()).fold((0.0_f64, true), |(max_val, within), idx| {
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
            let val = (s - 0.5).abs() * 2.0;
            //println!("{}", val);
            (
                max_val.max(val as f64),
                within && (s >= lim_sx && s <= lim_dx), 
            )
        });
        //println!("max: {:?} f_test", max_s);
        return (max_s, self.alpha, all_within_limits)*/

    }
}

impl HypothesisTest for F {
    
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> bool
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();



        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;

            if !(self.compare_matrices(
                    idx_M_small,
                    P_small.get_transitions().as_ref().unwrap(),
                    P_small.get_cim().as_ref().unwrap(),
                    idx_M_big,
                    P_big.get_transitions().as_ref().unwrap(),
                    P_big.get_cim().as_ref().unwrap(),
                )){
                return false;
            }
        }
        return true;
    }
    /*
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> (f64, bool)
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();



        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;

            let res = self.compare_matrices(
                    idx_M_small,
                    P_small.get_transitions().as_ref().unwrap(),
                    P_small.get_cim().as_ref().unwrap(),
                    idx_M_big,
                    P_big.get_transitions().as_ref().unwrap(),
                    P_big.get_cim().as_ref().unwrap(),
                );

            if !(res.1){
                return (res.0/self.alpha, false);
            }
        }
        return (0.0, true);
    }
*/

    fn compute_pvalues<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> (f64, f64, bool)
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning, 
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();


        //let mut min = f64::INFINITY;
        //let mut max = 0.0;
        let mut sum = 0.0;
        let mut alpha = 0.0;
        let mut all_within_limits = true;

        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;

            let p_value = self.compute_pvalue_F_test(
                                idx_M_small,
                                P_small.get_transitions().as_ref().unwrap(),
                                P_small.get_cim().as_ref().unwrap(),
                                idx_M_big,
                                P_big.get_transitions().as_ref().unwrap(),
                                P_big.get_cim().as_ref().unwrap(),
                            );
            alpha = p_value.1;
            all_within_limits = all_within_limits && p_value.2;
            /*if min > p_value.0{
                min = p_value.0;
            }*/
            /*if max < p_value.0{
                max = p_value.0;
            }*/
            sum += p_value.0;

        }
        //return ((min)/self.alpha, alpha, all_within_limits);
        //return ((max)/self.alpha, alpha, all_within_limits);
        return ((sum)/self.alpha, alpha, all_within_limits); //p_value mean and limits p_value. | alpha always the same?

    }
}



// Original Code
/*
impl F {
    pub fn new(alpha: f64) -> F {
        F { alpha }
    }

    /// Compare two matrices extracted from two 3rd-orer tensors.
    ///
    /// # Arguments
    ///
    /// * `i` - Position of the matrix of `M1` to compare with `M2`.
    /// * `M1` - 3rd-order tensor 1.
    /// * `j` - Position of the matrix of `M2` to compare with `M1`.
    /// * `M2` - 3rd-order tensor 2.
    ///
    /// # Returns
    ///
    /// * `true` - when the matrices `M1` and `M2` are very similar, then **independendent**.
    /// * `false` - when the matrices `M1` and `M2` are too different, then **dependent**.

    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        cim_1: &Array3<f64>,
        j: usize,
        M2: &Array3<usize>,
        cim_2: &Array3<f64>,
    ) -> bool {
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let cim_1 = cim_1.index_axis(Axis(0), i);
        let cim_2 = cim_2.index_axis(Axis(0), j);
        let r1 = M1.sum_axis(Axis(1));
        let r2 = M2.sum_axis(Axis(1));
        let q1 = cim_1.diag();
        let q2 = cim_2.diag();
        for idx in 0..r1.shape()[0] {
            let s = q2[idx] / q1[idx];
            let F = FisherSnedecor::new(r1[idx], r2[idx]).unwrap();
            let s = F.cdf(s);
            let lim_sx = self.alpha / 2.0;
            let lim_dx = 1.0 - (self.alpha / 2.0);
            if s < lim_sx || s > lim_dx {
                return false;
            }
        }
        true
    }
}

impl HypothesisTest for F {
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> bool
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();
        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;
            if !self.compare_matrices(
                idx_M_small,
                P_small.get_transitions().as_ref().unwrap(),
                P_small.get_cim().as_ref().unwrap(),
                idx_M_big,
                P_big.get_transitions().as_ref().unwrap(),
                P_big.get_cim().as_ref().unwrap(),
            ) {
                return false;
            }
        }
        return true;
    }
}
*/

// try 1 

impl ChiSquare {
    pub fn new(alpha: f64) -> ChiSquare {
        ChiSquare { alpha }
    }

    /// Compare two matrices extracted from two 3rd-orer tensors.
    ///
    /// # Arguments
    ///
    /// * `i` - Position of the matrix of `M1` to compare with `M2`.
    /// * `M1` - 3rd-order tensor 1.
    /// * `j` - Position of the matrix of `M2` to compare with `M1`.
    /// * `M2` - 3rd-order tensor 2.
    ///
    /// # Returns
    ///
    /// * `true` - when the matrices `M1` and `M2` are very similar, then **independendent**.
    /// * `false` - when the matrices `M1` and `M2` are too different, then **dependent**.

    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>,
    ) -> bool {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
        // Also: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let K = M1.sum_axis(Axis(1)) / M2.sum_axis(Axis(1));
        let K = K.mapv(f64::sqrt);
        // Reshape to column vector.
        let K = {
            let n = K.len();
            K.into_shape((n, 1)).unwrap()
        };
        let L = 1.0 / &K;
        let mut X_2 = (&K * &M2 - &L * &M1).mapv(|a| a.powi(2)) / (&M2 + &M1);
        X_2.diag_mut().fill(0.0);
        let X_2 = X_2.sum_axis(Axis(1));
        let n = ChiSquared::new((X_2.dim() - 1) as f64).unwrap();
        let ret = X_2.into_iter().all(|x| n.cdf(x) < (1.0 - (self.alpha / (((M2.shape()[0] * M2.shape()[1]) as f64)))));
        ret
    }
    
/*
    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>,
    ) -> bool {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
        // Also: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let K = M1.sum_axis(Axis(1)) / M2.sum_axis(Axis(1));
        let K = K.mapv(f64::sqrt);
        // Reshape to column vector.
        let K = {
            let n = K.len();
            K.into_shape((n, 1)).unwrap()
        };
        let L = 1.0 / &K;
        let mut X_2 = (&K * &M2 - &L * &M1).mapv(|a| a.powi(2)) / (&M2 + &M1);
        X_2.diag_mut().fill(0.0);
        let X_2 = X_2.sum_axis(Axis(1));
        let n = ChiSquared::new((X_2.dim() - 1) as f64).unwrap();
        let ret = X_2.into_iter().all(|x| n.cdf(x) < (1.0 - self.alpha));
        ret
    }
*//*
    pub fn compute_pvalue_chi_squared( 
        &self,
        i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>,
    ) -> (f64, f64, bool) {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
        // Also: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        //Mapping conteggio parte da alpha 

        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let K = M1.sum_axis(Axis(1)) / M2.sum_axis(Axis(1));
        let K = K.mapv(f64::sqrt);
        // Reshape to column vector.
        let K = {
            let n = K.len();
            K.into_shape((n, 1)).unwrap()
        };
        let L = 1.0 / &K;
        let mut X_2 = (&K * &M2 - &L * &M1).mapv(|a| a.powi(2)) / (&M2 + &M1);
        X_2.diag_mut().fill(0.0);
        let X_2 = X_2.sum_axis(Axis(1));
        let n = ChiSquared::new((X_2.dim() - 1) as f64).unwrap();
        let res: Vec<f64> = X_2.clone().into_iter().map(|x| n.cdf(x)).collect();
        let ret = res.clone().into_iter().all(|x| x < (1.0 - self.alpha));
        let p: f64 = res.iter().sum(); // sum and then we use this to evaluate the mean
        return ((p  as f64), self.alpha, ret);
    }
*/
    pub fn compute_pvalue_chi_squared( 
        &self,
        i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>,
    ) -> (f64, f64, bool) {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
        // Also: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        //Mapping conteggio parte da alpha 

        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let K = M1.sum_axis(Axis(1)) / M2.sum_axis(Axis(1));
        let K = K.mapv(f64::sqrt);
        // Reshape to column vector.
        let K = {
            let n = K.len();
            K.into_shape((n, 1)).unwrap()
        };
        let L = 1.0 / &K;
        let mut X_2 = (&K * &M2 - &L * &M1).mapv(|a| a.powi(2)) / (&M2 + &M1);
        X_2.diag_mut().fill(0.0);
        let X_2 = X_2.sum_axis(Axis(1));
        let n = ChiSquared::new((X_2.dim() - 1) as f64).unwrap();
        let res: Vec<f64> = X_2.clone().into_iter().map(|x| n.cdf(x)).collect();
        let ret = res.clone().into_iter().map(|x| x < (1.0 - (self.alpha / (((M2.shape()[0] * M2.shape()[1]) as f64))))).reduce(|a, b| a & b).unwrap_or(true); 
        let p: f64 = res.iter().sum(); // sum and then we use this to evaluate the mean
        return ((p  as f64), self.alpha, ret);

        
        /*let p: f64 = res.iter().cloned().fold(f64::INFINITY, f64::min);

        return (p, self.alpha, ret);*/
        //println!("res: {:?} chi squared", res);
        /*let p: f64 = res.iter().cloned().fold(0.0_f64, f64::max);
        //println!("p: {:?}", p);
        return (p, self.alpha, ret);*/
    }
}

impl HypothesisTest for ChiSquare {
    
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> bool
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();


        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;
            if !self.compare_matrices(
                idx_M_small,
                P_small.get_transitions().as_ref().unwrap(),
                idx_M_big,
                P_big.get_transitions().as_ref().unwrap(),
            ) {
                return false;
            }
        }
        return true;
    }

/*
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> (f64, bool)
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();


        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;

            let res = self.compute_pvalue_chi_squared(
                idx_M_small,
                P_small.get_transitions().as_ref().unwrap(),
                idx_M_big,
                P_big.get_transitions().as_ref().unwrap(),
            );
            if !res.2{
                return (res.0/self.alpha, false);
            }
        }
        return (0.0, true);
    }
*/
    fn compute_pvalues<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> (f64, f64, bool)
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();

        //let mut min:f64 = f64::INFINITY;
        let mut sum:f64 = 0.0;
        //let mut max = 0.0;
        let mut count:f64 = 0.0;
        let mut alpha:f64 = 0.0;
        let mut all_within_limits = true;

        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;    
            let p_value = self.compute_pvalue_chi_squared(
                            idx_M_small,
                            P_small.get_transitions().as_ref().unwrap(),
                            idx_M_big,
                            P_big.get_transitions().as_ref().unwrap(),
                        );
            alpha = p_value.1; // basta una volta se alpha sempre uguale
            sum += p_value.0;
            /*if min > p_value.0{
                min = p_value.0;
            }*/
            /*if max < p_value.0{
                max = p_value.0;
            }*/

            count += 1.0;
            all_within_limits = all_within_limits && p_value.2;
        }

        return ((sum)/self.alpha, alpha, all_within_limits);
        //return ((min)/self.alpha, alpha, all_within_limits);
        //return ((max)/self.alpha, alpha, all_within_limits);
    }

}



// Original Code!!!!!
/*
impl ChiSquare {
    pub fn new(alpha: f64) -> ChiSquare {
        ChiSquare { alpha }
    }

    /// Compare two matrices extracted from two 3rd-orer tensors.
    ///
    /// # Arguments
    ///
    /// * `i` - Position of the matrix of `M1` to compare with `M2`.
    /// * `M1` - 3rd-order tensor 1.
    /// * `j` - Position of the matrix of `M2` to compare with `M1`.
    /// * `M2` - 3rd-order tensor 2.
    ///
    /// # Returns
    ///
    /// * `true` - when the matrices `M1` and `M2` are very similar, then **independendent**.
    /// * `false` - when the matrices `M1` and `M2` are too different, then **dependent**.

    pub fn compare_matrices(
        &self,
        i: usize,
        M1: &Array3<usize>,
        j: usize,
        M2: &Array3<usize>,
    ) -> bool {
        // Bregoli, A., Scutari, M. and Stella, F., 2021.
        // A constraint-based algorithm for the structural learning of
        // continuous-time Bayesian networks.
        // International Journal of Approximate Reasoning, 138, pp.105-122.
        // Also: https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
        let M1 = M1.index_axis(Axis(0), i).mapv(|x| x as f64);
        let M2 = M2.index_axis(Axis(0), j).mapv(|x| x as f64);
        let K = M1.sum_axis(Axis(1)) / M2.sum_axis(Axis(1));
        let K = K.mapv(f64::sqrt);
        // Reshape to column vector.
        let K = {
            let n = K.len();
            K.into_shape((n, 1)).unwrap()
        };
        let L = 1.0 / &K;
        let mut X_2 = (&K * &M2 - &L * &M1).mapv(|a| a.powi(2)) / (&M2 + &M1);
        X_2.diag_mut().fill(0.0);
        let X_2 = X_2.sum_axis(Axis(1));
        let n = ChiSquared::new((X_2.dim() - 1) as f64).unwrap();
        let ret = X_2.into_iter().all(|x| n.cdf(x) < (1.0 - self.alpha));
        ret
    }
}

impl HypothesisTest for ChiSquare {
    fn call<T, P>(
        &self,
        net: &T,
        child_node: usize,
        parent_node: usize,
        separation_set: &BTreeSet<usize>,
        dataset: &Dataset,
        cache: &mut Cache<P>,
    ) -> bool
    where
        T: process::NetworkProcess,
        P: parameter_learning::ParameterLearning,
    {
        let P_small = match cache.fit(net, &dataset, child_node, Some(separation_set.clone())) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let mut extended_separation_set = separation_set.clone();
        extended_separation_set.insert(parent_node);

        let P_big = match cache.fit(
            net,
            &dataset,
            child_node,
            Some(extended_separation_set.clone()),
        ) {
            Params::DiscreteStatesContinousTime(node) => node,
        };
        let partial_cardinality_product: usize = extended_separation_set
            .iter()
            .take_while(|x| **x != parent_node)
            .map(|x| net.get_node(*x).get_reserved_space_as_parent())
            .product();
        for idx_M_big in 0..P_big.get_transitions().as_ref().unwrap().shape()[0] {
            let idx_M_small: usize = idx_M_big % partial_cardinality_product
                + (idx_M_big
                    / (partial_cardinality_product
                        * net.get_node(parent_node).get_reserved_space_as_parent()))
                    * partial_cardinality_product;
            if !self.compare_matrices(
                idx_M_small,
                P_small.get_transitions().as_ref().unwrap(),
                idx_M_big,
                P_big.get_transitions().as_ref().unwrap(),
            ) {
                return false;
            }
        }
        return true;
    }
}
*/
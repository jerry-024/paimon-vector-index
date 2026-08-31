// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::blas::sgemm_a_bt;
use crate::distance::{fvec_l2sqr, fvec_l2sqr_four, fvec_norm_l2sqr};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::BinaryHeap;

/// Cap aggregate concurrent assignment scratch to ~16MB (4M f32 elements).
const MAX_MATRIX_ELEMS: usize = 4 * 1024 * 1024;
const MIN_BLOCK_ROWS: usize = 32;
const MIN_BLOCK_FLOPS: usize = 4_000_000;
const TARGET_BLOCKS: usize = 64;
const DIRECT_ROWS_PER_THREAD: usize = 8;
const MIN_GEMM_DIM: usize = 32;
const CENTROID_TILE_COLS: usize = 4096;

#[derive(Clone, Copy)]
struct DistanceIndex(f32, usize);

impl PartialEq for DistanceIndex {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for DistanceIndex {}

impl PartialOrd for DistanceIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.cmp(&other.1))
    }
}

fn push_bounded_topk(heap: &mut BinaryHeap<DistanceIndex>, limit: usize, candidate: DistanceIndex) {
    if heap.len() < limit {
        heap.push(candidate);
    } else if heap.peek().is_some_and(|worst| candidate < *worst) {
        *heap.peek_mut().unwrap() = candidate;
    }
}

#[cfg(test)]
std::thread_local! {
    static CERTIFIED_SGEMM_ROWS: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn record_certified_sgemm_rows(rows: usize) {
    CERTIFIED_SGEMM_ROWS.with(|count| {
        if let Some(current) = count.get() {
            count.set(Some(current + rows));
        }
    });
}

fn checked_matrix_len(name: &str, rows: usize, cols: usize) -> usize {
    rows.checked_mul(cols)
        .unwrap_or_else(|| panic!("{name} shape overflows usize"))
}

fn assert_data_shape(data: &[f32], n: usize, d: usize) {
    let expected = checked_matrix_len("data", n, d);
    assert_eq!(data.len(), expected, "data length does not match n * d");
}

fn use_direct_batch(rows: usize, d: usize, threads: usize) -> bool {
    d < MIN_GEMM_DIM || rows < DIRECT_ROWS_PER_THREAD.saturating_mul(threads.max(1))
}

fn direct_topk_linear_worker_limit(k: usize, nprobe: usize, workers: usize) -> usize {
    let tuples = if nprobe <= 1 { 1 } else { k };
    let per_worker = tuples.saturating_mul(topk_tuple_elems());
    if per_worker == 0 {
        return 0;
    }
    (MAX_MATRIX_ELEMS / per_worker).min(workers)
}

fn use_parallel_direct_topk(nq: usize, k: usize, nprobe: usize, threads: usize) -> bool {
    let nprobe = nprobe.min(k);
    let workers = nq.min(threads.max(1)).max(1);
    let tuples = if use_bounded_direct_topk(k, nprobe, workers) {
        nprobe
    } else if nprobe <= 1 {
        1
    } else {
        k
    };
    tuples
        .saturating_mul(topk_tuple_elems())
        .saturating_mul(workers)
        <= MAX_MATRIX_ELEMS
}

fn use_bounded_direct_topk(k: usize, nprobe: usize, threads: usize) -> bool {
    nprobe > 1
        && nprobe < k
        && k.saturating_mul(topk_tuple_elems())
            .saturating_mul(threads.max(1))
            > MAX_MATRIX_ELEMS
}

fn topk_tuple_elems() -> usize {
    std::mem::size_of::<(f32, usize)>().div_ceil(std::mem::size_of::<f32>())
}

fn topk_worker_scratch(k: usize, nprobe: usize) -> usize {
    let tuple_buffers = 1 + usize::from(nprobe > 1 && nprobe < k);
    (if nprobe == 1 { 1 } else { k })
        .saturating_mul(topk_tuple_elems())
        .saturating_mul(tuple_buffers)
}

fn centroid_tile_size(k: usize, d: usize, nprobe: usize, threads: usize) -> Option<usize> {
    let worker_budget = MAX_MATRIX_ELEMS / threads.max(1);
    if nprobe == 0 || nprobe >= k {
        return None;
    }
    let needs_tiling = if nprobe == 1 {
        k.saturating_mul(2) >= worker_budget
    } else {
        topk_worker_scratch(k, nprobe) >= worker_budget
    };
    if !needs_tiling {
        return None;
    }
    let heap_scratch = nprobe.saturating_mul(topk_tuple_elems()).saturating_mul(2);
    let tile = CENTROID_TILE_COLS
        .min(worker_budget.saturating_sub(d.saturating_add(heap_scratch).saturating_add(4)))
        .min(k);
    (tile > 0 && tile < k).then_some(tile)
}

pub(crate) fn assignment_block_plan(
    n: usize,
    d: usize,
    k: usize,
    threads: usize,
    fixed_scratch: usize,
    row_scratch: usize,
) -> (usize, bool) {
    let row_elems = k.max(1).saturating_add(row_scratch);
    let max_rows = (MAX_MATRIX_ELEMS.saturating_sub(fixed_scratch) / row_elems).max(1);
    if threads <= 1 {
        return (max_rows, false);
    }

    let row_flops = k.saturating_mul(d).saturating_mul(2).max(1);
    let min_rows = MIN_BLOCK_ROWS
        .max(MIN_BLOCK_FLOPS.div_ceil(row_flops))
        .min(max_rows);
    let worker_budget = MAX_MATRIX_ELEMS / threads;
    if fixed_scratch >= worker_budget {
        return (max_rows, false);
    }
    let budget_rows = ((worker_budget - fixed_scratch) / row_elems)
        .max(1)
        .min(max_rows);
    if budget_rows < min_rows && budget_rows.saturating_mul(row_flops) < MIN_BLOCK_FLOPS {
        return (budget_rows, false);
    }

    (
        n.div_ceil(TARGET_BLOCKS)
            .max(min_rows)
            .min(max_rows)
            .min(budget_rows),
        true,
    )
}

pub struct KMeansConfig {
    pub niter: usize,
    pub nredo: usize,
    pub max_points_per_centroid: usize,
    pub seed: u64,
    /// Balance factor: penalizes large clusters to produce more uniform partitions.
    /// 0.0 = standard k-means. Higher values = more balanced.
    /// Typical value: 0.1 for IVF construction.
    pub balance_factor: f32,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        KMeansConfig {
            niter: 25,
            nredo: 1,
            max_points_per_centroid: 256,
            seed: 1234,
            balance_factor: 0.0,
        }
    }
}

const EPS: f32 = 1.0 / 1024.0;

/// Threshold above which hierarchical k-means is used.
const HIERARCHICAL_THRESHOLD: usize = 256;

pub fn kmeans_train(config: &KMeansConfig, data: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    assert_data_shape(data, n, d);
    checked_matrix_len("centroid", k, d);
    if k > HIERARCHICAL_THRESHOLD && n > k {
        kmeans_train_hierarchical(config, data, n, d, k)
    } else {
        kmeans_train_with_init(config, data, n, d, k, None)
    }
}

/// Hierarchical k-means for large k (> 256).
/// Starts with initial_k clusters and iteratively splits the largest until target k is reached.
fn kmeans_train_hierarchical(
    config: &KMeansConfig,
    data: &[f32],
    n: usize,
    d: usize,
    target_k: usize,
) -> Vec<f32> {
    use std::cmp::Ordering;

    #[derive(Clone)]
    struct Cluster {
        centroid: Vec<f32>,
        indices: Vec<usize>,
    }

    impl Eq for Cluster {}
    impl PartialEq for Cluster {
        fn eq(&self, other: &Self) -> bool {
            self.indices.len() == other.indices.len()
        }
    }
    impl Ord for Cluster {
        fn cmp(&self, other: &Self) -> Ordering {
            self.indices.len().cmp(&other.indices.len())
        }
    }
    impl PartialOrd for Cluster {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    // Subsample for training
    let max_n = target_k * config.max_points_per_centroid;
    let (train_data, train_n) = if n > max_n {
        let sub = subsample(data, n, d, max_n, &mut rng);
        (sub, max_n)
    } else {
        (data.to_vec(), n)
    };

    // Step 1: Train initial_k clusters
    let initial_k = 16.min(target_k);
    let initial_config = KMeansConfig {
        niter: config.niter,
        seed: config.seed,
        ..KMeansConfig::default()
    };
    let initial_centroids =
        kmeans_train_with_init(&initial_config, &train_data, train_n, d, initial_k, None);

    // Assign all points to initial clusters
    let mut assignments = vec![0usize; train_n];
    assign_clusters_fast(
        &train_data,
        train_n,
        d,
        &initial_centroids,
        initial_k,
        &mut assignments,
        0.0,
    );

    // Build initial clusters
    let mut heap: BinaryHeap<Cluster> = BinaryHeap::new();
    for c in 0..initial_k {
        let indices: Vec<usize> = (0..train_n).filter(|&i| assignments[i] == c).collect();
        let centroid = initial_centroids[c * d..(c + 1) * d].to_vec();
        heap.push(Cluster { centroid, indices });
    }

    // Step 2: Iteratively split the largest cluster.
    let mut finalized: Vec<Vec<f32>> = Vec::new();
    let split_k = 2; // Split into 2 each time

    while finalized.len() + heap.len() < target_k {
        let largest = match heap.pop() {
            Some(cluster) => cluster,
            None => break,
        };

        if largest.indices.len() < split_k * 2 {
            finalized.push(largest.centroid);
            continue;
        }

        let sub_n = largest.indices.len();
        let mut sub_data = vec![0.0f32; sub_n * d];
        for (new_idx, &orig_idx) in largest.indices.iter().enumerate() {
            sub_data[new_idx * d..(new_idx + 1) * d]
                .copy_from_slice(&train_data[orig_idx * d..(orig_idx + 1) * d]);
        }

        let sub_config = KMeansConfig {
            niter: 10,
            seed: config.seed + finalized.len() as u64,
            ..KMeansConfig::default()
        };
        let sub_centroids = kmeans_train_with_init(&sub_config, &sub_data, sub_n, d, split_k, None);

        let mut sub_assignments = vec![0usize; sub_n];
        assign_clusters_fast(
            &sub_data,
            sub_n,
            d,
            &sub_centroids,
            split_k,
            &mut sub_assignments,
            0.0,
        );

        let children: Vec<Cluster> = (0..split_k)
            .filter_map(|sc| {
                let sub_indices: Vec<usize> = (0..sub_n)
                    .filter(|&i| sub_assignments[i] == sc)
                    .map(|i| largest.indices[i])
                    .collect();
                if sub_indices.is_empty() {
                    None
                } else {
                    Some(Cluster {
                        centroid: sub_centroids[sc * d..(sc + 1) * d].to_vec(),
                        indices: sub_indices,
                    })
                }
            })
            .collect();

        // Finalize a degenerate split instead of re-queuing it forever.
        if children.len() < 2 {
            finalized.push(largest.centroid);
        } else {
            for child in children {
                heap.push(child);
            }
        }
    }

    // Collect all centroids
    let mut result = Vec::with_capacity(target_k * d);
    for c in finalized {
        result.extend_from_slice(&c);
    }
    while let Some(cluster) = heap.pop() {
        result.extend_from_slice(&cluster.centroid);
        if result.len() >= target_k * d {
            break;
        }
    }

    // If the hierarchy exhausted before reaching target_k (e.g. highly
    // duplicated data), pad by repeating valid centroids. Zero padding would
    // fabricate origin centroids that exist nowhere in the data.
    if result.len() < target_k * d && !result.is_empty() {
        let produced = result.len() / d;
        for i in produced..target_k {
            let src = (i % produced) * d;
            result.extend_from_within(src..src + d);
        }
    }
    result.resize(target_k * d, 0.0);
    result
}

pub fn kmeans_train_with_init(
    config: &KMeansConfig,
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    initial_centroids: Option<&[f32]>,
) -> Vec<f32> {
    assert_data_shape(data, n, d);
    let centroid_len = checked_matrix_len("centroid", k, d);
    if let Some(init) = initial_centroids {
        assert_eq!(
            init.len(),
            centroid_len,
            "initial_centroids length does not match k * d"
        );
    }
    if n == 0 || k == 0 {
        return vec![0.0; centroid_len];
    }

    let mut rng = StdRng::seed_from_u64(config.seed);

    let max_n = k * config.max_points_per_centroid;
    let (train_data, train_n) = if n > max_n {
        let sub = subsample(data, n, d, max_n, &mut rng);
        (sub, max_n)
    } else {
        (data.to_vec(), n)
    };

    if train_n <= k {
        let mut centroids = vec![0.0f32; centroid_len];
        for i in 0..k {
            let src = i % train_n;
            centroids[i * d..(i + 1) * d].copy_from_slice(&train_data[src * d..(src + 1) * d]);
        }
        return centroids;
    }

    let mut best_centroids = vec![0.0f32; centroid_len];
    let mut best_obj = f32::MAX;

    let nredo = if initial_centroids.is_some() {
        1
    } else {
        config.nredo
    };

    for redo in 0..nredo {
        let mut centroids = if redo == 0 {
            if let Some(init) = initial_centroids {
                init.to_vec()
            } else {
                kmeans_plusplus_init(&train_data, train_n, d, k, &mut rng)
            }
        } else {
            kmeans_plusplus_init(&train_data, train_n, d, k, &mut rng)
        };
        let mut assignments = vec![0usize; train_n];
        let mut prev_obj = f32::MAX;

        for _iter in 0..config.niter {
            let obj = assign_clusters_fast(
                &train_data,
                train_n,
                d,
                &centroids,
                k,
                &mut assignments,
                config.balance_factor,
            );
            update_centroids(
                &train_data,
                train_n,
                d,
                &mut centroids,
                k,
                &assignments,
                &mut rng,
            );

            if prev_obj < f32::MAX {
                let rel_change = (prev_obj - obj).abs() / prev_obj.max(1e-10);
                if rel_change < 1e-6 {
                    break;
                }
            }
            prev_obj = obj;
        }

        if prev_obj < best_obj {
            best_obj = prev_obj;
            best_centroids.copy_from_slice(&centroids);
        }
    }

    best_centroids
}

fn kmeans_plusplus_init(data: &[f32], n: usize, d: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut centroids = vec![0.0f32; k * d];

    let first = rng.gen_range(0..n);
    centroids[..d].copy_from_slice(&data[first * d..(first + 1) * d]);

    let mut min_dists = vec![f32::MAX; n];

    for c in 1..k {
        let prev = &centroids[(c - 1) * d..c * d];
        let mut total = 0.0f32;
        for i in 0..n {
            let dist = fvec_l2sqr(&data[i * d..(i + 1) * d], prev);
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
            total += min_dists[i];
        }

        let target = rng.gen::<f32>() * total;
        let mut cumulative = 0.0f32;
        let mut selected = n - 1;
        for i in 0..n {
            cumulative += min_dists[i];
            if cumulative >= target {
                selected = i;
                break;
            }
        }

        centroids[c * d..(c + 1) * d].copy_from_slice(&data[selected * d..(selected + 1) * d]);
    }

    centroids
}

/// Fast assignment using sgemm: ||x-c||² = ||x||² + ||c||² - 2·x·cᵀ.
/// Supports balance_factor to penalize large clusters.
///
/// balance_factor == 0: rows are processed as independent Rayon blocks. Each
/// row's result does not depend on block boundaries; the objective keeps the
/// historical serial chunk order so Rayon pool sizes reproduce bitwise.
/// balance_factor > 0: keeps the historical serial chunking because cluster
/// size penalties are computed from each chunk's incoming assignments.
fn assign_clusters_fast(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    assignments: &mut [usize],
    balance_factor: f32,
) -> f32 {
    if balance_factor > 0.0 {
        return assign_clusters_balanced_serial(
            data,
            n,
            d,
            centroids,
            k,
            assignments,
            balance_factor,
        );
    }
    if n == 0 {
        return 0.0;
    }
    if d == 0 {
        // Degenerate dimension: every distance is zero. Matches the serial
        // path instead of panicking in par_chunks(0).
        assignments.fill(0);
        return 0.0;
    }

    let c_norms: Vec<f32> = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect();

    let max_rows = (MAX_MATRIX_ELEMS / k.max(1)).max(1);
    let (block_rows, parallel) = assignment_block_plan(n, d, k, rayon::current_num_threads(), 0, 0);
    if n <= block_rows {
        return assign_block(data, n, d, centroids, k, &c_norms, assignments, &mut []);
    } else if !parallel && block_rows == max_rows {
        return data
            .chunks(block_rows * d)
            .zip(assignments.chunks_mut(block_rows))
            .map(|(block_data, block_assign)| {
                assign_block(
                    block_data,
                    block_assign.len(),
                    d,
                    centroids,
                    k,
                    &c_norms,
                    block_assign,
                    &mut [],
                )
            })
            .sum();
    }

    let mut row_objs = vec![0.0f32; n];
    if parallel {
        data.par_chunks(block_rows * d)
            .zip(assignments.par_chunks_mut(block_rows))
            .zip(row_objs.par_chunks_mut(block_rows))
            .for_each(|((block_data, block_assign), block_objs)| {
                assign_block(
                    block_data,
                    block_assign.len(),
                    d,
                    centroids,
                    k,
                    &c_norms,
                    block_assign,
                    block_objs,
                );
            });
    } else {
        data.chunks(block_rows * d)
            .zip(assignments.chunks_mut(block_rows))
            .zip(row_objs.chunks_mut(block_rows))
            .for_each(|((block_data, block_assign), block_objs)| {
                assign_block(
                    block_data,
                    block_assign.len(),
                    d,
                    centroids,
                    k,
                    &c_norms,
                    block_assign,
                    block_objs,
                );
            });
    }

    row_objs
        .chunks(max_rows)
        .map(|chunk| chunk.iter().sum::<f32>())
        .sum()
}

/// Assign one row block: sgemm inner products + per-row argmin.
fn assign_block(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    c_norms: &[f32],
    assignments: &mut [usize],
    row_objs: &mut [f32],
) -> f32 {
    let mut ip_matrix = vec![0.0f32; n * k];
    sgemm_a_bt(n, k, d, 1.0, data, centroids, 0.0, &mut ip_matrix);

    let mut objective = 0.0f32;
    for i in 0..n {
        let x_norm = fvec_norm_l2sqr(&data[i * d..(i + 1) * d]);
        let mut best = 0;
        let mut best_dist = f32::MAX;
        let row = i * k;
        for c in 0..k {
            let dist = x_norm + c_norms[c] - 2.0 * ip_matrix[row + c];
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        assignments[i] = best;
        if row_objs.is_empty() {
            objective += best_dist;
        } else {
            row_objs[i] = best_dist;
        }
    }
    objective
}

/// Historical serial path for balance_factor > 0. Chunk boundaries are part of
/// the observable behavior: cluster sizes are recomputed from each chunk's
/// incoming assignments.
fn assign_clusters_balanced_serial(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    assignments: &mut [usize],
    balance_factor: f32,
) -> f32 {
    if n * k > MAX_MATRIX_ELEMS {
        let chunk_n = MAX_MATRIX_ELEMS / k;
        let mut total_obj = 0.0f32;
        let mut offset = 0;
        while offset < n {
            let cn = (n - offset).min(chunk_n);
            total_obj += assign_clusters_balanced_serial(
                &data[offset * d..(offset + cn) * d],
                cn,
                d,
                centroids,
                k,
                &mut assignments[offset..offset + cn],
                balance_factor,
            );
            offset += cn;
        }
        return total_obj;
    }

    let x_norms: Vec<f32> = (0..n)
        .map(|i| fvec_norm_l2sqr(&data[i * d..(i + 1) * d]))
        .collect();
    let c_norms: Vec<f32> = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect();

    let mut ip_matrix = vec![0.0f32; n * k];
    sgemm_a_bt(n, k, d, 1.0, data, centroids, 0.0, &mut ip_matrix);

    // Compute cluster sizes for balance penalty
    let mut cluster_sizes = vec![0u32; k];
    for &a in assignments.iter() {
        if a < k {
            cluster_sizes[a] += 1;
        }
    }

    let mut total_obj = 0.0f32;
    for i in 0..n {
        let mut best = 0;
        let mut best_dist = f32::MAX;
        let row = i * k;
        for c in 0..k {
            let mut dist = x_norms[i] + c_norms[c] - 2.0 * ip_matrix[row + c];
            // Balance penalty: prefer smaller clusters
            if cluster_sizes[c] > 0 {
                dist += balance_factor * (cluster_sizes[c] as f32).ln();
            }
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        assignments[i] = best;
        total_obj += best_dist;
    }

    total_obj
}

fn update_centroids(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &mut [f32],
    k: usize,
    assignments: &[usize],
    rng: &mut StdRng,
) {
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * d];

    for i in 0..n {
        let c = assignments[i];
        counts[c] += 1;
        for j in 0..d {
            sums[c * d + j] += data[i * d + j];
        }
    }

    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                centroids[c * d + j] = sums[c * d + j] * inv;
            }
        }
    }

    for c in 0..k {
        if counts[c] > 0 {
            continue;
        }

        let donor = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &cnt)| cnt)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if counts[donor] <= 1 {
            let idx = rng.gen_range(0..n);
            centroids[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
            continue;
        }

        let donor_copy: Vec<f32> = centroids[donor * d..(donor + 1) * d].to_vec();
        centroids[c * d..(c + 1) * d].copy_from_slice(&donor_copy);

        for j in 0..d {
            if j.is_multiple_of(2) {
                centroids[c * d + j] *= 1.0 + EPS;
                centroids[donor * d + j] *= 1.0 - EPS;
            } else {
                centroids[c * d + j] *= 1.0 - EPS;
                centroids[donor * d + j] *= 1.0 + EPS;
            }
        }

        counts[c] = counts[donor] / 2;
        counts[donor] -= counts[c];
    }
}

pub fn find_nearest(point: &[f32], centroids: &[f32], k: usize, d: usize) -> usize {
    let mut best = 0;
    let mut best_dist = f32::MAX;
    let four_end = k / 4 * 4;
    for c in (0..four_end).step_by(4) {
        let distances = fvec_l2sqr_four(
            point,
            &centroids[c * d..(c + 1) * d],
            &centroids[(c + 1) * d..(c + 2) * d],
            &centroids[(c + 2) * d..(c + 3) * d],
            &centroids[(c + 3) * d..(c + 4) * d],
        );
        for (offset, dist) in distances.into_iter().enumerate() {
            if dist < best_dist {
                best_dist = dist;
                best = c + offset;
            }
        }
    }
    for c in four_end..k {
        let dist = fvec_l2sqr(point, &centroids[c * d..(c + 1) * d]);
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

pub(crate) fn find_nearest_batch(
    data: &[f32],
    n: usize,
    centroids: &[f32],
    k: usize,
    d: usize,
) -> Vec<usize> {
    if n == 0 || k == 0 {
        return Vec::new();
    }
    if use_direct_batch(n, d, rayon::current_num_threads()) {
        return (0..n)
            .into_par_iter()
            .map(|i| find_nearest(&data[i * d..(i + 1) * d], centroids, k, d))
            .collect();
    }
    let c_norms: Vec<f32> = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect();
    let mut out = vec![k; n];
    certified_topk_blocks(
        data,
        n,
        centroids,
        &c_norms,
        k,
        d,
        1,
        &mut out,
        |slot, top| {
            if top[0].0 < f32::MAX {
                *slot = top[0].1;
            }
        },
    );
    for (i, assignment) in out.iter_mut().enumerate().filter(|(_, c)| **c == k) {
        *assignment = find_nearest(&data[i * d..(i + 1) * d], centroids, k, d);
    }
    out
}

/// Batch coarse search with the contract of `find_topk` (direct `fvec_l2sqr`
/// distances, ties by centroid index) at the cost of blocked SGEMM.
///
/// Rows are processed in SGEMM blocks sized by `assignment_block_plan`, in
/// parallel when the plan allows. Before the GEMM every row is screened: a
/// row whose norm or centroid norms are not finite, or whose GEMM error band
/// `2·E_gemm` is not smaller than its direct distance to the centroid mean
/// (data translated far from the origin, where `|x|² + |c|² - 2x·c` cancels
/// catastrophically), is scanned directly instead. A block with no other rows
/// skips the GEMM entirely. Each remaining row goes through
/// `certified_topk_row`; `emit` receives every row's top list in its slot.
#[allow(clippy::too_many_arguments)]
fn certified_topk_blocks<T: Send>(
    data: &[f32],
    n: usize,
    centroids: &[f32],
    c_norms: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
    out: &mut [T],
    emit: impl Fn(&mut T, &[(f32, usize)]) + Sync,
) {
    let c_max = centroid_norm_upper(c_norms, d);
    let mut mean = vec![0.0f64; d];
    for row in centroids[..k * d].chunks_exact(d) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += *v as f64;
        }
    }
    let mean: Vec<f32> = mean.iter().map(|m| (m / k as f64) as f32).collect();
    let threads = rayon::current_num_threads();
    if let Some(tile_cols) = centroid_tile_size(k, d, nprobe, threads) {
        return certified_topk_blocks_tiled(
            data, n, centroids, c_norms, c_max, &mean, k, d, nprobe, tile_cols, out, &emit,
        );
    }
    let topk_scratch = topk_worker_scratch(k, nprobe);
    let (block_rows, parallel) = assignment_block_plan(n, d, k, threads, topk_scratch, 2);
    let direct_workers = if parallel {
        n.div_ceil(block_rows).min(threads).max(1)
    } else {
        1
    };
    let block = |(b, chunk): (usize, &mut [T])| {
        let start = b * block_rows;
        let rows = chunk.len();
        let x = &data[start * d..(start + rows) * d];
        let mut x_norms = vec![0.0f32; rows];
        let mut use_gemm = vec![false; rows];
        let mut safe_rows = 0;
        for i in 0..rows {
            let x_i = &x[i * d..(i + 1) * d];
            x_norms[i] = fvec_norm_l2sqr(x_i);
            let e_gemm = gemm_error(x_norms[i], c_max, d);
            let to_mean = fvec_l2sqr(x_i, &mean) as f64;
            // `!(a < b)` also catches NaN on either side.
            use_gemm[i] = 2.0 * e_gemm < to_mean;
            safe_rows += usize::from(use_gemm[i]);
        }
        let mut dists: Vec<(f32, usize)> = Vec::with_capacity(if nprobe == 1 { 1 } else { k });
        if safe_rows == rows {
            let mut ip = vec![0.0f32; rows * k];
            #[cfg(test)]
            record_certified_sgemm_rows(rows);
            sgemm_a_bt(rows, k, d, 1.0, x, centroids, 0.0, &mut ip);
            for (i, slot) in chunk.iter_mut().enumerate() {
                let x_i = &x[i * d..(i + 1) * d];
                let top = certified_topk_row(
                    x_i,
                    x_norms[i],
                    &ip[i * k..(i + 1) * k],
                    centroids,
                    c_norms,
                    c_max,
                    d,
                    nprobe,
                    direct_workers,
                    &mut dists,
                );
                emit(slot, top);
            }
            return;
        }

        for (i, slot) in chunk.iter_mut().enumerate() {
            if !use_gemm[i] {
                let x_i = &x[i * d..(i + 1) * d];
                let (ids, distances) =
                    find_topk_with_workers(x_i, centroids, k, d, nprobe, direct_workers);
                dists.clear();
                dists.extend(distances.into_iter().zip(ids));
                emit(slot, &dists);
            }
        }
        if safe_rows == 0 {
            return;
        }

        let live_scratch = topk_scratch.saturating_add(rows.saturating_mul(2));
        let packed_rows = assignment_block_plan(
            safe_rows,
            d,
            k,
            if parallel {
                rayon::current_num_threads()
            } else {
                1
            },
            live_scratch,
            d.saturating_add(2),
        )
        .0;
        let mut indices = Vec::with_capacity(packed_rows);
        let mut packed = Vec::with_capacity(packed_rows * d);
        let mut ip = Vec::with_capacity(packed_rows * k);
        let mut process_safe = |indices: &[usize], packed: &[f32]| {
            let batch_rows = indices.len();
            ip.resize(batch_rows * k, 0.0);
            #[cfg(test)]
            record_certified_sgemm_rows(batch_rows);
            sgemm_a_bt(batch_rows, k, d, 1.0, packed, centroids, 0.0, &mut ip);
            for (packed_row, &row) in indices.iter().enumerate() {
                let x_i = &x[row * d..(row + 1) * d];
                let top = certified_topk_row(
                    x_i,
                    x_norms[row],
                    &ip[packed_row * k..(packed_row + 1) * k],
                    centroids,
                    c_norms,
                    c_max,
                    d,
                    nprobe,
                    direct_workers,
                    &mut dists,
                );
                emit(&mut chunk[row], top);
            }
        };
        for (row, &safe) in use_gemm.iter().enumerate() {
            if !safe {
                continue;
            }
            indices.push(row);
            packed.extend_from_slice(&x[row * d..(row + 1) * d]);
            if indices.len() == packed_rows {
                process_safe(&indices, &packed);
                indices.clear();
                packed.clear();
            }
        }
        if !indices.is_empty() {
            process_safe(&indices, &packed);
        }
    };
    if parallel {
        out.par_chunks_mut(block_rows).enumerate().for_each(block);
    } else {
        out.chunks_mut(block_rows).enumerate().for_each(block);
    }
}

#[allow(clippy::too_many_arguments)]
fn certified_topk_blocks_tiled<T: Send>(
    data: &[f32],
    n: usize,
    centroids: &[f32],
    c_norms: &[f32],
    c_max: f64,
    mean: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
    tile_cols: usize,
    out: &mut [T],
    emit: &(impl Fn(&mut T, &[(f32, usize)]) + Sync),
) {
    let threads = rayon::current_num_threads();
    let heap_scratch = nprobe.saturating_mul(topk_tuple_elems()).saturating_mul(2);
    let (block_rows, planned_parallel) = assignment_block_plan(
        n,
        d,
        tile_cols,
        threads,
        0,
        d.saturating_add(heap_scratch).saturating_add(4),
    );
    let block_flops = block_rows
        .saturating_mul(k)
        .saturating_mul(d)
        .saturating_mul(2);
    let parallel = planned_parallel || (threads > 1 && block_flops >= MIN_BLOCK_FLOPS);
    let direct_workers = if parallel {
        n.div_ceil(block_rows).min(threads).max(1)
    } else {
        1
    };
    let block = |(b, chunk): (usize, &mut [T])| {
        let start = b * block_rows;
        let rows = chunk.len();
        let x = &data[start * d..(start + rows) * d];
        let mut x_norms = vec![0.0f32; rows];
        let mut gemm_errors = vec![0.0f64; rows];
        let mut safe_indices = Vec::with_capacity(rows);
        let mut direct = Vec::new();
        for (row, slot) in chunk.iter_mut().enumerate() {
            let x_i = &x[row * d..(row + 1) * d];
            x_norms[row] = fvec_norm_l2sqr(x_i);
            let e_gemm = gemm_error(x_norms[row], c_max, d);
            gemm_errors[row] = e_gemm;
            let to_mean = fvec_l2sqr(x_i, mean) as f64;
            if 2.0 * e_gemm < to_mean {
                safe_indices.push(row);
            } else {
                let (ids, distances) =
                    find_topk_with_workers(x_i, centroids, k, d, nprobe, direct_workers);
                direct.clear();
                direct.extend(distances.into_iter().zip(ids));
                emit(slot, &direct);
            }
        }
        if safe_indices.is_empty() {
            return;
        }

        let mut packed = Vec::with_capacity(safe_indices.len() * d);
        for &row in &safe_indices {
            packed.extend_from_slice(&x[row * d..(row + 1) * d]);
        }
        let mut states: Vec<_> = (0..safe_indices.len())
            .map(|_| {
                (
                    BinaryHeap::with_capacity(nprobe),
                    BinaryHeap::with_capacity(nprobe),
                    true,
                )
            })
            .collect();
        let mut ip = Vec::with_capacity(safe_indices.len() * tile_cols);
        for tile_start in (0..k).step_by(tile_cols) {
            let tile_len = tile_cols.min(k - tile_start);
            ip.resize(safe_indices.len() * tile_len, 0.0);
            #[cfg(test)]
            record_certified_sgemm_rows(safe_indices.len());
            sgemm_a_bt(
                safe_indices.len(),
                tile_len,
                d,
                1.0,
                &packed,
                &centroids[tile_start * d..(tile_start + tile_len) * d],
                0.0,
                &mut ip,
            );
            for (packed_row, &row) in safe_indices.iter().enumerate() {
                let (approximate_top, exact_top, valid) = &mut states[packed_row];
                if !*valid {
                    continue;
                }
                let inner_products = &ip[packed_row * tile_len..(packed_row + 1) * tile_len];
                for (offset, &inner_product) in inner_products.iter().enumerate() {
                    let c = tile_start + offset;
                    let approximate = x_norms[row] + c_norms[c] - 2.0 * inner_product;
                    if !approximate.is_finite() {
                        *valid = false;
                        break;
                    }
                    push_bounded_topk(
                        approximate_top,
                        nprobe,
                        DistanceIndex(approximate.max(0.0), c),
                    );
                }
                if !*valid {
                    continue;
                }
                let upper = if approximate_top.len() < nprobe {
                    f64::INFINITY
                } else {
                    let kth = approximate_top.peek().unwrap().0 as f64;
                    kth + 2.0 * certified_error(gemm_errors[row], kth, d)
                };
                let x_i = &x[row * d..(row + 1) * d];
                // The upper bound only shrinks, so discarded candidates cannot re-enter later.
                for (offset, &inner_product) in inner_products.iter().enumerate() {
                    let c = tile_start + offset;
                    let approximate = (x_norms[row] + c_norms[c] - 2.0 * inner_product).max(0.0);
                    if approximate as f64 <= upper {
                        push_bounded_topk(
                            exact_top,
                            nprobe,
                            DistanceIndex(fvec_l2sqr(x_i, &centroids[c * d..(c + 1) * d]), c),
                        );
                    }
                }
            }
        }

        for ((approximate_top, exact_top, valid), row) in states.into_iter().zip(safe_indices) {
            let slot = &mut chunk[row];
            if !valid || approximate_top.len() < nprobe || exact_top.len() < nprobe {
                let x_i = &x[row * d..(row + 1) * d];
                let (ids, distances) =
                    find_topk_with_workers(x_i, centroids, k, d, nprobe, direct_workers);
                direct.clear();
                direct.extend(distances.into_iter().zip(ids));
                emit(slot, &direct);
                continue;
            }
            let top: Vec<_> = exact_top
                .into_sorted_vec()
                .into_iter()
                .map(|candidate| (candidate.0, candidate.1))
                .collect();
            emit(slot, &top);
        }
    };
    if parallel {
        out.par_chunks_mut(block_rows).enumerate().for_each(block);
    } else {
        out.chunks_mut(block_rows).enumerate().for_each(block);
    }
}

fn find_top1(point: &[f32], centroids: &[f32], k: usize, d: usize) -> (f32, usize) {
    let mut best = None;
    let mut consider = |candidate: (f32, usize)| {
        if best
            .as_ref()
            .is_none_or(|current| compare_distance_then_index(&candidate, current).is_lt())
        {
            best = Some(candidate);
        }
    };
    let four_end = k / 4 * 4;
    for c in (0..four_end).step_by(4) {
        let distances = fvec_l2sqr_four(
            point,
            &centroids[c * d..(c + 1) * d],
            &centroids[(c + 1) * d..(c + 2) * d],
            &centroids[(c + 2) * d..(c + 3) * d],
            &centroids[(c + 3) * d..(c + 4) * d],
        );
        for (offset, distance) in distances.into_iter().enumerate() {
            consider((distance, c + offset));
        }
    }
    for c in four_end..k {
        consider((fvec_l2sqr(point, &centroids[c * d..(c + 1) * d]), c));
    }
    best.expect("top-1 requires at least one centroid")
}

pub fn find_topk(
    point: &[f32],
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
) -> (Vec<usize>, Vec<f32>) {
    find_topk_with_workers(point, centroids, k, d, nprobe, 1)
}

#[inline]
fn find_topk_with_workers(
    point: &[f32],
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
    workers: usize,
) -> (Vec<usize>, Vec<f32>) {
    let nprobe = nprobe.min(k);
    if nprobe == 0 {
        return (Vec::new(), Vec::new());
    }
    if nprobe == 1 {
        let (distance, index) = find_top1(point, centroids, k, d);
        return (vec![index], vec![distance]);
    }
    find_topk_multiple(point, centroids, k, d, nprobe, workers)
}

fn find_topk_multiple(
    point: &[f32],
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
    workers: usize,
) -> (Vec<usize>, Vec<f32>) {
    let bounded = use_bounded_direct_topk(k, nprobe, workers);
    let mut dists = Vec::with_capacity(if bounded { 0 } else { k });
    let mut heap = BinaryHeap::with_capacity(if bounded { nprobe } else { 0 });
    let mut consider = |candidate: (f32, usize)| {
        if bounded {
            push_bounded_topk(&mut heap, nprobe, DistanceIndex(candidate.0, candidate.1));
        } else {
            dists.push(candidate);
        }
    };
    let four_end = k / 4 * 4;
    for c in (0..four_end).step_by(4) {
        let distances = fvec_l2sqr_four(
            point,
            &centroids[c * d..(c + 1) * d],
            &centroids[(c + 1) * d..(c + 2) * d],
            &centroids[(c + 2) * d..(c + 3) * d],
            &centroids[(c + 3) * d..(c + 4) * d],
        );
        for (offset, distance) in distances.into_iter().enumerate() {
            consider((distance, c + offset));
        }
    }
    for c in four_end..k {
        consider((fvec_l2sqr(point, &centroids[c * d..(c + 1) * d]), c));
    }
    if bounded {
        let top = heap.into_sorted_vec();
        return (
            top.iter().map(|candidate| candidate.1).collect(),
            top.iter().map(|candidate| candidate.0).collect(),
        );
    }
    select_topk_prefix(&mut dists, nprobe);
    let indices: Vec<usize> = dists[..nprobe].iter().map(|&(_, i)| i).collect();
    let distances: Vec<f32> = dists[..nprobe].iter().map(|&(d, _)| d).collect();
    (indices, distances)
}

fn find_topk_batch_direct(
    queries: &[f32],
    nq: usize,
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let threads = rayon::current_num_threads();
    let requested_workers = nq.min(threads).max(1);
    // Keep the faster linear selection and run excess queries in bounded waves
    // instead of switching algorithms when nq crosses the scratch limit.
    let linear_workers = direct_topk_linear_worker_limit(k, nprobe, requested_workers);
    if linear_workers > 0 {
        let wave = requested_workers.min(linear_workers);
        let mut out = vec![(Vec::new(), Vec::new()); nq];
        for start in (0..nq).step_by(wave) {
            let end = (start + wave).min(nq);
            let query_chunk = &queries[start * d..end * d];
            out[start..end]
                .par_iter_mut()
                .enumerate()
                .for_each(|(qi, slot)| {
                    *slot = find_topk_with_workers(
                        &query_chunk[qi * d..(qi + 1) * d],
                        centroids,
                        k,
                        d,
                        nprobe,
                        wave,
                    );
                });
        }
        return out.into_iter().unzip();
    }

    let parallel = use_parallel_direct_topk(nq, k, nprobe, threads);
    let workers = if parallel { requested_workers } else { 1 };
    let search = |qi| {
        find_topk_with_workers(
            &queries[qi * d..(qi + 1) * d],
            centroids,
            k,
            d,
            nprobe,
            workers,
        )
    };
    if parallel {
        (0..nq).into_par_iter().map(search).unzip()
    } else {
        (0..nq).map(search).unzip()
    }
}

/// Batch find top-nprobe nearest centroids for multiple queries.
/// Returns (all_indices, all_distances) each of length nq * nprobe.
pub fn find_topk_batch(
    queries: &[f32],
    nq: usize,
    centroids: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    if use_direct_batch(nq, d, rayon::current_num_threads()) {
        return find_topk_batch_direct(queries, nq, centroids, k, d, nprobe);
    }
    let centroid_norms = (0..k)
        .map(|c| fvec_norm_l2sqr(&centroids[c * d..(c + 1) * d]))
        .collect::<Vec<_>>();
    find_topk_batch_with_centroid_norms(queries, nq, centroids, &centroid_norms, k, d, nprobe)
}

pub(crate) fn find_topk_batch_with_centroid_norms(
    queries: &[f32],
    nq: usize,
    centroids: &[f32],
    centroid_norms: &[f32],
    k: usize,
    d: usize,
    nprobe: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    debug_assert_eq!(centroid_norms.len(), k);
    let nprobe = nprobe.min(k);
    if nprobe == 0 {
        return (vec![Vec::new(); nq], vec![Vec::new(); nq]);
    }
    if nprobe == k || use_direct_batch(nq, d, rayon::current_num_threads()) {
        return find_topk_batch_direct(queries, nq, centroids, k, d, nprobe);
    }
    let mut out: Vec<(Vec<usize>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); nq];
    certified_topk_blocks(
        queries,
        nq,
        centroids,
        centroid_norms,
        k,
        d,
        nprobe,
        &mut out,
        |slot, top| {
            slot.0.extend(top.iter().map(|&(_, i)| i));
            slot.1.extend(top.iter().map(|&(dist, _)| dist));
        },
    );
    out.into_iter().unzip()
}

/// Upper bound on the largest centroid norm from the f32 squared norms.
fn centroid_norm_upper(c_norms: &[f32], d: usize) -> f64 {
    let max = c_norms.iter().copied().fold(0.0f32, f32::max) as f64;
    ((max + 2.0 * d as f64 * f32::MIN_POSITIVE as f64) / (1.0 - gamma_f32(d))).sqrt()
}

/// Bound on `|fl(|x|² + |c|² - 2 x·c) - |x - c|²|` for a row of f32 squared
/// norm `x_norm` and centroids of norm at most `c_max`: the two norms and the
/// inner product each carry the standard `γ_d` dot-product rounding, the two
/// combining operations one rounding each. The absolute terms cover gradual
/// underflow and FTZ; certification assumes DAZ is disabled (Rust's default).
fn gemm_error(x_norm: f32, c_max: f64, d: usize) -> f64 {
    let min_normal = f32::MIN_POSITIVE as f64;
    let x_up = ((x_norm as f64 + 2.0 * d as f64 * min_normal) / (1.0 - gamma_f32(d))).sqrt();
    let sum = x_up + c_max;
    gamma_f32(d.saturating_add(2)) * sum * sum
        + d.saturating_mul(8).saturating_add(4) as f64 * min_normal
}

fn gamma_f32(operations: usize) -> f64 {
    let error = (f32::EPSILON as f64 / 2.0) * operations as f64;
    if error < 1.0 {
        error / (1.0 - error)
    } else {
        f64::INFINITY
    }
}

fn certified_error(e_gemm: f64, kth: f64, d: usize) -> f64 {
    let gamma = gamma_f32(d.saturating_add(3));
    let denominator = 1.0 - 2.0 * gamma;
    if denominator > 0.0 {
        let direct_absolute_error =
            d.saturating_mul(3).saturating_add(1) as f64 * f32::MIN_POSITIVE as f64;
        (e_gemm * (1.0 + gamma) + gamma * kth + direct_absolute_error) / denominator
    } else {
        f64::INFINITY
    }
}

/// Direct distances of four centroids at a time (same accumulation order as
/// `fvec_l2sqr`, so the result equals `find_topk`'s bit for bit).
fn direct_distances(x: &[f32], centroids: &[f32], d: usize, slots: &mut [(f32, usize)]) {
    let (chunks, remainder) = slots.as_chunks_mut::<4>();
    for group in chunks {
        let dists = fvec_l2sqr_four(
            x,
            &centroids[group[0].1 * d..(group[0].1 + 1) * d],
            &centroids[group[1].1 * d..(group[1].1 + 1) * d],
            &centroids[group[2].1 * d..(group[2].1 + 1) * d],
            &centroids[group[3].1 * d..(group[3].1 + 1) * d],
        );
        for (slot, dist) in group.iter_mut().zip(dists) {
            slot.0 = dist;
        }
    }
    for slot in remainder {
        slot.0 = fvec_l2sqr(x, &centroids[slot.1 * d..(slot.1 + 1) * d]);
    }
}

/// Top-`nprobe` centroids of one row from its SGEMM inner products, with the
/// contract of `find_topk` (direct `fvec_l2sqr` distances, ties by index).
///
/// With `E` chosen so
/// `E = gemm_error + γ(d+3)·(D + 2E + gemm_error) + (3d+1)·MIN_POSITIVE`
/// and `D` the `nprobe`-th SGEMM distance, every candidate through `D + 2E`
/// is within `E` of its direct distance: a centroid whose SGEMM distance is
/// below `D - 2E` is in the direct top-`nprobe` set and one above `D + 2E`
/// is not. When nothing outside the SGEMM top-`nprobe` lies within `D + 2E`
/// the set is certified as is (the common case); otherwise the direct kernel
/// decides the centroids inside `[D - 2E, D + 2E]`, typically one or two.
/// The selected centroids are then re-measured with the direct kernel and
/// ordered by (distance, index), so the returned values equal `find_topk`.
#[allow(clippy::too_many_arguments)]
fn certified_topk_row<'a>(
    x: &[f32],
    x_norm: f32,
    ip: &[f32],
    centroids: &[f32],
    c_norms: &[f32],
    c_max: f64,
    d: usize,
    nprobe: usize,
    direct_workers: usize,
    dists: &'a mut Vec<(f32, usize)>,
) -> &'a [(f32, usize)] {
    let k = c_norms.len();
    let e_gemm = gemm_error(x_norm, c_max, d);
    dists.clear();
    if nprobe == 1 && k > 1 {
        // Argmin pass tracking the runner-up value (no scratch, no select).
        let mut best = f32::INFINITY;
        let mut best_idx = 0;
        let mut second = f32::INFINITY;
        for c in 0..k {
            let approximate = x_norm + c_norms[c] - 2.0 * ip[c];
            if !approximate.is_finite() {
                let (ids, distances) =
                    find_topk_with_workers(x, centroids, k, d, nprobe, direct_workers);
                dists.extend(distances.into_iter().zip(ids));
                return dists;
            }
            let dist = approximate.max(0.0);
            if dist < best {
                second = best;
                best = dist;
                best_idx = c;
            } else if dist < second {
                second = dist;
            }
        }
        let upper = best as f64 + 2.0 * certified_error(e_gemm, best as f64, d);
        let mut top = (f32::INFINITY, best_idx);
        if second as f64 > upper {
            top.0 = fvec_l2sqr(x, &centroids[best_idx * d..(best_idx + 1) * d]);
        } else {
            top.1 = usize::MAX;
            for c in 0..k {
                let dist = (x_norm + c_norms[c] - 2.0 * ip[c]).max(0.0);
                if dist as f64 <= upper {
                    let direct = fvec_l2sqr(x, &centroids[c * d..(c + 1) * d]);
                    if direct < top.0 || (direct == top.0 && c < top.1) {
                        top = (direct, c);
                    }
                }
            }
        }
        dists.push(top);
        return &dists[..];
    }
    for c in 0..k {
        let approximate = x_norm + c_norms[c] - 2.0 * ip[c];
        if !approximate.is_finite() {
            let (ids, distances) =
                find_topk_with_workers(x, centroids, k, d, nprobe, direct_workers);
            dists.clear();
            dists.extend(distances.into_iter().zip(ids));
            return dists;
        }
        dists.push((approximate.max(0.0), c));
    }
    if nprobe >= k {
        direct_distances(x, centroids, d, dists);
        dists.sort_unstable_by(compare_distance_then_index);
        return &dists[..];
    }
    // Partition so [..nprobe] holds the nprobe smallest and [nprobe] the next.
    dists.select_nth_unstable_by(nprobe, compare_distance_then_index);
    let kth = dists[..nprobe]
        .iter()
        .map(|&(dist, _)| dist)
        .fold(0.0f32, f32::max) as f64;
    let next = dists[nprobe].0 as f64;
    let e = certified_error(e_gemm, kth, d);
    let upper = kth + 2.0 * e;
    if next > upper {
        direct_distances(x, centroids, d, &mut dists[..nprobe]);
        dists[..nprobe].sort_unstable_by(compare_distance_then_index);
        return &dists[..nprobe];
    }
    let lower = kth - 2.0 * e;
    // Members below `lower` are certainly in; move them to the front.
    let mut certain = 0;
    for i in 0..nprobe {
        if (dists[i].0 as f64) < lower {
            dists.swap(certain, i);
            certain += 1;
        }
    }
    // Everything in [lower, upper] is decided by the direct kernel.
    let mut band = Vec::with_capacity(dists.len() - certain);
    band.extend(
        dists[certain..]
            .iter()
            .filter(|&&(dist, _)| (dist as f64) <= upper)
            .copied(),
    );
    direct_distances(x, centroids, d, &mut band);
    let slots = nprobe - certain;
    debug_assert!(band.len() >= slots);
    select_topk_prefix(&mut band, slots);
    dists[certain..nprobe].copy_from_slice(&band[..slots]);
    direct_distances(x, centroids, d, &mut dists[..certain]);
    dists[..nprobe].sort_unstable_by(compare_distance_then_index);
    &dists[..nprobe]
}

fn select_topk_prefix(dists: &mut [(f32, usize)], nprobe: usize) {
    debug_assert!(nprobe > 0 && nprobe <= dists.len());
    if nprobe < dists.len() {
        dists.select_nth_unstable_by(nprobe - 1, compare_distance_then_index);
    }
    // Centroid indices make this comparator a total order, so stable sorting
    // is unnecessary and would allocate additional scratch space.
    dists[..nprobe].sort_unstable_by(compare_distance_then_index);
}

fn compare_distance_then_index(left: &(f32, usize), right: &(f32, usize)) -> std::cmp::Ordering {
    left.0
        .total_cmp(&right.0)
        .then_with(|| left.1.cmp(&right.1))
}

// --- Streaming Coreset K-means ---

/// Streaming k-means trainer for very large datasets.
/// Processes data in chunks, compresses each chunk into a weighted coreset,
/// then trains final centroids on the accumulated coreset.
pub struct StreamingKMeans {
    pub d: usize,
    pub k: usize,
    pub chunk_size: usize,
    config: KMeansConfig,
    /// Accumulated coreset: (centroids, weights)
    coreset_centroids: Vec<f32>,
    coreset_weights: Vec<f32>,
}

impl StreamingKMeans {
    /// Create a streaming k-means trainer.
    /// chunk_size: number of vectors per chunk (e.g., k * 256)
    pub fn new(d: usize, k: usize, chunk_size: usize, config: KMeansConfig) -> Self {
        StreamingKMeans {
            d,
            k,
            chunk_size,
            config,
            coreset_centroids: Vec::new(),
            coreset_weights: Vec::new(),
        }
    }

    /// Feed a chunk of training data. Can be called multiple times.
    /// Each chunk is compressed into k weighted centroids (coreset).
    pub fn add_chunk(&mut self, data: &[f32], n: usize) {
        let d = self.d;
        let chunk_k = self.k.min(n);

        if chunk_k == 0 || n == 0 {
            return;
        }

        // Train k-means on this chunk
        let chunk_config = KMeansConfig {
            niter: 15,
            seed: self.config.seed + self.coreset_weights.len() as u64,
            ..KMeansConfig::default()
        };
        let centroids = kmeans_train_with_init(&chunk_config, data, n, d, chunk_k, None);

        // Assign points to centroids to compute weights
        let mut assignments = vec![0usize; n];
        assign_clusters_fast(data, n, d, &centroids, chunk_k, &mut assignments, 0.0);

        let mut weights = vec![0.0f32; chunk_k];
        for &a in &assignments {
            weights[a] += 1.0;
        }

        // Append to coreset
        self.coreset_centroids.extend_from_slice(&centroids);
        self.coreset_weights.extend_from_slice(&weights);
    }

    /// Finalize: train final centroids on the accumulated weighted coreset.
    pub fn finalize(&self) -> Vec<f32> {
        let d = self.d;
        let coreset_n = self.coreset_weights.len();

        if coreset_n == 0 {
            return vec![0.0f32; self.k * d];
        }

        if coreset_n <= self.k {
            let mut result = self.coreset_centroids.clone();
            result.resize(self.k * d, 0.0);
            return result;
        }

        // Weighted k-means on coreset
        weighted_kmeans_train(
            &self.config,
            &self.coreset_centroids,
            &self.coreset_weights,
            coreset_n,
            d,
            self.k,
        )
    }

    /// Total vectors processed so far.
    pub fn total_weight(&self) -> f32 {
        self.coreset_weights.iter().sum()
    }
}

/// Weighted k-means: each point has a weight that affects centroid computation.
fn weighted_kmeans_train(
    config: &KMeansConfig,
    data: &[f32],
    weights: &[f32],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(config.seed);

    if n <= k {
        let mut centroids = vec![0.0f32; k * d];
        for i in 0..k {
            let src = i % n;
            centroids[i * d..(i + 1) * d].copy_from_slice(&data[src * d..(src + 1) * d]);
        }
        return centroids;
    }

    let mut centroids = kmeans_plusplus_init(data, n, d, k, &mut rng);
    let mut assignments = vec![0usize; n];

    for _iter in 0..config.niter {
        // Assign (unweighted distance)
        assign_clusters_fast(data, n, d, &centroids, k, &mut assignments, 0.0);

        // Update with weights
        let mut sums = vec![0.0f32; k * d];
        let mut total_weights = vec![0.0f32; k];

        for i in 0..n {
            let c = assignments[i];
            let w = weights[i];
            total_weights[c] += w;
            for j in 0..d {
                sums[c * d + j] += w * data[i * d + j];
            }
        }

        for c in 0..k {
            if total_weights[c] > 0.0 {
                let inv = 1.0 / total_weights[c];
                for j in 0..d {
                    centroids[c * d + j] = sums[c * d + j] * inv;
                }
            } else {
                // Reinit empty cluster
                let idx = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[idx * d..(idx + 1) * d]);
            }
        }
    }

    centroids
}

fn subsample(data: &[f32], n: usize, d: usize, target_n: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..target_n {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let mut result = vec![0.0f32; target_n * d];
    for (out_i, &src_i) in indices[..target_n].iter().enumerate() {
        result[out_i * d..(out_i + 1) * d].copy_from_slice(&data[src_i * d..(src_i + 1) * d]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certified_error_covers_band_upper_bound() {
        let d = 768;
        let e_gemm = gamma_f32(d + 2) * 4.0;
        let kth = 0.1;
        let gamma = gamma_f32(d + 3);
        let error = certified_error(e_gemm, kth, d);
        let upper = kth + 2.0 * error;

        let direct_absolute_error = (3 * d + 1) as f64 * f32::MIN_POSITIVE as f64;
        assert!(error >= e_gemm + gamma * (upper + e_gemm) + direct_absolute_error);
    }

    #[test]
    fn test_certified_error_falls_back_when_fixed_point_diverges() {
        assert!(certified_error(1.0, 1.0, 6_000_000).is_infinite());
    }

    #[test]
    fn test_subnormal_batch_topk_matches_direct() {
        let query = [f32::from_bits(2646992242)];
        let centroids = [
            497491591, 451116127, 476312835, 503292106, 505618023, 2646911911, 2618864963,
            2647061976, 483230981, 500562856, 502683715, 504768900, 2632103736, 2648519606,
            2634666709, 2629586502,
        ]
        .map(f32::from_bits);
        let queries = query.repeat(8);
        let (batch, batch_distances) =
            pool(1).install(|| find_topk_batch(&queries, 8, &centroids, 16, 1, 1));
        let (direct, direct_distances) = find_topk(&query, &centroids, 16, 1, 1);

        assert_eq!(direct, [5]);
        assert_eq!(batch, vec![direct; 8]);
        assert_eq!(batch_distances, vec![direct_distances; 8]);
    }

    #[test]
    fn test_subnormal_batch_topk_band_matches_direct() {
        let query = [f32::from_bits(500581713)];
        let centroids = [
            495350057, 2636893317, 2635975005, 2651790365, 500667861, 2643010036, 495349473,
            488134825, 487899297, 502064699, 2649052619, 2645746487, 506974333, 2651576280,
            2645106766, 487307224,
        ]
        .map(f32::from_bits);
        let queries = query.repeat(8);
        let (batch, batch_distances) =
            pool(1).install(|| find_topk_batch(&queries, 8, &centroids, 16, 1, 3));
        let (direct, direct_distances) = find_topk(&query, &centroids, 16, 1, 3);

        assert_eq!(direct, [4, 9, 0]);
        assert_eq!(batch, vec![direct; 8]);
        assert_eq!(batch_distances, vec![direct_distances; 8]);
    }

    /// Sequential scalar reference for cluster assignment. Mirrors the
    /// balance-penalty semantics of a single (unchunked) call.
    fn assign_clusters_reference(
        data: &[f32],
        n: usize,
        d: usize,
        centroids: &[f32],
        k: usize,
        assignments: &mut [usize],
        balance_factor: f32,
    ) -> f32 {
        let mut cluster_sizes = vec![0u32; k];
        if balance_factor > 0.0 {
            for &a in assignments.iter() {
                if a < k {
                    cluster_sizes[a] += 1;
                }
            }
        }
        let mut total_obj = 0.0f32;
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let mut dist =
                    fvec_l2sqr(&data[i * d..(i + 1) * d], &centroids[c * d..(c + 1) * d]);
                if balance_factor > 0.0 && cluster_sizes[c] > 0 {
                    dist += balance_factor * (cluster_sizes[c] as f32).ln();
                }
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            assignments[i] = best;
            total_obj += best_dist;
        }
        total_obj
    }

    fn deterministic_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    #[test]
    fn test_assign_clusters_matches_reference_shapes() {
        // Shapes chosen to cover: tiny, uneven final row block, and the
        // chunked path (n * k > MAX_MATRIX_ELEMS).
        let shapes: &[(usize, usize, usize)] = &[(17, 4, 5), (1003, 7, 9), (70_000, 64, 8)];
        for &(n, k, d) in shapes {
            let data = deterministic_data(n, d, 7);
            let centroids = deterministic_data(k, d, 11);

            let mut fast = vec![0usize; n];
            let obj_fast = assign_clusters_fast(&data, n, d, &centroids, k, &mut fast, 0.0);

            let mut reference = vec![0usize; n];
            let obj_ref =
                assign_clusters_reference(&data, n, d, &centroids, k, &mut reference, 0.0);

            assert_eq!(
                fast, reference,
                "assignments diverge for shape ({n},{k},{d})"
            );
            let rel = (obj_fast - obj_ref).abs() / obj_ref.max(1e-10);
            assert!(
                rel < 1e-5,
                "objective rel err {rel} for shape ({n},{k},{d})"
            );
        }
    }

    #[test]
    fn test_assign_clusters_target_boundary_shape() {
        // The target coarse assignment shape n=244606, k=16 sits just below
        // MAX_MATRIX_ELEMS. Use a small d instead of 768 to keep memory low.
        let (n, k, d) = (244_606, 16, 4);
        let data = deterministic_data(n, d, 13);
        let centroids = deterministic_data(k, d, 17);

        let mut fast = vec![0usize; n];
        let obj_fast = assign_clusters_fast(&data, n, d, &centroids, k, &mut fast, 0.0);

        let mut reference = vec![0usize; n];
        let obj_ref = assign_clusters_reference(&data, n, d, &centroids, k, &mut reference, 0.0);

        assert_eq!(fast, reference);
        let rel = (obj_fast - obj_ref).abs() / obj_ref.max(1e-10);
        assert!(rel < 1e-5, "objective rel err {rel}");
    }

    #[test]
    fn test_assign_clusters_cross_thread_objective() {
        // Fixed block boundaries make assignments and the objective
        // bitwise reproducible across Rayon pool sizes.
        for &(n, k, d) in &[
            (1003usize, 7usize, 9usize),
            (70_000, 64, 8),
            (244_606, 16, 4),
            (524_289, 2, 1),
        ] {
            let data = deterministic_data(n, d, 19);
            let centroids = deterministic_data(k, d, 23);

            let mut a1 = vec![0usize; n];
            let obj1 =
                pool(1).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut a1, 0.0));

            let mut a8 = vec![0usize; n];
            let obj8 =
                pool(8).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut a8, 0.0));

            assert_eq!(a1, a8, "assignments diverge across pools for ({n},{k},{d})");
            assert_eq!(
                obj1.to_bits(),
                obj8.to_bits(),
                "objective diverges across pools for ({n},{k},{d})"
            );
        }
    }

    #[test]
    fn test_parallel_assignment_respects_aggregate_scratch_budget() {
        let (rows, parallel) = assignment_block_plan(262_144, 768, 1024, 16, 0, 0);
        assert!(parallel);
        assert_eq!(rows, 256);
        assert!(rows * 1024 * 16 <= MAX_MATRIX_ELEMS);

        let (rows, parallel) = assignment_block_plan(2_000_000, 1, 2, 16, 0, 0);
        assert!(!parallel);
        assert_eq!(rows, 131_072);
        assert!(rows * 2 * 16 <= MAX_MATRIX_ELEMS);

        let (rows, parallel) = assignment_block_plan(100_000, 32, 4096, 32, 0, 0);
        assert!(parallel);
        assert_eq!(rows, 32);
        assert!(rows * 4096 * 32 <= MAX_MATRIX_ELEMS);

        let (rows, parallel) = assignment_block_plan(100_000, 32, 4096, 32, 4096 * 6, 0);
        assert!(parallel);
        assert_eq!(rows, 26);
        assert!(rows * 4096 + 4096 * 6 <= MAX_MATRIX_ELEMS / 32);

        let fixed = topk_worker_scratch(4096, 8);
        let (rows, parallel) = assignment_block_plan(100_000, 32, 4096, 32, fixed, 2);
        assert!(parallel);
        assert_eq!(rows, 23);
        assert!(rows * (4096 + 2) + fixed <= MAX_MATRIX_ELEMS / 32);

        let live_scratch = fixed + rows * 2;
        let (packed_rows, packed_parallel) =
            assignment_block_plan(rows, 32, 4096, 32, live_scratch, 32 + 2);
        assert!(packed_parallel);
        assert!(packed_rows * (4096 + 32 + 2) + live_scratch <= MAX_MATRIX_ELEMS / 32);
    }

    #[test]
    fn test_tiled_certified_topk_matches_direct() {
        let (n, k, d, nprobe) = (32, 32_769, 2, 2);
        let centroids = deterministic_data(k, d, 41);
        let queries = deterministic_data(n, d, 43);
        let centroid_norms: Vec<_> = centroids.chunks(d).map(fvec_norm_l2sqr).collect();
        let mut actual = vec![(Vec::new(), Vec::new()); n];

        pool(32).install(|| {
            assert_eq!(centroid_tile_size(k, d, nprobe, 32), Some(4096));
            certified_topk_blocks(
                &queries,
                n,
                &centroids,
                &centroid_norms,
                k,
                d,
                nprobe,
                &mut actual,
                |slot, top| {
                    slot.0.extend(top.iter().map(|&(_, index)| index));
                    slot.1.extend(top.iter().map(|&(distance, _)| distance));
                },
            );
        });

        for row in 0..n {
            let expected = find_topk(&queries[row * d..(row + 1) * d], &centroids, k, d, nprobe);
            assert_eq!(actual[row].0, expected.0, "row {row} indices differ");
            assert_eq!(
                actual[row]
                    .1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .1
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "row {row} distances differ"
            );
        }
    }

    #[test]
    fn test_tiled_certified_top1_matches_direct() {
        let (n, k, d, nprobe) = (4, 65_537, 2, 1);
        let centroids = deterministic_data(k, d, 47);
        let queries = deterministic_data(n, d, 53);
        let centroid_norms: Vec<_> = centroids.chunks(d).map(fvec_norm_l2sqr).collect();
        let mut actual = vec![(Vec::new(), Vec::new()); n];

        pool(32).install(|| {
            assert_eq!(centroid_tile_size(k, d, nprobe, 32), Some(4096));
            certified_topk_blocks(
                &queries,
                n,
                &centroids,
                &centroid_norms,
                k,
                d,
                nprobe,
                &mut actual,
                |slot, top| {
                    slot.0.extend(top.iter().map(|&(_, index)| index));
                    slot.1.extend(top.iter().map(|&(distance, _)| distance));
                },
            );
        });

        for row in 0..n {
            let expected = find_topk(&queries[row * d..(row + 1) * d], &centroids, k, d, nprobe);
            assert_eq!(actual[row].0, expected.0, "row {row} indices differ");
            assert_eq!(
                actual[row].1[0].to_bits(),
                expected.1[0].to_bits(),
                "row {row} distance differs"
            );
        }
    }

    #[test]
    fn test_small_batch_direct_cutoff() {
        assert!(use_direct_batch(31, 32, 4));
        assert!(!use_direct_batch(32, 32, 4));
        assert!(use_direct_batch(32, 8, 4));

        let d = 32;
        let centroids = [-1.0f32; 64];
        let centroid_norms = [32.0, 32.0];
        let queries = vec![10.0; 8 * d];
        let (direct_rows, blocked_rows) = pool(1).install(|| {
            CERTIFIED_SGEMM_ROWS.with(|count| count.set(Some(0)));
            find_topk_batch_with_centroid_norms(
                &queries[..7 * d],
                7,
                &centroids,
                &centroid_norms,
                2,
                d,
                1,
            );
            let direct_rows = CERTIFIED_SGEMM_ROWS.with(|count| count.get().unwrap());
            find_topk_batch_with_centroid_norms(&queries, 8, &centroids, &centroid_norms, 2, d, 1);
            let blocked_rows = CERTIFIED_SGEMM_ROWS.with(|count| count.replace(None).unwrap());
            (direct_rows, blocked_rows)
        });
        assert_eq!(direct_rows, 0);
        assert_eq!(blocked_rows, 8);
    }

    #[test]
    fn test_assign_clusters_preserves_serial_chunk_objective() {
        let (n, k, d) = (70_000usize, 64usize, 8usize);
        let data = deterministic_data(n, d, 27);
        let centroids = deterministic_data(k, d, 29);

        let mut fast = vec![0usize; n];
        let fast_obj =
            pool(8).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut fast, 0.0));

        let c_norms: Vec<f32> = centroids.chunks(d).map(fvec_norm_l2sqr).collect();
        let max_rows = MAX_MATRIX_ELEMS / k;
        let mut serial = vec![0usize; n];
        let mut serial_rows = vec![0.0f32; n];
        let serial_obj: f32 = data
            .chunks(max_rows * d)
            .zip(serial.chunks_mut(max_rows))
            .zip(serial_rows.chunks_mut(max_rows))
            .map(|((block_data, block_assign), block_objs)| {
                assign_block(
                    block_data,
                    block_assign.len(),
                    d,
                    &centroids,
                    k,
                    &c_norms,
                    block_assign,
                    block_objs,
                );
                block_objs.iter().sum::<f32>()
            })
            .sum();

        assert_eq!(fast, serial);
        assert_eq!(fast_obj.to_bits(), serial_obj.to_bits());
    }

    #[test]
    fn test_split_training_shape_uses_single_sgemm() {
        let (n, k, d) = (512usize, 2usize, 768usize);
        let data = deterministic_data(n, d, 29);
        let centroids = deterministic_data(k, d, 31);

        let mut fast = vec![0usize; n];
        let fast_obj =
            pool(8).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut fast, 0.0));

        let c_norms: Vec<f32> = centroids.chunks(d).map(fvec_norm_l2sqr).collect();
        let mut single = vec![0usize; n];
        let mut single_rows = vec![0.0f32; n];
        assign_block(
            &data,
            n,
            d,
            &centroids,
            k,
            &c_norms,
            &mut single,
            &mut single_rows,
        );
        let single_obj: f32 = single_rows.iter().sum();

        assert_eq!(fast, single);
        assert_eq!(fast_obj.to_bits(), single_obj.to_bits());
    }

    #[test]
    fn test_small_assignments_bitwise_reproducible_across_pools() {
        let (n, k, d) = (256usize, 2usize, 64usize);
        let data = deterministic_data(n, d, 47);
        let centroids = deterministic_data(k, d, 53);

        let mut a1 = vec![0usize; n];
        let obj1 =
            pool(1).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut a1, 0.0));

        let mut a8 = vec![0usize; n];
        let obj8 =
            pool(8).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut a8, 0.0));

        assert_eq!(a1, a8);
        assert_eq!(obj1.to_bits(), obj8.to_bits());
    }

    #[test]
    fn test_assign_clusters_balanced_stays_serial() {
        // balance_factor > 0 keeps the serial chunked path, so results must be
        // bitwise identical regardless of the Rayon pool size. n*k exceeds
        // MAX_MATRIX_ELEMS to exercise the chunk boundaries.
        let (n, k, d) = (70_000usize, 64usize, 8usize);
        let data = deterministic_data(n, d, 29);
        let centroids = deterministic_data(k, d, 31);
        let seed_assign: Vec<usize> = (0..n).map(|i| i % k).collect();

        let mut a1 = seed_assign.clone();
        let obj1 =
            pool(1).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut a1, 0.1));

        let mut a8 = seed_assign.clone();
        let obj8 =
            pool(8).install(|| assign_clusters_fast(&data, n, d, &centroids, k, &mut a8, 0.1));

        assert_eq!(a1, a8);
        assert_eq!(obj1.to_bits(), obj8.to_bits());
    }

    #[test]
    fn test_assign_clusters_bitwise_reproducible_fixed_pool() {
        // Same data, seed, and pool size must reproduce k-means bitwise.
        let n = 3000;
        let d = 6;
        let data = deterministic_data(n, d, 37);
        let config = KMeansConfig::default();

        let run = || {
            pool(4).install(|| {
                let flat = kmeans_train_with_init(&config, &data, n, d, 24, None);
                let hier = kmeans_train(&config, &data, n, d, 300);
                let mut assignments = vec![0usize; n];
                let obj = assign_clusters_fast(&data, n, d, &flat, 24, &mut assignments, 0.0);
                (flat, hier, assignments, obj)
            })
        };

        let (flat_a, hier_a, assign_a, obj_a) = run();
        let (flat_b, hier_b, assign_b, obj_b) = run();

        assert!(flat_a
            .iter()
            .zip(&flat_b)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(hier_a
            .iter()
            .zip(&hier_b)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(assign_a, assign_b);
        assert_eq!(obj_a.to_bits(), obj_b.to_bits());
    }

    #[test]
    fn test_two_clusters() {
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(0.1);
            data.push(0.1);
        }
        for _ in 0..50 {
            data.push(10.1);
            data.push(10.1);
        }

        let config = KMeansConfig::default();
        let centroids = kmeans_train(&config, &data, 100, 2, 2);

        let c0 = if centroids[0] < 5.0 {
            &centroids[0..2]
        } else {
            &centroids[2..4]
        };
        let c1 = if centroids[0] < 5.0 {
            &centroids[2..4]
        } else {
            &centroids[0..2]
        };

        assert!(c0[0] < 2.0 && c0[1] < 2.0);
        assert!(c1[0] > 8.0 && c1[1] > 8.0);
    }

    #[test]
    fn test_find_topk() {
        let centroids = [0.0, 0.0, 10.0, 0.0, 5.0, 5.0];
        let query = [1.0, 1.0];
        let (indices, _) = find_topk(&query, &centroids, 3, 2, 2);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_find_top1_without_full_scratch_preserves_direct_order() {
        let centroids = [1.0, 0.0, -1.0, 0.0, 2.0, 0.0];
        let query = [0.0, 0.0];

        assert_eq!(find_top1(&query, &centroids, 3, 2), (1.0, 0));
    }

    #[test]
    fn test_direct_topk_parallelism_respects_aggregate_scratch_budget() {
        assert!(use_parallel_direct_topk(32, 1_048_576, 1, 32));
        assert!(use_parallel_direct_topk(32, 1_048_576, 2, 32));
        assert!(use_parallel_direct_topk(2, 524_288, 2, 32));
        assert!(!use_parallel_direct_topk(32, 1_048_576, 1_048_575, 32));
    }

    #[test]
    fn test_direct_topk_linear_workers_are_capped_without_switching_algorithm() {
        let k = 262_144;
        let nprobe = 65_536;
        assert_eq!(direct_topk_linear_worker_limit(k, nprobe, 4), 4);
        assert_eq!(direct_topk_linear_worker_limit(k, nprobe, 5), 4);
        assert!(!use_bounded_direct_topk(k, nprobe, 4));
        assert!(use_bounded_direct_topk(k, nprobe, 5));
    }

    #[test]
    fn test_direct_topk_waves_support_zero_dimension() {
        assert_eq!(
            find_topk_batch(&[], 3, &[], 0, 0, 1),
            (vec![Vec::new(); 3], vec![Vec::new(); 3])
        );
    }

    #[test]
    fn test_bounded_direct_topk_preserves_distance_then_index_order() {
        let k = 32_769;
        let mut centroids: Vec<f32> = (0..k).map(|i| i as f32).collect();
        centroids[k - 1] = 1.0;

        let actual = pool(32).install(|| find_topk(&[0.0], &centroids, k, 1, 2));

        assert!(use_bounded_direct_topk(k, 2, 32));
        assert_eq!(actual, (vec![0, 1], vec![0.0, 1.0]));
    }

    #[test]
    fn test_kmeans_rejects_invalid_data_shapes_before_early_return() {
        let config = KMeansConfig::default();

        let short =
            std::panic::catch_unwind(|| kmeans_train_with_init(&config, &[0.0; 3], 2, 2, 0, None));
        let long = std::panic::catch_unwind(|| kmeans_train(&config, &[0.0; 5], 2, 2, 0));
        let overflow = std::panic::catch_unwind(|| {
            kmeans_train_with_init(&config, &[], usize::MAX, 2, 0, None)
        });

        assert!(short.is_err());
        assert!(long.is_err());
        assert!(overflow.is_err());
    }

    #[test]
    fn test_kmeans_rejects_invalid_centroid_shapes_before_early_return() {
        let config = KMeansConfig::default();

        let short = std::panic::catch_unwind(|| {
            kmeans_train_with_init(&config, &[], 0, 2, 1, Some(&[0.0]))
        });
        let long = std::panic::catch_unwind(|| {
            kmeans_train_with_init(&config, &[], 0, 2, 1, Some(&[0.0; 3]))
        });
        let overflow = std::panic::catch_unwind(|| {
            kmeans_train_with_init(&config, &[], 0, 2, usize::MAX, None)
        })
        .unwrap_err();
        let overflow_message = overflow
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| overflow.downcast_ref::<&str>().copied())
            .unwrap_or_default();

        assert!(short.is_err());
        assert!(long.is_err());
        assert!(overflow_message.contains("centroid shape overflows usize"));
    }

    #[test]
    fn test_find_topk_batch_matches_full_sort_with_ties() {
        let d = 2;
        let k = 32;
        let nprobe = 5;
        let centroids: Vec<f32> = (0..k)
            .flat_map(|i| [i as f32 % 4.0, (i / 4) as f32])
            .collect();
        let queries = vec![0.5, 0.5, 2.5, 3.5, 1.5, 1.5];
        let (actual_indices, actual_distances) =
            find_topk_batch(&queries, 3, &centroids, k, d, nprobe);

        for qi in 0..3 {
            let query = &queries[qi * d..(qi + 1) * d];
            let mut expected: Vec<(f32, usize)> = (0..k)
                .map(|ci| (fvec_l2sqr(query, &centroids[ci * d..(ci + 1) * d]), ci))
                .collect();
            expected.sort_by(compare_distance_then_index);
            assert_eq!(
                actual_indices[qi],
                expected[..nprobe]
                    .iter()
                    .map(|&(_, index)| index)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                actual_distances[qi],
                expected[..nprobe]
                    .iter()
                    .map(|&(distance, _)| distance)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_find_nearest_batch_matches_scalar() {
        let d = 5;
        let k = 4;
        let n = 17;
        let centroids: Vec<f32> = (0..k * d).map(|i| i as f32 * 0.25 - 2.0).collect();
        let data: Vec<f32> = (0..n * d)
            .map(|i| ((i * 13 % 29) as f32) * 0.1 - 1.0)
            .collect();

        let batch = find_nearest_batch(&data, n, &centroids, k, d);
        let scalar: Vec<usize> = (0..n)
            .map(|i| find_nearest(&data[i * d..(i + 1) * d], &centroids, k, d))
            .collect();

        assert_eq!(batch, scalar);
    }

    #[test]
    fn test_batch_distance_matches_scalar_with_large_translation() {
        let (d, k, n) = (16, 2, 2);
        let mut centroids = vec![1e8; k * d];
        centroids[0] += 8.0;
        let data = vec![1e8; n * d];

        assert_eq!(find_nearest(&data[..d], &centroids, k, d), 1);
        assert_eq!(find_nearest_batch(&data, n, &centroids, k, d), vec![1; n]);
        let (indices, distances) = find_topk_batch(&data, n, &centroids, k, d, 1);
        assert_eq!(indices, vec![vec![1]; n]);
        assert_eq!(distances, vec![vec![0.0]; n]);
    }

    #[test]
    fn test_hot_start_converges_faster() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 500;
        let d = 4;
        let k = 4;

        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>() * 10.0).collect();

        let config = KMeansConfig {
            niter: 25,
            ..KMeansConfig::default()
        };
        let centroids = kmeans_train(&config, &data, n, d, k);

        // Hot-start with previous centroids should converge in fewer iterations
        let config2 = KMeansConfig {
            niter: 3,
            ..KMeansConfig::default()
        };
        let centroids2 = kmeans_train_with_init(&config2, &data, n, d, k, Some(&centroids));

        // Should be very close to the original since it started from converged state
        let mut total_diff = 0.0f32;
        for i in 0..k * d {
            total_diff += (centroids[i] - centroids2[i]).abs();
        }
        assert!(
            total_diff < 1.0,
            "Hot-start centroids drifted too much: {}",
            total_diff
        );
    }

    #[test]
    fn test_streaming_coreset_kmeans() {
        let n = 5000;
        let d = 4;
        let k = 10;
        let chunk_size = 1000;

        let mut rng = StdRng::seed_from_u64(42);
        // Generate clustered data
        let mut data = Vec::new();
        for cluster in 0..k {
            let cx = cluster as f32 * 20.0;
            let cy = cluster as f32 * 20.0;
            for _ in 0..n / k {
                data.push(cx + rng.gen::<f32>() * 2.0);
                data.push(cy + rng.gen::<f32>() * 2.0);
                data.push(rng.gen::<f32>());
                data.push(rng.gen::<f32>());
            }
        }

        let config = KMeansConfig::default();
        let mut streaming = StreamingKMeans::new(d, k, chunk_size, config);

        // Feed data in chunks
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk_n = chunk_end - chunk_start;
            streaming.add_chunk(&data[chunk_start * d..chunk_end * d], chunk_n);
        }

        assert!((streaming.total_weight() - n as f32).abs() < 1.0);

        let centroids = streaming.finalize();
        assert_eq!(centroids.len(), k * d);

        // Centroids should be diverse
        let first = &centroids[0..d];
        let mut diverse = false;
        for i in 1..k {
            if fvec_l2sqr(&centroids[i * d..(i + 1) * d], first) > 1.0 {
                diverse = true;
                break;
            }
        }
        assert!(diverse, "Streaming centroids are not diverse");
    }

    #[test]
    fn test_hierarchical_exact_k() {
        // Requested k must be returned exactly, including non-power-of-two k.
        let d = 4;
        let n = 4000;
        let data = deterministic_data(n, d, 41);
        let config = KMeansConfig::default();
        for &k in &[257usize, 1000, 1024] {
            let centroids = kmeans_train(&config, &data, n, d, k);
            assert_eq!(centroids.len(), k * d, "wrong centroid count for k={k}");
            for &v in &centroids {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_hierarchical_deterministic_same_seed() {
        let d = 4;
        let n = 3000;
        let data = deterministic_data(n, d, 43);
        let config = KMeansConfig::default();
        let a = kmeans_train(&config, &data, n, d, 300);
        let b = kmeans_train(&config, &data, n, d, 300);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn test_hierarchical_strict_largest_first_fixture() {
        let d = 2;
        let k = 257;
        let mut data = Vec::new();

        for i in 0..1024 {
            data.push((i % 32) as f32 * 0.01);
            data.push((i / 32) as f32 * 0.01);
        }
        for cluster in 1..16 {
            for i in 0..64 {
                data.push(cluster as f32 * 1000.0 + (i % 8) as f32 * 0.01);
                data.push((i / 8) as f32 * 0.01);
            }
        }

        let centroids = kmeans_train(&KMeansConfig::default(), &data, data.len() / d, d, k);
        let largest_cluster_centroids = centroids
            .chunks_exact(d)
            .filter(|centroid| centroid[0] < 500.0)
            .count();

        // Strict pop/split/reinsert assigns 232 centroids here; batched parent
        // pops assign 234 and therefore change the trained index.
        assert_eq!(
            largest_cluster_centroids, 232,
            "hierarchical split order changed"
        );
    }

    #[test]
    fn test_hierarchical_tiny_split_candidates() {
        // Highly duplicated data creates tiny/empty split candidates; the
        // hierarchy must still return exactly k centroids.
        let d = 4;
        let k = 300;
        let n = 600;
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            let v = (i % 5) as f32;
            for j in 0..d {
                data[i * d + j] = v;
            }
        }
        let config = KMeansConfig::default();
        let centroids = kmeans_train(&config, &data, n, d, k);
        assert_eq!(centroids.len(), k * d);
        for &v in &centroids {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_hierarchical_duplicate_data_pads_with_valid_centroids() {
        // All-duplicate non-zero data exhausts the split hierarchy early.
        // Padding must repeat valid centroids, never fabricate zeros.
        let d = 4;
        let k = 300;
        let n = 2000;
        let data = vec![10.0f32; n * d];
        let config = KMeansConfig::default();
        let centroids = kmeans_train(&config, &data, n, d, k);
        assert_eq!(centroids.len(), k * d);
        for c in 0..k {
            let row = &centroids[c * d..(c + 1) * d];
            // Empty-cluster handling perturbs donors by ±EPS, so allow a small
            // relative band around 10.0; zero padding would land far outside.
            assert!(
                row.iter().all(|&v| (9.0..=11.0).contains(&v)),
                "centroid {c} is not derived from the data: {row:?}"
            );
        }
    }

    #[test]
    fn test_assign_clusters_zero_dimension_does_not_panic() {
        let n = 5;
        let k = 3;
        let mut assignments = vec![7usize; n];
        let obj = assign_clusters_fast(&[], n, 0, &[], k, &mut assignments, 0.0);
        assert_eq!(assignments, vec![0usize; n]);
        assert_eq!(obj, 0.0);
    }

    #[test]
    fn test_hierarchical_kmeans() {
        let n = 2000;
        let d = 4;
        let k = 300; // > 256, triggers hierarchical

        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>() * 100.0).collect();

        let config = KMeansConfig::default();
        let centroids = kmeans_train(&config, &data, n, d, k);

        assert_eq!(centroids.len(), k * d);

        // All centroids should be finite
        for &v in &centroids {
            assert!(v.is_finite(), "Non-finite centroid value: {}", v);
        }

        // Centroids should be diverse (not all the same)
        let first = &centroids[0..d];
        let mut all_same = true;
        for i in 1..k {
            if &centroids[i * d..(i + 1) * d] != first {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "All centroids are identical");
    }

    #[test]
    fn test_batch_falls_back_when_approximate_distance_overflows() {
        let scale = f32::MAX.sqrt();
        let query = scale * 0.80f32.sqrt();
        let centroids = [scale * 0.19f32.sqrt(), scale * 0.30f32.sqrt()];
        let queries = [query, query];

        assert_eq!(find_nearest_batch(&queries, 2, &centroids, 2, 1), [1, 1]);
    }

    #[test]
    fn test_non_finite_nearest_batch_matches_scalar() {
        let d = MIN_GEMM_DIM;
        let query = vec![f32::INFINITY; d];
        let mut centroids = vec![f32::INFINITY; 2 * d];
        centroids[d..].fill(0.0);
        let expected = find_nearest(&query, &centroids, 2, d);

        assert_eq!(
            pool(1).install(|| find_nearest_batch(&query.repeat(8), 8, &centroids, 2, d)),
            vec![expected; 8]
        );
    }

    #[test]
    fn test_batch_topk_all_centroids_skips_sgemm() {
        let d = MIN_GEMM_DIM;
        let k = 64;
        let nq = DIRECT_ROWS_PER_THREAD;
        let mut centroids = vec![-1.0; k * d];
        centroids[k / 2 * d..].fill(1.0);
        let query = vec![10.0; d];
        let expected = find_topk(&query, &centroids, k, d, k);

        let ((indices, distances), gemm_rows) = pool(1).install(|| {
            CERTIFIED_SGEMM_ROWS.with(|count| count.set(Some(0)));
            let result = find_topk_batch(&query.repeat(nq), nq, &centroids, k, d, k);
            let rows = CERTIFIED_SGEMM_ROWS.with(|count| count.replace(None).unwrap());
            (result, rows)
        });

        assert_eq!(indices, vec![expected.0; nq]);
        assert_eq!(distances, vec![expected.1; nq]);
        assert_eq!(gemm_rows, 0);
    }

    #[test]
    fn test_mixed_block_sgemm_only_processes_safe_rows() {
        let d = MIN_GEMM_DIM;
        let mut centroids = vec![-1.0; 2 * d];
        centroids[d..].fill(1.0);
        let mut queries = vec![10.0; 32 * d];
        for row in (1..32).step_by(2) {
            queries[row * d] = f32::NAN;
        }

        let (nearest, gemm_rows) = pool(1).install(|| {
            CERTIFIED_SGEMM_ROWS.with(|count| count.set(Some(0)));
            let nearest = find_nearest_batch(&queries, 32, &centroids, 2, d);
            let rows = CERTIFIED_SGEMM_ROWS.with(|count| count.replace(None).unwrap());
            (nearest, rows)
        });

        assert_eq!(
            nearest,
            (0..32)
                .map(|row| usize::from(row % 2 == 0))
                .collect::<Vec<_>>()
        );
        assert_eq!(gemm_rows, 16);
    }

    /// Hybrid contract: batch results must equal the direct `find_topk` set
    /// (ties by index) on benign, offset, and tied data.
    #[test]
    fn test_batch_topk_matches_direct_contract() {
        let d = 96;
        let k = 300;
        let nq = 1200;
        let mut rng = StdRng::seed_from_u64(9);
        let base: Vec<f32> = (0..k * d).map(|_| rng.gen::<f32>() - 0.5).collect();
        let mut centroids = base.clone();
        // exact duplicates (ties) and near-duplicates (1 ulp apart)
        for c in 0..20 {
            let (src, dst) = (c * 7 % k, k - 1 - c);
            let row: Vec<f32> = centroids[src * d..(src + 1) * d].to_vec();
            centroids[dst * d..(dst + 1) * d].copy_from_slice(&row);
            if c % 2 == 0 {
                centroids[dst * d] = f32::from_bits(centroids[dst * d].to_bits() + 1);
            }
        }
        let mut queries: Vec<f32> = (0..nq * d).map(|_| rng.gen::<f32>() - 0.5).collect();
        // rows sitting exactly on a centroid, and rows in the middle of two
        for q in 0..200 {
            let c = q % k;
            queries[q * d..(q + 1) * d].copy_from_slice(&centroids[c * d..(c + 1) * d]);
        }
        for q in 200..400 {
            let (a, b) = ((q * 3) % k, (q * 5 + 1) % k);
            for j in 0..d {
                queries[q * d + j] = 0.5 * (centroids[a * d + j] + centroids[b * d + j]);
            }
        }
        // non-finite and overflowing rows must behave like `find_topk`
        queries[400 * d] = f32::NAN;
        queries[401 * d + 3] = f32::INFINITY;
        queries[402 * d + 5] = -f32::INFINITY;
        for v in queries[403 * d..404 * d].iter_mut() {
            *v = 1.0e30;
        }
        for (label, offset, scale) in [
            ("benign", 0.0f32, 1.0f32),
            ("offset1e8", 1.0e8, 4.0),
            ("offset1e4", 1.0e4, 1.0),
        ] {
            let c2: Vec<f32> = centroids.iter().map(|v| v * scale + offset).collect();
            let q2: Vec<f32> = queries.iter().map(|v| v * scale + offset).collect();
            for nprobe in [1usize, 2, 8, 64, k] {
                let (batch, bdist) = find_topk_batch(&q2, nq, &c2, k, d, nprobe);
                for qi in 0..nq {
                    let (direct, ddist) = find_topk(&q2[qi * d..(qi + 1) * d], &c2, k, d, nprobe);
                    assert_eq!(
                        batch[qi], direct,
                        "{label} nprobe={nprobe} row {qi} ids differ"
                    );
                    let bb: Vec<u32> = bdist[qi].iter().map(|v| v.to_bits()).collect();
                    let db: Vec<u32> = ddist.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(bb, db, "{label} nprobe={nprobe} row {qi} distances differ");
                }
                let nearest = find_nearest_batch(&q2, nq, &c2, k, d);
                for qi in 0..nq {
                    assert_eq!(
                        nearest[qi],
                        find_nearest(&q2[qi * d..(qi + 1) * d], &c2, k, d),
                        "{label} nearest row {qi}"
                    );
                }
            }
        }
    }
}

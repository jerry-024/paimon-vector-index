// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Exact coarse assignment through a low-dimensional projection.
//!
//! `IVFPQIndex::add` must find the nearest coarse centroid of every row. The
//! exact scan is one `n × nlist × d` GEMM, which dominates `add` once PQ
//! encoding is vectorized. This module keeps the result exact but does most of
//! the work in a `d'`-dimensional PCA subspace of the centroids:
//!
//! 1. project the row and centroids with a contractive PCA projection `P`
//!    (`d' × d`);
//! 2. for every centroid compute a conservative lower bound on the true
//!    squared distance, accounting for the f32 projection and GEMM errors;
//! 3. evaluate exact distances in ascending bound order and stop as soon as
//!    the next bound cannot beat the best exact distance (branch-and-bound).
//!
//! Step 2 is a valid lower bound for any contractive `P`, so the assignment
//! never depends on how well the PCA converged; `P` only decides how many
//! exact evaluations step 3 needs. Rows without low-dimensional structure
//! degrade to checking every centroid, i.e. the exact scan plus one small
//! GEMM. `d'` is therefore chosen by cost: at `train`, candidate widths are
//! tried on a sample of training vectors, the number of exact checks each
//! needs is measured, and the width minimizing `d'·nlist + w·checks·d` wins;
//! automatic mode keeps the projection only when that cost is clearly below
//! the exact scan's `d·nlist`. (Without a sample the width falls back to 95%
//! explained variance, gated by `d' ≤ d / 3`.)
//!
//! The projection is a training artifact: it is derived deterministically from
//! the centroids, shared by `IVFPQIndex::from_trained`, and never serialized.

use crate::blas::sgemm_a_bt;
use crate::distance::{fvec_l2sqr, fvec_l2sqr_four};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Fraction of centroid variance the projection must explain.
const VARIANCE_TARGET: f64 = 0.95;
/// Without a calibration sample, automatic mode keeps the projection only
/// when `d' * MAX_AUTO_DP_DIVISOR <= d`.
const MAX_AUTO_DP_DIVISOR: usize = 3;
/// Calibration samples below this fall back to the variance rule.
const MIN_CALIBRATION_ROWS: usize = 256;
/// Rows of the training data used to calibrate `d'`.
pub(crate) const CALIBRATION_ROWS: usize = 2048;
/// Per-centroid cost of computing one bound, in GEMM multiply-add units.
const BOUND_COST: f64 = 4.0;
/// Cost of one exact-check multiply-add relative to a GEMM multiply-add:
/// exact checks gather scattered centroids row by row, with no blocking.
/// Fitted on Cohere-768 / nlist=4096 timings across d' = 32..384.
const EXACT_CHECK_WEIGHT: f64 = 6.0;
/// With a calibration sample, automatic mode keeps the projection only when
/// its modeled cost is below this fraction of the exact scan.
const MAX_AUTO_COST_FRACTION: f64 = 0.7;
/// Block subspace iterations; the bound stays valid regardless of convergence.
const SUBSPACE_ITERATIONS: usize = 8;
const SUBSPACE_SEED: u64 = 0x7a5e_c7ed;
/// Smallest projection worth a GEMM; also the rounding unit for `d'`.
const MIN_DP: usize = 8;
/// Rows per parallel block; the projected score matrix is `rows × nlist`.
const MAX_BLOCK_ROWS: usize = 1024;
/// Leave a visible margin below one after rounding the projection to f32.
const CONTRACTION_MARGIN: f64 = 0.999;
/// Candidates evaluated per branch-and-bound stage before re-pruning.
const STAGE: usize = 32;

/// How `IVFPQIndex::add` chooses between the exact centroid scan and the
/// projected branch-and-bound. Both produce the exact nearest centroid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProjectedAssignment {
    /// Use the projection when the centroids are compressible enough for it
    /// to pay off; otherwise scan.
    #[default]
    Auto,
    /// Always build and use the projection (still exact).
    Enabled,
    /// Always scan.
    Disabled,
}

/// Contractive projection of the coarse centroids plus the per-centroid data
/// the lower bound needs.
#[derive(Debug)]
pub struct CoarseProjection {
    d: usize,
    dp: usize,
    explained_variance: f64,
    /// `dp × d`, scaled so its operator norm does not exceed one.
    proj: Vec<f32>,
    /// `nlist × dp`: projected centroids.
    cents_p: Vec<f32>,
    /// `|cents_p[c]|²`.
    cents_p_norms: Vec<f64>,
    /// `|cents_p[c]|`, cached to keep square roots out of assignment's hot loop.
    cents_p_norms_sqrt: Vec<f64>,
    /// Upper bound on the f32 GEMM error in each projected centroid.
    cents_p_errors: Vec<f64>,
}

impl CoarseProjection {
    /// Fit a projection to `nlist` centroids of dimension `d`.
    ///
    /// `calibration` holds `calibration_rows` sample vectors in the centroid
    /// space (typically training vectors). When present, `d'` is chosen by
    /// measuring, for each candidate width, how many exact distance checks
    /// the branch-and-bound needs on that sample and picking the width that
    /// minimizes `d' * nlist + EXACT_CHECK_WEIGHT * checks * d`; automatic mode then keeps the
    /// projection only when that cost is clearly below the exact scan's
    /// `d * nlist`. Without a sample, `d'` falls back to the smallest width
    /// explaining `VARIANCE_TARGET` of the centroid variance, gated by
    /// `MAX_AUTO_DP_DIVISOR`.
    ///
    /// Returns `None` when the projection is not worth it (`force == false`)
    /// or cannot be built (too few centroids, zero variance).
    pub(crate) fn train(
        cents: &[f32],
        nlist: usize,
        d: usize,
        force: bool,
        calibration: &[f32],
        calibration_rows: usize,
    ) -> Option<Self> {
        if nlist == 0 || d == 0 {
            return None;
        }
        // Largest d' auto mode may pick, rounded down so the gate below holds;
        // forced mode searches up to d / 2 (beyond that the projected GEMM
        // costs about as much as the scan it is meant to replace).
        let auto_limit = (d / MAX_AUTO_DP_DIVISOR) / MIN_DP * MIN_DP;
        let block = if force {
            (d / 2).max(MIN_DP).min(d)
        } else {
            auto_limit
        };
        if !force && (block < MIN_DP || nlist < 2 * block) {
            return None;
        }
        let block = block.min(nlist).max(1);

        let mean = column_mean(cents, nlist, d);
        let centered: Vec<f32> = cents
            .iter()
            .enumerate()
            .map(|(i, v)| v - mean[i % d])
            .collect();
        let total_variance = centered
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            / nlist as f64;
        if total_variance.is_nan() || total_variance <= 0.0 {
            return None;
        }

        let (basis, eigenvalues) = top_subspace(&centered, nlist, d, block);
        let calibration_rows = calibration_rows.min(calibration.len() / d.max(1));
        let dp = if calibration_rows >= MIN_CALIBRATION_ROWS {
            Self::select_width_by_cost(
                cents,
                &basis,
                nlist,
                d,
                block,
                force,
                &calibration[..calibration_rows * d],
                calibration_rows,
            )?
        } else {
            Self::select_width_by_variance(&eigenvalues, total_variance, block, d, force)?
        };
        let explained_variance = eigenvalues[..dp].iter().sum::<f64>() / total_variance;
        Some(Self::from_basis(
            cents,
            &basis,
            nlist,
            d,
            dp,
            explained_variance,
        ))
    }

    /// Smallest width explaining `VARIANCE_TARGET` of the centroid variance.
    fn select_width_by_variance(
        eigenvalues: &[f64],
        total_variance: f64,
        block: usize,
        d: usize,
        force: bool,
    ) -> Option<usize> {
        let target = VARIANCE_TARGET * total_variance;
        let mut cumulative = 0.0;
        let mut dp = None;
        for (i, lambda) in eigenvalues.iter().enumerate() {
            cumulative += lambda;
            if cumulative >= target {
                dp = Some(i + 1);
                break;
            }
        }
        let dp = match dp {
            Some(dp) => dp.div_ceil(MIN_DP) * MIN_DP,
            None if force => block,
            None => return None,
        }
        .min(block);
        if !force && dp * MAX_AUTO_DP_DIVISOR > d {
            return None;
        }
        Some(dp)
    }

    /// Width minimizing the measured per-row cost on the calibration sample.
    #[allow(clippy::too_many_arguments)]
    fn select_width_by_cost(
        cents: &[f32],
        basis: &[f32],
        nlist: usize,
        d: usize,
        block: usize,
        force: bool,
        calibration: &[f32],
        calibration_rows: usize,
    ) -> Option<usize> {
        let exact_cost = (d * nlist) as f64;
        let mut best: Option<(f64, usize)> = None;
        let mut widths: Vec<usize> = [8usize, 4, 2]
            .iter()
            .map(|div| (block / div).div_ceil(MIN_DP) * MIN_DP)
            .chain([(block * 3 / 4).div_ceil(MIN_DP) * MIN_DP, block])
            .filter(|&w| w >= MIN_DP && w <= block)
            .collect();
        widths.sort_unstable();
        widths.dedup();
        for dp in widths {
            let candidate = Self::from_basis(cents, basis, nlist, d, dp, 0.0);
            let (_, evaluations) =
                candidate.assign_with_stats(calibration, calibration_rows, cents, nlist);
            let checks_per_row = evaluations as f64 / calibration_rows as f64;
            let cost = (dp * nlist) as f64
                + EXACT_CHECK_WEIGHT * checks_per_row * d as f64
                + BOUND_COST * nlist as f64;
            if best.is_none_or(|(c, _)| cost < c) {
                best = Some((cost, dp));
            }
        }
        let (cost, dp) = best?;
        if !force && cost > exact_cost * MAX_AUTO_COST_FRACTION {
            return None;
        }
        Some(dp)
    }

    fn from_basis(
        cents: &[f32],
        basis: &[f32],
        nlist: usize,
        d: usize,
        dp: usize,
        explained_variance: f64,
    ) -> Self {
        let mut proj = basis[..dp * d].to_vec();
        make_contractive(&mut proj, dp, d);
        let mut cents_p = vec![0.0f32; nlist * dp];
        // Project uncentered vectors. Translation cancels in x-c, while this
        // avoids a second source of f32 rounding in the lower bound.
        sgemm_a_bt(nlist, dp, d, 1.0, cents, &proj, 0.0, &mut cents_p);
        let cents_p_norms: Vec<f64> = (0..nlist)
            .map(|c| norm_l2sqr_f64(&cents_p[c * dp..(c + 1) * dp]))
            .collect();
        let cents_p_norms_sqrt = cents_p_norms.iter().map(|norm| norm.sqrt()).collect();
        let cents_p_errors: Vec<f64> = cents
            .chunks_exact(d)
            .map(|c| projection_error(norm_upper(c), d, dp))
            .collect();
        Self {
            d,
            dp,
            explained_variance,
            proj,
            cents_p,
            cents_p_norms,
            cents_p_norms_sqrt,
            cents_p_errors,
        }
    }

    pub fn dimension(&self) -> usize {
        self.dp
    }

    pub fn explained_variance(&self) -> f64 {
        self.explained_variance
    }

    /// Exact nearest centroid of every row (ties: smallest centroid index).
    /// `data` must already be in the centroid space (normalized / rotated).
    pub(crate) fn assign(&self, data: &[f32], n: usize, cents: &[f32], nlist: usize) -> Vec<usize> {
        self.assign_with_stats(data, n, cents, nlist).0
    }

    /// `assign` plus the total number of exact distance evaluations, for
    /// tests and benchmarks.
    fn assign_with_stats(
        &self,
        data: &[f32],
        n: usize,
        cents: &[f32],
        nlist: usize,
    ) -> (Vec<usize>, usize) {
        let d = self.d;
        let dp = self.dp;
        let gemm_error_factor = 2.0 * gamma_f32(dp.saturating_mul(2).saturating_add(1));
        let f64_error_factor = gamma_f64(dp.saturating_mul(2).saturating_add(8));
        debug_assert_eq!(self.cents_p.len(), nlist * dp);
        let mut out = vec![0usize; n];
        let (block_rows, _) =
            crate::kmeans::assignment_block_plan(n, dp, nlist, rayon::current_num_threads());
        let block_rows = block_rows.min(MAX_BLOCK_ROWS);
        let evaluations = out
            .par_chunks_mut(block_rows)
            .enumerate()
            .map(|(bi, chunk)| {
                let rows = chunk.len();
                let row0 = bi * block_rows;
                let x = &data[row0 * d..(row0 + rows) * d];
                let mut xp = vec![0.0f32; rows * dp];
                sgemm_a_bt(rows, dp, d, 1.0, x, &self.proj, 0.0, &mut xp);
                let mut ip = vec![0.0f32; rows * nlist];
                sgemm_a_bt(rows, nlist, dp, 1.0, &xp, &self.cents_p, 0.0, &mut ip);

                let mut bounds = vec![0.0f64; nlist];
                let mut candidates: Vec<(f64, u32)> = Vec::new();
                let mut evaluations = 0usize;
                for i in 0..rows {
                    let xp_i = &xp[i * dp..(i + 1) * dp];
                    let xn = norm_l2sqr_f64(xp_i);
                    let xn_sqrt = xn.sqrt();
                    let x_i = &x[i * d..(i + 1) * d];
                    let x_error = projection_error(norm_upper(x_i), d, dp);
                    let ip_i = &ip[i * nlist..(i + 1) * nlist];
                    for c in 0..nlist {
                        bounds[c] = projected_distance_lower_bound(
                            xn,
                            xn_sqrt,
                            self.cents_p_norms[c],
                            self.cents_p_norms_sqrt[c],
                            ip_i[c],
                            x_error + self.cents_p_errors[c],
                            gemm_error_factor,
                            f64_error_factor,
                        );
                    }
                    let min_bound = bounds.iter().copied().fold(f64::INFINITY, f64::min);
                    let first = bounds.iter().position(|&b| b == min_bound).unwrap_or(0);
                    let mut best = fvec_l2sqr(x_i, &cents[first * d..(first + 1) * d]);
                    let mut best_idx = first;
                    evaluations += 1;
                    let mut best_upper = distance_upper_bound(best, d);
                    candidates.clear();
                    for (c, &bound) in bounds.iter().enumerate() {
                        if c != first && bound <= best_upper {
                            candidates.push((bound, c as u32));
                        }
                    }
                    // Evaluate in stages of the `STAGE` smallest bounds: the
                    // best distance usually drops within the first few exact
                    // checks, which prunes the rest without sorting them.
                    let mut pending = candidates.as_mut_slice();
                    while !pending.is_empty() {
                        let stage = STAGE.min(pending.len());
                        if stage < pending.len() {
                            pending.select_nth_unstable_by(stage - 1, compare_bound_then_index);
                        }
                        let (head, tail) = pending.split_at_mut(stage);
                        head.sort_unstable_by(compare_bound_then_index);
                        evaluations += evaluate_candidates(
                            x_i,
                            cents,
                            d,
                            head,
                            &mut best,
                            &mut best_idx,
                            &mut best_upper,
                        );
                        // Everything left has a bound >= the last of `head`.
                        if head.last().is_some_and(|&(bound, _)| bound > best_upper) {
                            break;
                        }
                        let mut kept = 0;
                        for j in 0..tail.len() {
                            if tail[j].0 <= best_upper {
                                tail.swap(kept, j);
                                kept += 1;
                            }
                        }
                        pending = &mut tail[..kept];
                    }
                    chunk[i] = best_idx;
                }
                evaluations
            })
            .sum();
        (out, evaluations)
    }

    /// Lower bound of `|x - c|²` for one row/centroid pair, for tests.
    #[cfg(test)]
    fn lower_bound(&self, x: &[f32], c: usize) -> f64 {
        let d = self.d;
        let dp = self.dp;
        let mut xp = vec![0.0f32; dp];
        sgemm_a_bt(1, dp, d, 1.0, x, &self.proj, 0.0, &mut xp);
        let cp = &self.cents_p[c * dp..(c + 1) * dp];
        let ip = xp.iter().zip(cp).map(|(a, b)| a * b).sum();
        let x_norm = norm_l2sqr_f64(&xp);
        projected_distance_lower_bound(
            x_norm,
            x_norm.sqrt(),
            self.cents_p_norms[c],
            self.cents_p_norms_sqrt[c],
            ip,
            projection_error(norm_upper(x), d, dp) + self.cents_p_errors[c],
            2.0 * gamma_f32(dp.saturating_mul(2).saturating_add(1)),
            gamma_f64(dp.saturating_mul(2).saturating_add(8)),
        )
    }
}

fn gamma(unit_roundoff: f64, operations: usize) -> f64 {
    let error = unit_roundoff * operations as f64;
    if error < 1.0 {
        error / (1.0 - error)
    } else {
        f64::INFINITY
    }
}

fn gamma_f32(operations: usize) -> f64 {
    gamma(f32::EPSILON as f64 / 2.0, operations)
}

fn gamma_f64(operations: usize) -> f64 {
    gamma(f64::EPSILON / 2.0, operations)
}

fn norm_l2sqr_f64(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64) * (*x as f64)).sum()
}

fn norm_upper(v: &[f32]) -> f64 {
    let norm = norm_l2sqr_f64(v);
    let error = gamma_f64(v.len().saturating_mul(2));
    if error >= 1.0 {
        f64::INFINITY
    } else {
        (norm / (1.0 - error)).sqrt()
    }
}

fn projection_error(norm: f64, d: usize, dp: usize) -> f64 {
    (dp as f64).sqrt() * gamma_f32(d.saturating_mul(2).saturating_add(1)) * norm
}

fn projected_distance_lower_bound(
    x_norm: f64,
    x_norm_sqrt: f64,
    c_norm: f64,
    c_norm_sqrt: f64,
    inner_product: f32,
    projection_error: f64,
    gemm_error_factor: f64,
    f64_error_factor: f64,
) -> f64 {
    let inner_product = inner_product as f64;
    let magnitude = x_norm + c_norm + 2.0 * inner_product.abs();
    let gemm_error = gemm_error_factor * x_norm_sqrt * c_norm_sqrt;
    let f64_error = f64_error_factor * magnitude;
    let projected_sqr = (x_norm + c_norm - 2.0 * inner_product - gemm_error - f64_error).max(0.0);
    let error_sqr = projection_error * projection_error;
    if projected_sqr <= error_sqr {
        0.0
    } else {
        // sqrt(projected_sqr) <= |xp| + |cp|. Using that upper bound in
        // (sqrt(projected_sqr) - projection_error)^2 keeps this conservative
        // without a square root for every row-centroid pair.
        (projected_sqr - 2.0 * projection_error * (x_norm_sqrt + c_norm_sqrt) + error_sqr).max(0.0)
    }
}

fn distance_upper_bound(distance: f32, d: usize) -> f64 {
    if !distance.is_finite() {
        return f64::INFINITY;
    }
    let relative_error = gamma_f32(d.saturating_mul(4).saturating_add(16));
    if relative_error >= 1.0 {
        return f64::INFINITY;
    }
    distance as f64 / (1.0 - relative_error) + d as f64 * f32::MIN_POSITIVE as f64
}

fn make_contractive(proj: &mut [f32], rows: usize, d: usize) {
    let dot_roundoff = gamma_f64(d.saturating_mul(2).saturating_add(1));
    let mut max_row_sum = 0.0f64;
    for i in 0..rows {
        let mut row_sum = 0.0;
        for j in 0..rows {
            let mut dot = 0.0;
            let mut absolute_sum = 0.0;
            for (a, b) in proj[i * d..(i + 1) * d]
                .iter()
                .zip(&proj[j * d..(j + 1) * d])
            {
                let product = (*a as f64) * (*b as f64);
                dot += product;
                absolute_sum += product.abs();
            }
            row_sum += dot.abs() + dot_roundoff * absolute_sum;
        }
        let sum_roundoff = gamma_f64(rows);
        max_row_sum = max_row_sum.max(row_sum / (1.0 - sum_roundoff).max(f64::MIN_POSITIVE));
    }
    let scale = CONTRACTION_MARGIN / max_row_sum.max(1.0).sqrt();
    for value in proj {
        *value *= scale as f32;
    }
}

fn compare_bound_then_index(a: &(f64, u32), b: &(f64, u32)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Exact distances for `head` (sorted by bound), four centroids at a time,
/// stopping once a bound cannot beat `best`. Returns the evaluation count.
fn evaluate_candidates(
    x: &[f32],
    cents: &[f32],
    d: usize,
    head: &[(f64, u32)],
    best: &mut f32,
    best_idx: &mut usize,
    best_upper: &mut f64,
) -> usize {
    let mut evaluations = 0;
    let mut pos = 0;
    while pos < head.len() {
        if head[pos].0 > *best_upper {
            break;
        }
        let take = 4.min(head.len() - pos);
        let ids = [
            head[pos].1 as usize,
            head[(pos + 1).min(head.len() - 1)].1 as usize,
            head[(pos + 2).min(head.len() - 1)].1 as usize,
            head[(pos + 3).min(head.len() - 1)].1 as usize,
        ];
        let dists = if take == 4 {
            fvec_l2sqr_four(
                x,
                &cents[ids[0] * d..(ids[0] + 1) * d],
                &cents[ids[1] * d..(ids[1] + 1) * d],
                &cents[ids[2] * d..(ids[2] + 1) * d],
                &cents[ids[3] * d..(ids[3] + 1) * d],
            )
        } else {
            let mut dists = [f32::INFINITY; 4];
            for (k, dist) in dists.iter_mut().enumerate().take(take) {
                *dist = fvec_l2sqr(x, &cents[ids[k] * d..(ids[k] + 1) * d]);
            }
            dists
        };
        evaluations += take;
        for k in 0..take {
            let c = ids[k];
            if dists[k] < *best || (dists[k] == *best && c < *best_idx) {
                *best = dists[k];
                *best_idx = c;
            }
        }
        *best_upper = distance_upper_bound(*best, d);
        pos += take;
    }
    evaluations
}

fn column_mean(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut mean = vec![0.0f64; d];
    for row in data[..n * d].chunks(d) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += *v as f64;
        }
    }
    mean.iter().map(|m| (m / n as f64) as f32).collect()
}

/// Top-`block` principal directions of the centered `n × d` matrix by block
/// subspace iteration with a Rayleigh-Ritz step. Returns the basis as
/// `block × d` orthonormal rows sorted by decreasing eigenvalue, plus the
/// eigenvalues (variance along each direction, `|Cv|² / n`).
fn top_subspace(centered: &[f32], n: usize, d: usize, block: usize) -> (Vec<f32>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(SUBSPACE_SEED);
    let mut basis: Vec<f32> = (0..block * d).map(|_| rng.gen::<f32>() - 0.5).collect();
    orthonormalize_rows(&mut basis, block, d, &mut rng);

    let transposed: Vec<f32> = (0..d * n).map(|i| centered[(i % n) * d + i / n]).collect();
    let mut projected_t = vec![0.0f32; block * n];
    for _ in 0..SUBSPACE_ITERATIONS {
        project_transposed(centered, n, d, &basis, block, &mut projected_t);
        // basis <- (Cᵀ (C Qᵀ))ᵀ = Yᵀ C
        sgemm_a_bt(block, d, n, 1.0, &projected_t, &transposed, 0.0, &mut basis);
        orthonormalize_rows(&mut basis, block, d, &mut rng);
    }

    // Rayleigh-Ritz: diagonalize Q C ᵀ C Qᵀ / n and rotate the basis.
    project_transposed(centered, n, d, &basis, block, &mut projected_t);
    let mut gram = vec![0.0f32; block * block];
    sgemm_a_bt(
        block,
        block,
        n,
        1.0,
        &projected_t,
        &projected_t,
        0.0,
        &mut gram,
    );
    let gram = DMatrix::from_fn(block, block, |i, j| gram[i * block + j] as f64 / n as f64);
    let eigen = SymmetricEigen::new(gram);
    let mut order: Vec<usize> = (0..block).collect();
    order.sort_by(|a, b| eigen.eigenvalues[*b].total_cmp(&eigen.eigenvalues[*a]));

    let mut rotated = vec![0.0f32; block * d];
    for (r, &src) in order.iter().enumerate() {
        for j in 0..block {
            let w = eigen.eigenvectors[(j, src)] as f32;
            if w != 0.0 {
                for k in 0..d {
                    rotated[r * d + k] += w * basis[j * d + k];
                }
            }
        }
    }
    orthonormalize_rows(&mut rotated, block, d, &mut rng);
    let eigenvalues = order
        .iter()
        .map(|&i| eigen.eigenvalues[i].max(0.0))
        .collect();
    (rotated, eigenvalues)
}

/// `out` (block × n) <- (C Qᵀ)ᵀ, chunked over rows of C in parallel.
fn project_transposed(
    centered: &[f32],
    n: usize,
    d: usize,
    basis: &[f32],
    block: usize,
    out: &mut [f32],
) {
    let chunk_rows = 1024usize;
    let chunks: Vec<Vec<f32>> = centered
        .par_chunks(chunk_rows * d)
        .map(|rows| {
            let r = rows.len() / d;
            let mut y = vec![0.0f32; r * block];
            sgemm_a_bt(r, block, d, 1.0, rows, basis, 0.0, &mut y);
            y
        })
        .collect();
    let mut row0 = 0;
    for y in chunks {
        let r = y.len() / block;
        for i in 0..r {
            for j in 0..block {
                out[j * n + row0 + i] = y[i * block + j];
            }
        }
        row0 += r;
    }
}

/// Modified Gram-Schmidt in f64 over the rows of `m` (`rows × d`). Rows that
/// vanish (rank deficiency) are replaced by random vectors so the basis stays
/// orthonormal; the bound remains valid for any orthonormal basis.
fn orthonormalize_rows(m: &mut [f32], rows: usize, d: usize, rng: &mut StdRng) {
    let mut acc: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut v: Vec<f64> = m[r * d..(r + 1) * d].iter().map(|x| *x as f64).collect();
        for _attempt in 0..3 {
            for q in &acc {
                let dot: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
                for (vi, qi) in v.iter_mut().zip(q) {
                    *vi -= dot * qi;
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-9 {
                for vi in v.iter_mut() {
                    *vi /= norm;
                }
                break;
            }
            v = (0..d).map(|_| rng.gen::<f64>() - 0.5).collect();
        }
        for (dst, src) in m[r * d..(r + 1) * d].iter_mut().zip(&v) {
            *dst = *src as f32;
        }
        acc.push(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans;

    fn low_rank_centroids(nlist: usize, d: usize, rank: usize, noise: f32, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let factors: Vec<f32> = (0..rank * d).map(|_| rng.gen::<f32>() - 0.5).collect();
        let mut out = vec![0.0f32; nlist * d];
        for row in out.chunks_mut(d) {
            let z: Vec<f32> = (0..rank).map(|_| rng.gen::<f32>() - 0.5).collect();
            for (j, v) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for t in 0..rank {
                    acc += z[t] * factors[t * d + j];
                }
                *v = acc + (rng.gen::<f32>() - 0.5) * noise + 3.0;
            }
        }
        out
    }

    fn rows_like(
        cents: &[f32],
        nlist: usize,
        d: usize,
        n: usize,
        spread: f32,
        seed: u64,
    ) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .flat_map(|i| {
                let c = (i * 7919) % nlist;
                (0..d)
                    .map(|j| cents[c * d + j] + (rng.gen::<f32>() - 0.5) * spread)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn projection_is_contractive() {
        let (nlist, d) = (512, 96);
        let cents = low_rank_centroids(nlist, d, 12, 0.05, 1);
        let p = CoarseProjection::train(&cents, nlist, d, true, &[], 0).unwrap();
        let mut max_row_sum = 0.0f64;
        for a in 0..p.dp {
            let mut row_sum = 0.0;
            for b in 0..p.dp {
                let dot: f64 = p.proj[a * d..(a + 1) * d]
                    .iter()
                    .zip(&p.proj[b * d..(b + 1) * d])
                    .map(|(x, y)| (*x as f64) * (*y as f64))
                    .sum();
                row_sum += dot.abs();
            }
            max_row_sum = max_row_sum.max(row_sum);
        }
        assert!(max_row_sum < 1.0, "Gram row sum is {max_row_sum}");
    }

    #[test]
    fn auto_mode_keeps_compressible_centroids_and_rejects_noise() {
        let (nlist, d) = (512, 96);
        let structured = low_rank_centroids(nlist, d, 12, 0.05, 2);
        let p = CoarseProjection::train(&structured, nlist, d, false, &[], 0).unwrap();
        assert!(p.dimension() * MAX_AUTO_DP_DIVISOR <= d);
        assert!(p.explained_variance() >= VARIANCE_TARGET);

        let mut rng = StdRng::seed_from_u64(3);
        let noise: Vec<f32> = (0..nlist * d).map(|_| rng.gen::<f32>()).collect();
        assert!(CoarseProjection::train(&noise, nlist, d, false, &[], 0).is_none());
        // Forced mode still builds one (and stays exact).
        assert!(CoarseProjection::train(&noise, nlist, d, true, &[], 0).is_some());
    }

    #[test]
    fn train_rejects_degenerate_inputs() {
        assert!(CoarseProjection::train(&[], 0, 16, true, &[], 0).is_none());
        assert!(CoarseProjection::train(&vec![1.0; 4 * 16], 4, 16, false, &[], 0).is_none());
        assert!(CoarseProjection::train(&vec![1.0; 64 * 16], 64, 16, true, &[], 0).is_none());
    }

    #[test]
    fn lower_bound_never_exceeds_distance() {
        let (nlist, d) = (256, 64);
        let cents = low_rank_centroids(nlist, d, 8, 0.3, 4);
        let p = CoarseProjection::train(&cents, nlist, d, true, &[], 0).unwrap();
        let rows = rows_like(&cents, nlist, d, 200, 1.0, 5);
        for i in 0..200 {
            let x = &rows[i * d..(i + 1) * d];
            for c in 0..nlist {
                let dist = x
                    .iter()
                    .zip(&cents[c * d..(c + 1) * d])
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>();
                let bound = p.lower_bound(x, c);
                assert!(bound <= dist, "row {i} c {c}: {bound} > {dist}");
            }
        }
    }

    #[test]
    fn assignment_matches_exact_scan_and_prunes() {
        let (nlist, d, n) = (512, 96, 3000);
        let cents = low_rank_centroids(nlist, d, 12, 0.05, 6);
        let rows = rows_like(&cents, nlist, d, n, 0.4, 7);
        let p = CoarseProjection::train(&cents, nlist, d, false, &[], 0).unwrap();
        let (got, evaluations) = p.assign_with_stats(&rows, n, &cents, nlist);
        let exact: Vec<usize> = rows
            .chunks_exact(d)
            .map(|x| kmeans::find_nearest(x, &cents, nlist, d))
            .collect();
        assert_eq!(got, exact);
        assert!(
            evaluations < n * nlist / 20,
            "pruning too weak: {evaluations} evaluations"
        );
    }

    #[test]
    fn assignment_is_exact_on_unstructured_data() {
        let (nlist, d, n) = (128, 32, 1000);
        let mut rng = StdRng::seed_from_u64(8);
        let cents: Vec<f32> = (0..nlist * d).map(|_| rng.gen::<f32>()).collect();
        let rows: Vec<f32> = (0..n * d).map(|_| rng.gen::<f32>()).collect();
        let p = CoarseProjection::train(&cents, nlist, d, true, &[], 0).unwrap();
        let got = p.assign(&rows, n, &cents, nlist);
        let exact: Vec<usize> = rows
            .chunks_exact(d)
            .map(|x| kmeans::find_nearest(x, &cents, nlist, d))
            .collect();
        assert_eq!(got, exact);
        for x in rows.chunks_exact(d) {
            for c in 0..nlist {
                let distance = x
                    .iter()
                    .zip(&cents[c * d..(c + 1) * d])
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum::<f64>();
                assert!(p.lower_bound(x, c) <= distance);
            }
        }
    }

    #[test]
    fn assignment_is_exact_with_large_common_offset() {
        let (nlist, d) = (64, 16);
        let cents: Vec<f32> = (0..nlist)
            .flat_map(|c| {
                (0..d).map(move |j| 100_000_000.0 + (((c * 7 + j * 3) % 31) as f32) * 8.0)
            })
            .collect();
        let rows: Vec<f32> = [3usize, 17, 42]
            .into_iter()
            .flat_map(|c| cents[c * d..(c + 1) * d].iter().copied())
            .collect();
        let p = CoarseProjection::train(&cents, nlist, d, true, &[], 0).unwrap();
        let got = p.assign(&rows, 3, &cents, nlist);
        let exact: Vec<usize> = rows
            .chunks_exact(d)
            .map(|x| kmeans::find_nearest(x, &cents, nlist, d))
            .collect();
        assert_eq!(got, exact);
    }

    #[test]
    fn calibration_picks_a_cheaper_width_and_gates_on_cost() {
        let (nlist, d) = (512, 96);
        let cents = low_rank_centroids(nlist, d, 12, 0.3, 31);
        let rows = rows_like(&cents, nlist, d, 4000, 1.0, 32);
        let calibrated = CoarseProjection::train(&cents, nlist, d, false, &rows, 2048).unwrap();
        let variance_rule = CoarseProjection::train(&cents, nlist, d, false, &[], 0).unwrap();
        let (_, cal_evals) = calibrated.assign_with_stats(&rows[2048 * d..], 1952, &cents, nlist);
        let (_, var_evals) =
            variance_rule.assign_with_stats(&rows[2048 * d..], 1952, &cents, nlist);
        let cost = |dp: usize, evals: usize| (dp * nlist + evals * d / 1952) as f64;
        assert!(
            cost(calibrated.dimension(), cal_evals) <= cost(variance_rule.dimension(), var_evals),
            "calibrated d'={} ({cal_evals} checks) vs variance d'={} ({var_evals} checks)",
            calibrated.dimension(),
            variance_rule.dimension()
        );
        // Unstructured rows: the modeled cost exceeds the scan, auto declines.
        let mut rng = StdRng::seed_from_u64(33);
        let noise_cents: Vec<f32> = (0..nlist * d).map(|_| rng.gen::<f32>()).collect();
        let noise_rows: Vec<f32> = (0..1024 * d).map(|_| rng.gen::<f32>()).collect();
        assert!(
            CoarseProjection::train(&noise_cents, nlist, d, false, &noise_rows, 1024).is_none()
        );
    }

    #[test]
    fn ties_pick_smallest_centroid_index() {
        let (nlist, d) = (64, 16);
        let mut cents = low_rank_centroids(nlist, d, 4, 0.1, 9);
        // Duplicate centroid 40 at index 3 and 50.
        let dup: Vec<f32> = cents[40 * d..41 * d].to_vec();
        cents[3 * d..4 * d].copy_from_slice(&dup);
        cents[50 * d..51 * d].copy_from_slice(&dup);
        let p = CoarseProjection::train(&cents, nlist, d, true, &[], 0).unwrap();
        let mut row = dup.clone();
        row[0] += 0.01;
        let got = p.assign(&row, 1, &cents, nlist);
        assert_eq!(got, vec![3]);
    }
}

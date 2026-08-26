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
//!    squared distance: the projected distance plus the reverse triangle
//!    inequality on the components outside the subspace
//!    (`|x-c|² >= |P₀(x-c)|² + (|P₀⊥x| - |P₀⊥c|)²`), with every f32
//!    projection, GEMM and f64 rounding error accounted for explicitly;
//! 3. evaluate exact distances in ascending bound order and stop as soon as
//!    the next bound cannot beat the best exact distance (branch-and-bound).
//!
//! Step 2 is a valid lower bound for any contractive `P`, so the assignment
//! never depends on how well the PCA converged; `P` only decides how many
//! exact evaluations step 3 needs. Rows without low-dimensional structure
//! degrade to checking every centroid, i.e. the exact scan plus one small
//! GEMM. At `train`, candidate widths are therefore timed on a sample of
//! training vectors, including projection, bound construction, candidate
//! collection, and exact checks. Automatic mode keeps the fastest projection
//! only when it is clearly faster than an exact scan of the same sample.
//! (Without a sample the width falls back to 95% explained variance, gated by
//! `d' ≤ d / 3`.)
//!
//! The projection is a training artifact: it is derived from the centroids,
//! shared by `IVFPQIndex::from_trained`, and never serialized.

use crate::blas::{dgemm_a_bt, sgemm_a_bt};
use crate::distance::{fvec_l2sqr, fvec_l2sqr_four};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Fraction of centroid variance the projection must explain.
const VARIANCE_TARGET: f64 = 0.95;
/// Without a calibration sample, automatic mode keeps the projection only
/// when `d' * MAX_AUTO_DP_DIVISOR <= d`.
const MAX_AUTO_DP_DIVISOR: usize = 3;
/// Calibration samples below this fall back to the variance rule.
const MIN_CALIBRATION_ROWS: usize = 256;
/// Rows of the training data used to calibrate `d'`.
pub(crate) const CALIBRATION_ROWS: usize = 2048;
/// With a calibration sample, automatic mode keeps the projection only when
/// its measured time is below this fraction of the exact scan.
const MAX_AUTO_COST_FRACTION: f64 = 0.85;
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
#[derive(Clone, Debug)]
pub struct CoarseProjection {
    d: usize,
    dp: usize,
    explained_variance: f64,
    /// Centroid mean (f64), subtracted before projecting rows and centroids.
    /// Translation cancels in the projected term but not in the residual
    /// term, whose norms must not be dominated by a common offset.
    mean: Vec<f64>,
    /// `dp × d`, scaled so its operator norm does not exceed one. Stored in
    /// f64 (exactly the f32 values used for the singular-value bounds) so the
    /// row projection carries only f64 rounding.
    proj: Vec<f64>,
    /// `nlist × dp`: projected centroids.
    cents_p: Vec<f32>,
    /// `|cents_p[c]|²`.
    cents_p_norms: Vec<f64>,
    /// `|cents_p[c]|`, cached to keep square roots out of assignment's hot loop.
    cents_p_norms_sqrt: Vec<f64>,
    /// Upper bound on the f32 GEMM error in each projected centroid.
    cents_p_errors: Vec<f64>,
    /// `|c - mean|²`, reused while calibrating narrower projections.
    cent_norms: Vec<f64>,
    /// Bounds on the squared singular values of `proj`: `|P v|² / sigma_max_sq
    /// <= |P₀ v|² <= |P v|² / sigma_min_sq` for the orthonormal basis `P₀` of
    /// its row space. `sigma_min_sq == 0` disables the residual term.
    sigma_min_sq: f64,
    sigma_max_sq: f64,
    /// Interval for `|P₀⊥ c|`, the norm of each centroid's component outside
    /// the projected subspace, including every rounding error above.
    cents_res_lo: Vec<f64>,
    cents_res_hi: Vec<f64>,
}

impl CoarseProjection {
    /// Fit a projection to `nlist` centroids of dimension `d`.
    ///
    /// `calibration` holds `calibration_rows` sample vectors in the centroid
    /// space (typically training vectors). When present, `d'` is chosen by
    /// timing each candidate width on that sample and picking the fastest;
    /// automatic mode then keeps the projection only when it is clearly
    /// faster than an exact scan of the same sample. Without a sample, `d'`
    /// falls back to the smallest width
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
        if calibration_rows >= MIN_CALIBRATION_ROWS {
            let mut projection = Self::select_width_by_time(
                cents,
                &basis,
                nlist,
                d,
                block,
                force,
                &calibration[..calibration_rows * d],
                calibration_rows,
            )?;
            projection.explained_variance =
                eigenvalues[..projection.dp].iter().sum::<f64>() / total_variance;
            return Some(projection);
        }
        let dp = Self::select_width_by_variance(&eigenvalues, total_variance, block, d, force)?;
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

    /// Width minimizing elapsed assignment time on the calibration sample.
    #[allow(clippy::too_many_arguments)]
    fn select_width_by_time(
        cents: &[f32],
        basis: &[f32],
        nlist: usize,
        d: usize,
        block: usize,
        force: bool,
        calibration: &[f32],
        calibration_rows: usize,
    ) -> Option<Self> {
        let exact_elapsed = if force {
            None
        } else {
            let started = Instant::now();
            let _ =
                crate::kmeans::find_nearest_batch(calibration, calibration_rows, cents, nlist, d);
            Some(started.elapsed())
        };
        let mut best: Option<(Duration, Self)> = None;
        let mut widths: Vec<usize> = [8usize, 4, 2]
            .iter()
            .map(|div| (block / div).div_ceil(MIN_DP) * MIN_DP)
            .chain([(block * 3 / 4).div_ceil(MIN_DP) * MIN_DP, block])
            .filter(|&w| w >= MIN_DP && w <= block)
            .collect();
        if force && widths.is_empty() {
            widths.push(block);
        }
        widths.sort_unstable();
        widths.dedup();
        let mut candidate = Self::from_basis(cents, basis, nlist, d, block, 0.0);
        for dp in widths.into_iter().rev() {
            candidate.truncate_dimension(dp);
            let started = Instant::now();
            let _ = candidate.assign_with_stats(calibration, calibration_rows, cents, nlist);
            let elapsed = started.elapsed();
            if best.as_ref().is_none_or(|(time, _)| elapsed <= *time) {
                best = Some((elapsed, candidate.clone()));
            }
        }
        let (elapsed, candidate) = best?;
        if let Some(exact_elapsed) =
            exact_elapsed.filter(|exact_elapsed| !is_fast_enough(elapsed, *exact_elapsed))
        {
            crate::logging::emit_log(
                crate::logging::LogLevel::Info,
                &format!(
                    "IVF-PQ projected assignment not used: best width d'={} measured {:.0}% of exact scan time",
                    candidate.dp,
                    elapsed.as_secs_f64() / exact_elapsed.as_secs_f64() * 100.0
                ),
            );
            return None;
        }
        Some(candidate)
    }

    fn truncate_dimension(&mut self, dp: usize) {
        if dp == self.dp {
            return;
        }
        debug_assert!(dp < self.dp);
        let old_dp = self.dp;
        self.proj.truncate(dp * self.d);
        self.cents_p = self
            .cents_p
            .chunks_exact(old_dp)
            .flat_map(|row| row[..dp].iter().copied())
            .collect();
        self.dp = dp;

        // By eigenvalue interlacing, the full-width singular-value bounds are
        // conservative for every row prefix.
        self.cents_p_norms = self.cents_p.chunks_exact(dp).map(norm_l2sqr_f64).collect();
        self.cents_p_norms_sqrt = self.cents_p_norms.iter().map(|norm| norm.sqrt()).collect();
        self.cents_p_errors = self
            .cents_p_norms_sqrt
            .iter()
            .zip(&self.cent_norms)
            .map(|(proj_norm, &cent_norm)| {
                projection_error(
                    centered_norm_upper(cent_norm, self.d),
                    *proj_norm,
                    self.d,
                    dp,
                )
            })
            .collect();
        (self.cents_res_lo, self.cents_res_hi) = self
            .cent_norms
            .iter()
            .zip(&self.cents_p_norms)
            .zip(&self.cents_p_errors)
            .map(|((&cent_norm, &proj_norm), &error)| {
                residual_interval(
                    cent_norm,
                    self.d,
                    proj_norm,
                    error,
                    self.sigma_min_sq,
                    self.sigma_max_sq,
                )
            })
            .unzip();
    }

    fn from_basis(
        cents: &[f32],
        basis: &[f32],
        nlist: usize,
        d: usize,
        dp: usize,
        explained_variance: f64,
    ) -> Self {
        let mut proj_f32 = basis[..dp * d].to_vec();
        make_contractive(&mut proj_f32, dp, d);
        let (sigma_min_sq, sigma_max_sq) = singular_value_bounds(&proj_f32, dp, d);
        let proj: Vec<f64> = proj_f32.iter().map(|v| *v as f64).collect();
        let mean: Vec<f64> = column_mean(cents, nlist, d)
            .iter()
            .map(|v| *v as f64)
            .collect();
        // Center and project in f64, then round to f32 for the row×centroid
        // GEMM. `cent_norms` are `|c - mean|²` in f64.
        let (cents_p, cent_norms) = project_rows(cents, nlist, d, &proj, dp, &mean);
        let cents_p_norms: Vec<f64> = (0..nlist)
            .map(|c| norm_l2sqr_f64(&cents_p[c * dp..(c + 1) * dp]))
            .collect();
        let cents_p_norms_sqrt: Vec<f64> = cents_p_norms.iter().map(|norm| norm.sqrt()).collect();
        let cents_p_errors: Vec<f64> = (0..nlist)
            .map(|c| {
                projection_error(
                    centered_norm_upper(cent_norms[c], d),
                    cents_p_norms_sqrt[c],
                    d,
                    dp,
                )
            })
            .collect();
        let (cents_res_lo, cents_res_hi): (Vec<f64>, Vec<f64>) = (0..nlist)
            .map(|c| {
                residual_interval(
                    cent_norms[c],
                    d,
                    cents_p_norms[c],
                    cents_p_errors[c],
                    sigma_min_sq,
                    sigma_max_sq,
                )
            })
            .unzip();
        Self {
            d,
            dp,
            explained_variance,
            mean,
            proj,
            cents_p,
            cents_p_norms,
            cents_p_norms_sqrt,
            cents_p_errors,
            cent_norms,
            sigma_min_sq,
            sigma_max_sq,
            cents_res_lo,
            cents_res_hi,
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
                let (xp, x_norms) = project_rows(x, rows, d, &self.proj, dp, &self.mean);
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
                    let x_error =
                        projection_error(centered_norm_upper(x_norms[i], d), xn_sqrt, d, dp);
                    let (x_res_lo, x_res_hi) = residual_interval(
                        x_norms[i],
                        d,
                        xn,
                        x_error,
                        self.sigma_min_sq,
                        self.sigma_max_sq,
                    );
                    let ip_i = &ip[i * nlist..(i + 1) * nlist];
                    let row = RowBoundTerms {
                        xn,
                        xn_sqrt,
                        x_error,
                        gemm_scaled: gemm_error_factor * xn_sqrt,
                        f64_error_factor,
                        x_res_lo,
                        x_res_hi,
                    };
                    // One straight-line pass per row: every term is a few
                    // multiply-adds and a max, so the compiler can vectorize it.
                    for (((((bound, &ip_c), &cn), &cs), &ce), (&rlo, &rhi)) in bounds
                        .iter_mut()
                        .zip(ip_i)
                        .zip(&self.cents_p_norms)
                        .zip(&self.cents_p_norms_sqrt)
                        .zip(&self.cents_p_errors)
                        .zip(self.cents_res_lo.iter().zip(&self.cents_res_hi))
                    {
                        *bound = pair_lower_bound(&row, ip_c, cn, cs, ce, rlo, rhi);
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
                    // Evaluate the `STAGE` smallest bounds first: the best
                    // distance usually drops within a few exact checks, which
                    // prunes most of the rest without sorting them. If it does
                    // not, sort what survives once and finish linearly, so weak
                    // pruning costs O(k log k) rather than one partition per
                    // stage.
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
                        if pending.len() > STAGE * 2 {
                            pending.sort_unstable_by(compare_bound_then_index);
                            evaluations += evaluate_candidates(
                                x_i,
                                cents,
                                d,
                                pending,
                                &mut best,
                                &mut best_idx,
                                &mut best_upper,
                            );
                            break;
                        }
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
        let (xp, x_norms) = project_rows(x, 1, d, &self.proj, dp, &self.mean);
        let cp = &self.cents_p[c * dp..(c + 1) * dp];
        let ip = xp.iter().zip(cp).map(|(a, b)| a * b).sum();
        let x_norm = norm_l2sqr_f64(&xp);
        let x_error = projection_error(centered_norm_upper(x_norms[0], d), x_norm.sqrt(), d, dp);
        let (x_res_lo, x_res_hi) = residual_interval(
            x_norms[0],
            d,
            x_norm,
            x_error,
            self.sigma_min_sq,
            self.sigma_max_sq,
        );
        let row = RowBoundTerms {
            xn: x_norm,
            xn_sqrt: x_norm.sqrt(),
            x_error,
            gemm_scaled: 2.0 * gamma_f32(dp.saturating_mul(2).saturating_add(1)) * x_norm.sqrt(),
            f64_error_factor: gamma_f64(dp.saturating_mul(2).saturating_add(8)),
            x_res_lo,
            x_res_hi,
        };
        pair_lower_bound(
            &row,
            ip,
            self.cents_p_norms[c],
            self.cents_p_norms_sqrt[c],
            self.cents_p_errors[c],
            self.cents_res_lo[c],
            self.cents_res_hi[c],
        )
    }
}

fn is_fast_enough(projected: Duration, exact: Duration) -> bool {
    projected <= exact.mul_f64(MAX_AUTO_COST_FRACTION)
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

/// Rows of `data` (`rows × d`, f32) centered by `mean`, projected with `proj`
/// (`dp × d`, f64) in f64 and rounded to f32, plus `|row - mean|²` in f64 for
/// each row. Row `i` of the projection is `P·(data[i] - mean)` up to the f64
/// dot-product error plus half an ulp per component.
fn project_rows(
    data: &[f32],
    rows: usize,
    d: usize,
    proj: &[f64],
    dp: usize,
    mean: &[f64],
) -> (Vec<f32>, Vec<f64>) {
    let mut centered = vec![0.0f64; rows * d];
    let mut norms = vec![0.0f64; rows];
    for (i, row) in data[..rows * d].chunks_exact(d).enumerate() {
        let mut norm = 0.0;
        for (j, v) in row.iter().enumerate() {
            let c = *v as f64 - mean[j];
            centered[i * d + j] = c;
            norm += c * c;
        }
        norms[i] = norm;
    }
    let mut out = vec![0.0f64; rows * dp];
    dgemm_a_bt(rows, dp, d, 1.0, &centered, proj, 0.0, &mut out);
    (out.iter().map(|v| *v as f32).collect(), norms)
}

/// Upper bound on the norm of a centered vector from its f64 squared norm
/// (`d` subtractions, squares and additions of f64 rounding).
fn centered_norm_upper(norm_sq: f64, d: usize) -> f64 {
    let error = gamma_f64(d.saturating_mul(3).saturating_add(1));
    if error >= 1.0 {
        f64::INFINITY
    } else {
        (norm_sq / (1.0 - error)).sqrt()
    }
}

/// Upper bound on `|P v - (P v)ᶜᵒᵐᵖ|` for a vector of norm at most `norm`
/// whose projection (norm `proj_norm`) was computed by `project_rows`: the f64
/// dot-product rounding over `d` terms in each of `dp` components, plus the
/// final rounding to f32.
fn projection_error(norm: f64, proj_norm: f64, d: usize, dp: usize) -> f64 {
    (dp as f64).sqrt() * gamma_f64(d.saturating_mul(2).saturating_add(1)) * norm
        + (f32::EPSILON as f64 / 2.0) * proj_norm
}

/// Per-row constants of the lower bound, hoisted out of the centroid loop.
struct RowBoundTerms {
    /// `|xp|²` and `|xp|` of the projected centered row (f64).
    xn: f64,
    xn_sqrt: f64,
    /// Bound on the projection error of the row.
    x_error: f64,
    /// `gemm_error_factor · |xp|`; times `|cp|` it bounds the f32 GEMM error.
    gemm_scaled: f64,
    f64_error_factor: f64,
    /// Interval for `|P₀⊥ x|`.
    x_res_lo: f64,
    x_res_hi: f64,
}

/// Lower bound on `|x - c|²` for one row/centroid pair: the projected
/// distance with the f32 GEMM, f64 and projection errors subtracted, plus the
/// reverse triangle inequality on the components outside the subspace
/// (`|P₀⊥(x-c)| >= | |P₀⊥x| - |P₀⊥c| |`, at the worst case of both intervals).
/// Branch-free so the per-row loop vectorizes.
#[inline(always)]
fn pair_lower_bound(
    row: &RowBoundTerms,
    inner_product: f32,
    c_norm: f64,
    c_norm_sqrt: f64,
    c_error: f64,
    c_res_lo: f64,
    c_res_hi: f64,
) -> f64 {
    let ip = inner_product as f64;
    let magnitude = row.xn + c_norm + 2.0 * ip.abs();
    let gemm_error = row.gemm_scaled * c_norm_sqrt;
    let f64_error = row.f64_error_factor * magnitude;
    let projected_sqr = (row.xn + c_norm - 2.0 * ip - gemm_error - f64_error).max(0.0);
    let projection_error = row.x_error + c_error;
    let error_sqr = projection_error * projection_error;
    // sqrt(projected_sqr) <= |xp| + |cp|. Using that upper bound in
    // (sqrt(projected_sqr) - projection_error)^2 keeps this conservative
    // without a square root per pair; it is zero once the error radius
    // covers the projected distance.
    let projected =
        (projected_sqr - 2.0 * projection_error * (row.xn_sqrt + c_norm_sqrt) + error_sqr).max(0.0);
    let projected = if projected_sqr > error_sqr {
        projected
    } else {
        0.0
    };
    let residual = (row.x_res_lo - c_res_hi)
        .max(c_res_lo - row.x_res_hi)
        .max(0.0);
    projected + residual * residual
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

/// Conservative `[sigma_min², sigma_max²]` of the `rows × d` matrix `proj`
/// from the eigenvalues of its Gram matrix, widened by the f64 rounding of
/// the Gram entries (Gershgorin) and a relative margin for the eigensolver.
/// Returns `(0, sigma_max²)` when the rows are numerically dependent.
fn singular_value_bounds(proj: &[f32], rows: usize, d: usize) -> (f64, f64) {
    if rows == 0 {
        return (0.0, 0.0);
    }
    let dot_roundoff = gamma_f64(d.saturating_mul(2).saturating_add(1));
    let mut gram = DMatrix::zeros(rows, rows);
    let mut max_entry_error = 0.0f64;
    for i in 0..rows {
        let mut row_error = 0.0;
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
            gram[(i, j)] = dot;
            row_error += dot_roundoff * absolute_sum;
        }
        max_entry_error = max_entry_error.max(row_error);
    }
    let eigen = SymmetricEigen::new(gram);
    let lambda_max = eigen.eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let lambda_min = eigen
        .eigenvalues
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let solver_margin = lambda_max * 1e-9 + max_entry_error;
    let sigma_max_sq = lambda_max + solver_margin;
    let sigma_min_sq = lambda_min - solver_margin;
    if sigma_min_sq.is_nan() || sigma_min_sq <= 0.0 || !sigma_max_sq.is_finite() {
        (0.0, sigma_max_sq.max(0.0))
    } else {
        (sigma_min_sq, sigma_max_sq)
    }
}

/// Interval for `|P₀⊥ v|` given `|v|²` of the centered vector in f64 (`d`
/// components), `|P v|²` as computed in f32 (`proj_norm_sq`, in f64), a bound
/// `proj_error` on `|P v - (P v)ᶜᵒᵐᵖ|`,
/// and the singular-value bounds of `P`. Every rounding error widens the
/// interval; `sigma_min_sq == 0` yields `[0, |v|]`, i.e. no residual term.
fn residual_interval(
    full_norm_sq: f64,
    d: usize,
    proj_norm_sq: f64,
    proj_error: f64,
    sigma_min_sq: f64,
    sigma_max_sq: f64,
) -> (f64, f64) {
    let full_hi = centered_norm_upper(full_norm_sq, d);
    let full_hi_sq = full_hi * full_hi;
    let full_lo_sq = full_norm_sq / (1.0 + gamma_f64(d.saturating_mul(3).saturating_add(1)));
    let proj_norm = proj_norm_sq.max(0.0).sqrt();
    let proj_hi = proj_norm + proj_error;
    let proj_lo = (proj_norm - proj_error).max(0.0);
    let hi_sq = if sigma_max_sq > 0.0 {
        (full_hi_sq - proj_lo * proj_lo / sigma_max_sq).max(0.0)
    } else {
        full_hi_sq
    };
    let lo_sq = if sigma_min_sq > 0.0 {
        (full_lo_sq - proj_hi * proj_hi / sigma_min_sq).max(0.0)
    } else {
        0.0
    };
    (lo_sq.sqrt(), hi_sq.sqrt().max(lo_sq.sqrt()))
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
                    .map(|(x, y)| *x * *y)
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
        for x in rows.chunks_exact(d).take(32) {
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
    fn forced_calibration_keeps_width_below_min_dp() {
        let (nlist, d) = (4, 16);
        let cents = low_rank_centroids(nlist, d, 2, 0.1, 30);
        let rows = rows_like(&cents, nlist, d, MIN_CALIBRATION_ROWS, 1.0, 31);
        let p =
            CoarseProjection::train(&cents, nlist, d, true, &rows, MIN_CALIBRATION_ROWS).unwrap();
        assert_eq!(p.dimension(), nlist);
        assert_eq!(p.assign(&rows, MIN_CALIBRATION_ROWS, &cents, nlist), {
            rows.chunks_exact(d)
                .map(|x| kmeans::find_nearest(x, &cents, nlist, d))
                .collect::<Vec<_>>()
        });
    }

    #[test]
    fn calibration_picks_a_fast_width_and_stays_exact() {
        let (nlist, d) = (512, 96);
        let cents = low_rank_centroids(nlist, d, 12, 0.3, 31);
        let rows = rows_like(&cents, nlist, d, 4000, 1.0, 32);
        let calibrated = CoarseProjection::train(&cents, nlist, d, true, &rows, 2048).unwrap();
        assert!(calibrated.dimension() <= d / 2);
        let (got, _) = calibrated.assign_with_stats(&rows[2048 * d..], 1952, &cents, nlist);
        let exact: Vec<usize> = rows[2048 * d..]
            .chunks_exact(d)
            .map(|x| kmeans::find_nearest(x, &cents, nlist, d))
            .collect();
        assert_eq!(got, exact);
        assert!(is_fast_enough(
            Duration::from_millis(85),
            Duration::from_millis(100)
        ));
        assert!(!is_fast_enough(
            Duration::from_millis(86),
            Duration::from_millis(100)
        ));
    }

    /// A common offset leaves every distance unchanged; the bound must stay
    /// as tight (rank-8 centroids at `1e8 + (v-3)*64`).
    #[test]
    fn translated_low_rank_data_prunes_as_well_as_centered_data() {
        let (nlist, d, n) = (512, 96, 4096);
        let cents = low_rank_centroids(nlist, d, 8, 0.3, 41);
        let rows = rows_like(&cents, nlist, d, n, 1.0, 42);
        let shift = |v: &[f32]| -> Vec<f32> { v.iter().map(|x| 1e8 + (x - 3.0) * 64.0).collect() };
        let (cents_t, rows_t) = (shift(&cents), shift(&rows));
        // Forced without calibration: elapsed-time width selection is not the subject here.
        let p = CoarseProjection::train(&cents, nlist, d, true, &[], 0).unwrap();
        let p_t = CoarseProjection::train(&cents_t, nlist, d, true, &[], 0).unwrap();
        let (got, evals) = p.assign_with_stats(&rows, n, &cents, nlist);
        let (got_t, evals_t) = p_t.assign_with_stats(&rows_t, n, &cents_t, nlist);
        let argmin_f64 = |rows: &[f32], cents: &[f32]| -> Vec<usize> {
            rows.chunks_exact(d)
                .map(|x| {
                    (0..nlist)
                        .map(|c| {
                            let dist: f64 = x
                                .iter()
                                .zip(&cents[c * d..(c + 1) * d])
                                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                                .sum();
                            (dist, c)
                        })
                        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                        .unwrap()
                        .1
                })
                .collect()
        };
        assert_eq!(got, argmin_f64(&rows, &cents));
        assert_eq!(got_t, argmin_f64(&rows_t, &cents_t));
        assert!(evals < n * nlist / 20, "centered: {evals} checks");
        assert!(
            evals_t <= evals * 2,
            "translated data pruned much worse: {evals_t} vs {evals} checks"
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

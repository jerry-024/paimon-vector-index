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
//! 1. center the row and the centroids and project them with an orthonormal
//!    `P` (`d' × d`);
//! 2. for every centroid compute a lower bound on the true squared distance
//!    `|x - c|² = |P(x-c)|² + |P⊥(x-c)|² ≥ |Px - Pc|² + (|P⊥x| - |P⊥c|)²`
//!    (the second term is the reverse triangle inequality on the residuals,
//!    whose norms are known from `|v|² - |Pv|²`);
//! 3. evaluate exact distances in ascending bound order and stop as soon as
//!    the next bound cannot beat the best exact distance (branch-and-bound).
//!
//! Step 2 is a valid lower bound for *any* orthonormal `P`, so the assignment
//! never depends on how well the PCA converged; `P` only decides how many
//! exact evaluations step 3 needs. Rows without low-dimensional structure
//! degrade to checking every centroid, i.e. the exact scan plus one small
//! GEMM, so automatic mode only keeps a projection when the centroids are
//! compressible enough (`d' ≤ d / 3` at 95% explained variance).
//!
//! The projection is a training artifact: it is derived deterministically from
//! the centroids, shared by `IVFPQIndex::from_trained`, and never serialized.

use crate::blas::sgemm_a_bt;
use crate::distance::{fvec_l2sqr, fvec_norm_l2sqr};
use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Fraction of centroid variance the projection must explain.
const VARIANCE_TARGET: f64 = 0.95;
/// Automatic mode keeps the projection only when `d' * MAX_AUTO_DP_DIVISOR <= d`.
const MAX_AUTO_DP_DIVISOR: usize = 3;
/// Block subspace iterations; the bound stays valid regardless of convergence.
const SUBSPACE_ITERATIONS: usize = 8;
const SUBSPACE_SEED: u64 = 0x7a5e_c7ed;
/// Smallest projection worth a GEMM; also the rounding unit for `d'`.
const MIN_DP: usize = 8;
/// Relative slack applied to every lower bound to absorb f32 rounding in the
/// projected GEMM. Pruning is only ever weakened by it, never the result.
const BOUND_SLACK: f32 = 1e-4;
/// Rows per parallel block; the projected score matrix is `rows × nlist`.
const MAX_BLOCK_ROWS: usize = 1024;
const MAX_BLOCK_SCORES: usize = 4 * 1024 * 1024;

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

/// Orthonormal projection of the coarse centroids plus the per-centroid data
/// the lower bound needs.
#[derive(Debug)]
pub struct CoarseProjection {
    d: usize,
    dp: usize,
    explained_variance: f64,
    /// Centroid mean, subtracted before projecting rows and centroids.
    mean: Vec<f32>,
    /// `dp × d`, orthonormal rows.
    proj: Vec<f32>,
    /// `nlist × dp`: projected centered centroids.
    cents_p: Vec<f32>,
    /// `|cents_p[c]|²`.
    cents_p_norms: Vec<f32>,
    /// `|P⊥(c - mean)|` per centroid.
    cents_res: Vec<f32>,
}

impl CoarseProjection {
    /// Fit a projection to `nlist` centroids of dimension `d`.
    ///
    /// Returns `None` when the projection is not worth it (`force == false`)
    /// or cannot be built: too few centroids, zero variance, or a `d'` that
    /// would not shrink the GEMM by at least `MAX_AUTO_DP_DIVISOR`.
    pub(crate) fn train(cents: &[f32], nlist: usize, d: usize, force: bool) -> Option<Self> {
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
        let explained_variance = eigenvalues[..dp].iter().sum::<f64>() / total_variance;
        let proj = basis[..dp * d].to_vec();

        let mut cents_p = vec![0.0f32; nlist * dp];
        sgemm_a_bt(nlist, dp, d, 1.0, &centered, &proj, 0.0, &mut cents_p);
        let cents_p_norms: Vec<f32> = (0..nlist)
            .map(|c| fvec_norm_l2sqr(&cents_p[c * dp..(c + 1) * dp]))
            .collect();
        let cents_res: Vec<f32> = (0..nlist)
            .map(|c| {
                let full = fvec_norm_l2sqr(&centered[c * d..(c + 1) * d]);
                (full - cents_p_norms[c]).max(0.0).sqrt()
            })
            .collect();
        Some(Self {
            d,
            dp,
            explained_variance,
            mean,
            proj,
            cents_p,
            cents_p_norms,
            cents_res,
        })
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
        debug_assert_eq!(self.cents_p.len(), nlist * dp);
        let mut out = vec![0usize; n];
        let block_rows = (MAX_BLOCK_SCORES / nlist.max(1)).clamp(1, MAX_BLOCK_ROWS);
        let evaluations = out
            .par_chunks_mut(block_rows)
            .enumerate()
            .map(|(bi, chunk)| {
                let rows = chunk.len();
                let row0 = bi * block_rows;
                let x = &data[row0 * d..(row0 + rows) * d];
                let centered: Vec<f32> = x
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v - self.mean[i % d])
                    .collect();
                let mut xp = vec![0.0f32; rows * dp];
                sgemm_a_bt(rows, dp, d, 1.0, &centered, &self.proj, 0.0, &mut xp);
                let mut ip = vec![0.0f32; rows * nlist];
                sgemm_a_bt(rows, nlist, dp, 1.0, &xp, &self.cents_p, 0.0, &mut ip);

                let mut bounds = vec![0.0f32; nlist];
                let mut candidates: Vec<(f32, u32)> = Vec::new();
                let mut evaluations = 0usize;
                for i in 0..rows {
                    let xp_i = &xp[i * dp..(i + 1) * dp];
                    let xn = fvec_norm_l2sqr(xp_i);
                    let x_res = (fvec_norm_l2sqr(&centered[i * d..(i + 1) * d]) - xn)
                        .max(0.0)
                        .sqrt();
                    let ip_i = &ip[i * nlist..(i + 1) * nlist];
                    let mut first = 0usize;
                    for c in 0..nlist {
                        let projected = (xn + self.cents_p_norms[c] - 2.0 * ip_i[c]).max(0.0);
                        let residual = x_res - self.cents_res[c];
                        let bound = (projected + residual * residual) * (1.0 - BOUND_SLACK);
                        bounds[c] = bound;
                        if bound < bounds[first] {
                            first = c;
                        }
                    }
                    let x_i = &x[i * d..(i + 1) * d];
                    let mut best = fvec_l2sqr(x_i, &cents[first * d..(first + 1) * d]);
                    let mut best_idx = first;
                    evaluations += 1;
                    candidates.clear();
                    for (c, &bound) in bounds.iter().enumerate() {
                        if c != first && bound <= best {
                            candidates.push((bound, c as u32));
                        }
                    }
                    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    for &(bound, c) in &candidates {
                        if bound > best {
                            break;
                        }
                        let c = c as usize;
                        let dist = fvec_l2sqr(x_i, &cents[c * d..(c + 1) * d]);
                        evaluations += 1;
                        if dist < best || (dist == best && c < best_idx) {
                            best = dist;
                            best_idx = c;
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
    fn lower_bound(&self, x: &[f32], c: usize) -> f32 {
        let d = self.d;
        let dp = self.dp;
        let centered: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v - self.mean[i])
            .collect();
        let mut xp = vec![0.0f32; dp];
        sgemm_a_bt(1, dp, d, 1.0, &centered, &self.proj, 0.0, &mut xp);
        let xn = fvec_norm_l2sqr(&xp);
        let x_res = (fvec_norm_l2sqr(&centered) - xn).max(0.0).sqrt();
        let projected = fvec_l2sqr(&xp, &self.cents_p[c * dp..(c + 1) * dp]);
        let residual = x_res - self.cents_res[c];
        (projected + residual * residual) * (1.0 - BOUND_SLACK)
    }
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
    fn projection_rows_are_orthonormal() {
        let (nlist, d) = (512, 96);
        let cents = low_rank_centroids(nlist, d, 12, 0.05, 1);
        let p = CoarseProjection::train(&cents, nlist, d, true).unwrap();
        for a in 0..p.dp {
            for b in 0..p.dp {
                let dot: f32 = p.proj[a * d..(a + 1) * d]
                    .iter()
                    .zip(&p.proj[b * d..(b + 1) * d])
                    .map(|(x, y)| x * y)
                    .sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-5, "row {a}·row {b} = {dot}");
            }
        }
    }

    #[test]
    fn auto_mode_keeps_compressible_centroids_and_rejects_noise() {
        let (nlist, d) = (512, 96);
        let structured = low_rank_centroids(nlist, d, 12, 0.05, 2);
        let p = CoarseProjection::train(&structured, nlist, d, false).unwrap();
        assert!(p.dimension() * MAX_AUTO_DP_DIVISOR <= d);
        assert!(p.explained_variance() >= VARIANCE_TARGET);

        let mut rng = StdRng::seed_from_u64(3);
        let noise: Vec<f32> = (0..nlist * d).map(|_| rng.gen::<f32>()).collect();
        assert!(CoarseProjection::train(&noise, nlist, d, false).is_none());
        // Forced mode still builds one (and stays exact).
        assert!(CoarseProjection::train(&noise, nlist, d, true).is_some());
    }

    #[test]
    fn train_rejects_degenerate_inputs() {
        assert!(CoarseProjection::train(&[], 0, 16, true).is_none());
        assert!(CoarseProjection::train(&vec![1.0; 4 * 16], 4, 16, false).is_none());
        assert!(CoarseProjection::train(&vec![1.0; 64 * 16], 64, 16, true).is_none());
    }

    #[test]
    fn lower_bound_never_exceeds_distance() {
        let (nlist, d) = (256, 64);
        let cents = low_rank_centroids(nlist, d, 8, 0.3, 4);
        let p = CoarseProjection::train(&cents, nlist, d, true).unwrap();
        let rows = rows_like(&cents, nlist, d, 200, 1.0, 5);
        for i in 0..200 {
            let x = &rows[i * d..(i + 1) * d];
            for c in 0..nlist {
                let dist = fvec_l2sqr(x, &cents[c * d..(c + 1) * d]);
                let bound = p.lower_bound(x, c);
                assert!(
                    bound <= dist * (1.0 + 1e-5) + 1e-6,
                    "row {i} c {c}: {bound} > {dist}"
                );
            }
        }
    }

    #[test]
    fn assignment_matches_exact_scan_and_prunes() {
        let (nlist, d, n) = (512, 96, 3000);
        let cents = low_rank_centroids(nlist, d, 12, 0.05, 6);
        let rows = rows_like(&cents, nlist, d, n, 0.4, 7);
        let p = CoarseProjection::train(&cents, nlist, d, false).unwrap();
        let (got, evaluations) = p.assign_with_stats(&rows, n, &cents, nlist);
        let exact = kmeans::find_nearest_batch(&rows, n, &cents, nlist, d);
        for i in 0..n {
            if got[i] != exact[i] {
                let x = &rows[i * d..(i + 1) * d];
                let a = fvec_l2sqr(x, &cents[got[i] * d..(got[i] + 1) * d]);
                let b = fvec_l2sqr(x, &cents[exact[i] * d..(exact[i] + 1) * d]);
                assert!(
                    (a - b).abs() <= 1e-5 * b.max(1e-12),
                    "row {i}: {} vs {}",
                    got[i],
                    exact[i]
                );
            }
        }
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
        let p = CoarseProjection::train(&cents, nlist, d, true).unwrap();
        let got = p.assign(&rows, n, &cents, nlist);
        let exact = kmeans::find_nearest_batch(&rows, n, &cents, nlist, d);
        for i in 0..n {
            if got[i] != exact[i] {
                let x = &rows[i * d..(i + 1) * d];
                let a = fvec_l2sqr(x, &cents[got[i] * d..(got[i] + 1) * d]);
                let b = fvec_l2sqr(x, &cents[exact[i] * d..(exact[i] + 1) * d]);
                assert!((a - b).abs() <= 1e-5 * b.max(1e-12), "row {i}");
            }
        }
    }

    #[test]
    fn ties_pick_smallest_centroid_index() {
        let (nlist, d) = (64, 16);
        let mut cents = low_rank_centroids(nlist, d, 4, 0.1, 9);
        // Duplicate centroid 40 at index 3 and 50.
        let dup: Vec<f32> = cents[40 * d..41 * d].to_vec();
        cents[3 * d..4 * d].copy_from_slice(&dup);
        cents[50 * d..51 * d].copy_from_slice(&dup);
        let p = CoarseProjection::train(&cents, nlist, d, true).unwrap();
        let mut row = dup.clone();
        row[0] += 0.01;
        let got = p.assign(&row, 1, &cents, nlist);
        assert_eq!(got, vec![3]);
    }
}

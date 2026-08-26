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

//! IVF-PQ add benchmark using one fixed trained index.
//!
//! Configure with `BENCH_N`, `BENCH_D`, `BENCH_NLIST`, `BENCH_M`,
//! `BENCH_TRAIN_N`, `BENCH_BATCH_SIZE`, and `BENCH_REPEATS`. Thread count
//! follows `RAYON_NUM_THREADS`.

use paimon_vindex_core::distance::MetricType;
use paimon_vindex_core::ivfpq::IVFPQIndex;
use paimon_vindex_core::kmeans::find_nearest;
use paimon_vindex_core::projected_assign::ProjectedAssignment;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

fn main() {
    let n = env_usize("BENCH_N", 50_000);
    let d = env_usize("BENCH_D", 128);
    let nlist = env_usize("BENCH_NLIST", 1024);
    let m = env_usize("BENCH_M", 16);
    let train_n = env_usize("BENCH_TRAIN_N", n.min(nlist * 64));
    let batch_size = env_usize("BENCH_BATCH_SIZE", 100_000);
    let repeats = env_usize("BENCH_REPEATS", 3);
    assert!(batch_size > 0);

    let mut train_rng = StdRng::seed_from_u64(20260821);
    let train_data = generate_normalized_vectors(train_n, d, &mut train_rng);
    let mut trained = IVFPQIndex::new(d, nlist, m, MetricType::InnerProduct, false);
    match std::env::var("BENCH_ASSIGN").as_deref() {
        Ok("exact") => trained = trained.with_projected_assignment(ProjectedAssignment::Disabled),
        Ok("projected") => {
            trained = trained.with_projected_assignment(ProjectedAssignment::Enabled)
        }
        _ => {}
    }
    let train_started = Instant::now();
    trained.train(&train_data, train_n);
    let train_seconds = train_started.elapsed().as_secs_f64();
    let projection = trained
        .coarse_projection()
        .map(|p| {
            format!(
                "d'={} ({:.1}% variance)",
                p.dimension(),
                p.explained_variance() * 100.0
            )
        })
        .unwrap_or_else(|| "none (exact scan)".to_string());
    eprintln!("train_seconds={train_seconds:.3} projection={projection}");

    println!("n,d,nlist,m,train_n,batch_size,threads,run,add_seconds,rows_per_second");
    for run in 1..=repeats {
        let mut index = IVFPQIndex::from_trained(&trained);
        let mut data_rng = StdRng::seed_from_u64(20260822);
        let mut add_elapsed = Duration::ZERO;
        for start in (0..n).step_by(batch_size) {
            let rows = batch_size.min(n - start);
            let data = generate_normalized_vectors(rows, d, &mut data_rng);
            let ids = (start as i64..(start + rows) as i64).collect::<Vec<_>>();
            let started = Instant::now();
            index.add(&data, &ids, rows);
            add_elapsed += started.elapsed();
        }
        let add_seconds = add_elapsed.as_secs_f64();
        assert_eq!(index.ids.iter().map(Vec::len).sum::<usize>(), n);
        report_agreement(&index, n, d, nlist, batch_size);
        println!(
            "{n},{d},{nlist},{m},{train_n},{batch_size},{},{run},{add_seconds:.6},{:.0}",
            rayon::current_num_threads(),
            n as f64 / add_seconds
        );
    }
}

/// Buckets of the first batch versus the exact scan. Both paths are exact,
/// so anything but a rounding-level near-tie is a bug.
fn report_agreement(index: &IVFPQIndex, n: usize, d: usize, nlist: usize, batch_size: usize) {
    let check = batch_size.min(n).min(20_000);
    let mut rng = StdRng::seed_from_u64(20260822);
    let first = generate_normalized_vectors(check, d, &mut rng);
    let mut bucket_of = vec![usize::MAX; check];
    for (list_id, ids_in_list) in index.ids.iter().enumerate() {
        for &id in ids_in_list {
            if (id as usize) < check {
                bucket_of[id as usize] = list_id;
            }
        }
    }
    let cents = index.quantizer_centroids();
    let agree = (0..check)
        .filter(|&i| find_nearest(&first[i * d..(i + 1) * d], cents, nlist, d) == bucket_of[i])
        .count();
    eprintln!(
        "agreement_with_exact_scan={:.5}",
        agree as f64 / check as f64
    );
}

/// `BENCH_DATA=uniform` (default): i.i.d. uniform coordinates, the worst case
/// for any projection. `BENCH_DATA=embedding`: rank-32 manifold plus 5%
/// noise, the shape of real embeddings.
fn generate_normalized_vectors(n: usize, d: usize, rng: &mut StdRng) -> Vec<f32> {
    let embedding_like = std::env::var("BENCH_DATA").is_ok_and(|v| v == "embedding");
    let rank = 32usize;
    let basis: Vec<f32> = {
        let mut basis_rng = StdRng::seed_from_u64(7);
        (0..rank * d)
            .map(|_| basis_rng.gen::<f32>() - 0.5)
            .collect()
    };
    let mut latent = vec![0.0f32; rank];
    let mut data = vec![0.0f32; n * d];
    for row in data.chunks_mut(d) {
        if embedding_like {
            for z in latent.iter_mut() {
                *z = rng.gen::<f32>() - 0.5;
            }
            for (j, value) in row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (t, z) in latent.iter().enumerate() {
                    acc += z * basis[t * d + j];
                }
                *value = acc + (rng.gen::<f32>() - 0.5) * 0.05;
            }
        } else {
            for value in row.iter_mut() {
                *value = rng.gen::<f32>() * 2.0 - 1.0;
            }
        }
        let norm_sq: f32 = row.iter().map(|v| v * v).sum();
        let inv_norm = norm_sq.sqrt().recip();
        for value in row {
            *value *= inv_norm;
        }
    }
    data
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("invalid {name}: {value}"))
        })
        .unwrap_or(default)
}

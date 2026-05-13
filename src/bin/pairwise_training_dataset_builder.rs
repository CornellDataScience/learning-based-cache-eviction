use std::collections::HashMap;
use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use lbce::core::cache::Cache;
use lbce::core::trace::{Request, RequestTrace};

use lbce::data::memory_builder::MainMemoryBuilder;
use lbce::data::pairwise_csv_writer::PairwiseCsvWriter;
use lbce::data::pairwise_samples::{
    PairwiseDatasetConfig, PairwiseDatasetGenerator, PairwiseSample,
};
use lbce::data::wiki_trace_loader;
use lbce::policies::belady::BeladyPolicy;

use lbce::workloads::bursty::BurstyWorkload;
use lbce::workloads::looping::LoopingWorkload;
use lbce::workloads::phase::PhaseWorkload;
use lbce::workloads::workload::Workload;
use lbce::workloads::zipf::ZipfWorkload;

#[derive(Clone, Debug)]
pub struct TrainingBuildConfig {
    pub seed: u64,
    pub cache_sizes: Vec<usize>,
    pub default_object_size: usize,

    pub max_samples_total: usize,
    pub max_samples_per_trace_run: usize,
    pub max_pairs_per_decision: usize,

    pub enable_bursty: bool,
    pub enable_looping: bool,
    pub enable_phase: bool,
    pub enable_zipf: bool,
    pub enable_mixed: bool,

    pub pairwise: PairwiseDatasetConfig,
}

impl Default for TrainingBuildConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            cache_sizes: vec![16, 32, 64],
            default_object_size: 128,
            max_samples_total: 150_000,
            max_samples_per_trace_run: 10_000,
            max_pairs_per_decision: 64,
            enable_bursty: true,
            enable_looping: true,
            enable_phase: true,
            enable_zipf: true,
            enable_mixed: true,
            pairwise: PairwiseDatasetConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NamedTrace {
    pub name: String,
    pub trace: RequestTrace,
}

pub struct TrainingDatasetBuilder;

impl TrainingDatasetBuilder {
    const SYNTHETIC_TARGET_SHARE: f32 = 0.35;
    const MIXED_TARGET_SHARE: f32 = 0.45;
    const REAL_TARGET_SHARE: f32 = 0.20;

    pub fn build_pairwise_training_dataset<const MM_SIZE: usize>(
        config: &TrainingBuildConfig,
        real_traces: &[NamedTrace],
    ) -> Vec<PairwiseSample> {
        println!("🚀 Starting pairwise dataset generation...");

        let mut rng = StdRng::seed_from_u64(config.seed);

        let (synthetic_traces, mixed_traces) = Self::build_training_trace_groups(config, &mut rng);
        println!(
            "📊 Trace groups: synthetic={} mixed={} real={}",
            synthetic_traces.len(),
            mixed_traces.len(),
            real_traces.len()
        );

        let synthetic_samples = Self::materialize_group::<MM_SIZE>(
            "synthetic",
            &synthetic_traces,
            config,
            &mut rng,
        );
        let mixed_samples = Self::materialize_group::<MM_SIZE>(
            "mixed",
            &mixed_traces,
            config,
            &mut rng,
        );
        let real_samples =
            Self::materialize_group::<MM_SIZE>("real", real_traces, config, &mut rng);

        let balanced = Self::rebalance_groups(
            vec![
                (
                    "synthetic",
                    synthetic_samples,
                    Self::SYNTHETIC_TARGET_SHARE,
                ),
                ("mixed", mixed_samples, Self::MIXED_TARGET_SHARE),
                ("real", real_samples, Self::REAL_TARGET_SHARE),
            ],
            config.max_samples_total,
            &mut rng,
        );

        println!("\n✅ Final dataset size: {} samples", balanced.len());
        balanced
    }

    pub fn build_and_write_pairwise_training_csv<const MM_SIZE: usize, P: AsRef<Path>>(
        config: &TrainingBuildConfig,
        real_traces: &[NamedTrace],
        output_path: P,
    ) -> std::io::Result<()> {
        let samples = Self::build_pairwise_training_dataset::<MM_SIZE>(config, real_traces);

        println!("\n💾 Writing dataset to CSV...");

        PairwiseCsvWriter::write_to_path(
            output_path,
            &samples,
            config.pairwise.decay_factors.len(),
        )?;

        println!("✅ CSV write complete");

        Ok(())
    }

    fn build_training_trace_groups(
        config: &TrainingBuildConfig,
        rng: &mut StdRng,
    ) -> (Vec<NamedTrace>, Vec<NamedTrace>) {
        let mut synthetic = Vec::new();

        if config.enable_bursty {
            synthetic.extend(Self::make_bursty_traces());
        }
        if config.enable_looping {
            synthetic.extend(Self::make_looping_traces());
        }
        if config.enable_phase {
            synthetic.extend(Self::make_phase_traces());
        }
        if config.enable_zipf {
            synthetic.extend(Self::make_zipf_traces(config.seed));
        }

        let mixed = if config.enable_mixed {
            Self::make_mixed_traces(&synthetic, rng)
        } else {
            Vec::new()
        };

        (synthetic, mixed)
    }

    fn materialize_group<const MM_SIZE: usize>(
        label: &str,
        traces: &[NamedTrace],
        config: &TrainingBuildConfig,
        rng: &mut StdRng,
    ) -> Vec<PairwiseSample> {
        if traces.is_empty() {
            println!("   {} group: no traces", label);
            return Vec::new();
        }

        let mut all_samples = Vec::new();
        println!("\n📦 Building {label} samples from {} traces", traces.len());

        for named_trace in traces {
            println!(
                "   🔹 Processing trace: {} (len = {})",
                named_trace.name,
                named_trace.trace.len()
            );

            for &cache_size in &config.cache_sizes {
                println!("      → Cache size: {}", cache_size);

                let mm = MainMemoryBuilder::from_trace::<MM_SIZE>(
                    &named_trace.trace,
                    config.default_object_size,
                );

                let mut cache = Cache::new(
                    cache_size,
                    BeladyPolicy::new(&named_trace.trace),
                    mm,
                );

                let samples = PairwiseDatasetGenerator::generate(
                    &named_trace.name,
                    &named_trace.trace,
                    &mut cache,
                    &config.pairwise,
                );

                println!("         Raw samples: {}", samples.len());

                let sampled = Self::downsample_trace_run_samples(
                    samples,
                    config.max_pairs_per_decision,
                    config.max_samples_per_trace_run,
                    rng,
                );

                println!("         Kept samples: {}", sampled.len());
                all_samples.extend(sampled);
            }
        }

        println!("   ✅ {label} sample count: {}", all_samples.len());
        all_samples
    }

    fn rebalance_groups(
        groups: Vec<(&'static str, Vec<PairwiseSample>, f32)>,
        max_samples_total: usize,
        rng: &mut StdRng,
    ) -> Vec<PairwiseSample> {
        let mut active: Vec<(&'static str, Vec<PairwiseSample>, f32)> = groups
            .into_iter()
            .filter(|(_, samples, _)| !samples.is_empty())
            .collect();

        if active.is_empty() {
            return Vec::new();
        }

        for (_, samples, _) in &mut active {
            samples.shuffle(rng);
        }

        let available_total: usize = active.iter().map(|(_, samples, _)| samples.len()).sum();
        let target_total = available_total.min(max_samples_total);
        let total_weight: f32 = active.iter().map(|(_, _, weight)| *weight).sum();

        let mut selected_counts = vec![0usize; active.len()];
        let mut selected_total = 0usize;

        for (idx, (_, samples, weight)) in active.iter().enumerate() {
            let normalized = if total_weight > 0.0 {
                *weight / total_weight
            } else {
                1.0 / active.len() as f32
            };
            let desired = ((target_total as f32) * normalized).round() as usize;
            let count = desired.min(samples.len());
            selected_counts[idx] = count;
            selected_total += count;
        }

        while selected_total > target_total {
            if let Some((idx, _)) = selected_counts
                .iter()
                .enumerate()
                .filter(|(_, count)| **count > 0)
                .max_by_key(|(_, count)| **count)
            {
                selected_counts[idx] -= 1;
                selected_total -= 1;
            } else {
                break;
            }
        }

        while selected_total < target_total {
            let mut progressed = false;

            for idx in 0..active.len() {
                if selected_total >= target_total {
                    break;
                }

                if selected_counts[idx] < active[idx].1.len() {
                    selected_counts[idx] += 1;
                    selected_total += 1;
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }

        let mut balanced = Vec::with_capacity(selected_total);

        for ((label, mut samples, _), count) in active.into_iter().zip(selected_counts.into_iter()) {
            println!("   🎯 keeping {} {} samples", count, label);
            samples.truncate(count);
            balanced.extend(samples);
        }

        balanced.shuffle(rng);
        balanced
    }

    fn make_bursty_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "bursty_small".to_string(),
                trace: Self::materialize_workload(BurstyWorkload::new(40, 40, 10, 20)),
            },
            NamedTrace {
                name: "bursty_medium".to_string(),
                trace: Self::materialize_workload(BurstyWorkload::new(60, 60, 20, 40)),
            },
        ]
    }

    fn make_looping_traces() -> Vec<NamedTrace> {
        vec![NamedTrace {
            name: "looping".to_string(),
            trace: Self::materialize_workload(LoopingWorkload::new((0..64).collect(), 4000)),
        }]
    }

    fn make_phase_traces() -> Vec<NamedTrace> {
        vec![NamedTrace {
            name: "phase".to_string(),
            trace: Self::materialize_workload(PhaseWorkload::new(6, 32, 500)),
        }]
    }

    fn make_zipf_traces(seed: u64) -> Vec<NamedTrace> {
        vec![NamedTrace {
            name: "zipf".to_string(),
            trace: Self::materialize_workload(ZipfWorkload::new(
                (0..128).collect(),
                5000,
                1.1,
                seed,
            )),
        }]
    }

    fn make_mixed_traces(existing: &[NamedTrace], rng: &mut StdRng) -> Vec<NamedTrace> {
        if existing.len() < 2 {
            return Vec::new();
        }

        let mut picked: Vec<RequestTrace> = existing
            .choose_multiple(rng, 2)
            .map(|nt| nt.trace.clone())
            .collect();

        let concat = Self::concat_traces(&picked);

        picked.shuffle(rng);
        let interleaved = Self::interleave_traces_round_robin(&picked);

        vec![
            NamedTrace {
                name: "mixed_concat".to_string(),
                trace: concat,
            },
            NamedTrace {
                name: "mixed_interleaved".to_string(),
                trace: interleaved,
            },
        ]
    }

    pub fn materialize_workload<W: Workload>(mut workload: W) -> RequestTrace {
        let mut trace = RequestTrace::new();
        while let Some(key) = workload.next_request() {
            trace.push(Request::new(key));
        }
        trace
    }

    pub fn concat_traces(traces: &[RequestTrace]) -> RequestTrace {
        let mut out = RequestTrace::new();
        for trace in traces {
            for req in trace.requests() {
                out.push(*req);
            }
        }
        out
    }

    pub fn interleave_traces_round_robin(traces: &[RequestTrace]) -> RequestTrace {
        let mut out = RequestTrace::new();
        let mut positions = vec![0usize; traces.len()];
        let mut remaining = true;

        while remaining {
            remaining = false;
            for (i, trace) in traces.iter().enumerate() {
                if positions[i] < trace.len() {
                    out.push(trace.requests()[positions[i]]);
                    positions[i] += 1;
                    remaining = true;
                }
            }
        }

        out
    }

    fn downsample_trace_run_samples(
        samples: Vec<PairwiseSample>,
        max_pairs_per_decision: usize,
        max_samples_per_trace_run: usize,
        rng: &mut StdRng,
    ) -> Vec<PairwiseSample> {
        let mut by_tick: HashMap<u64, Vec<PairwiseSample>> = HashMap::new();

        for sample in samples {
            by_tick.entry(sample.tick).or_default().push(sample);
        }

        let mut kept = Vec::new();

        for (_, mut group) in by_tick {
            group.shuffle(rng);
            if group.len() > max_pairs_per_decision {
                group.truncate(max_pairs_per_decision);
            }
            kept.extend(group);
        }

        kept.shuffle(rng);

        if kept.len() > max_samples_per_trace_run {
            kept.truncate(max_samples_per_trace_run);
        }

        kept
    }
}

fn main() {
    const MM_SIZE: usize = 200_000;

    let config = TrainingBuildConfig::default();

    // Any extra CLI args are treated as paths to Wikipedia cache trace files.
    let real_traces: Vec<NamedTrace> = std::env::args().skip(1).filter_map(|arg| {
        let path = std::path::PathBuf::from(&arg);
        let name = path.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| arg.clone());
        match wiki_trace_loader::load(&path) {
            Ok(wt) => {
                println!("📂 Loaded wiki trace '{}': {} requests", name, wt.trace.len());
                Some(NamedTrace { name, trace: wt.trace })
            }
            Err(e) => {
                eprintln!("⚠️  Skipping '{}': {}", arg, e);
                None
            }
        }
    }).collect();

    match TrainingDatasetBuilder::build_and_write_pairwise_training_csv::<MM_SIZE, _>(
        &config,
        &real_traces,
        "artifacts/datasets/pairwise_training_dataset.csv",
    ) {
        Ok(()) => println!("\n🎉 Done! Dataset ready."),
        Err(e) => {
            eprintln!("❌ Failed: {}", e);
            std::process::exit(1);
        }
    }
}

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
pub struct EvalBuildConfig {
    pub seed: u64,
    pub cache_sizes: Vec<usize>,
    pub default_object_size: usize,

    pub max_samples_total: usize,
    pub max_samples_per_trace_run: usize,
    pub max_pairs_per_decision: usize,

    pub pairwise: PairwiseDatasetConfig,
}

impl Default for EvalBuildConfig {
    fn default() -> Self {
        Self {
            seed: 999,
            cache_sizes: vec![16, 32, 64],
            default_object_size: 128,
            max_samples_total: 80_000,
            max_samples_per_trace_run: 6_000,
            max_pairs_per_decision: 48,
            pairwise: PairwiseDatasetConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NamedTrace {
    pub name: String,
    pub trace: RequestTrace,
}

pub struct EvalDatasetBuilder;

impl EvalDatasetBuilder {
    const SYNTHETIC_TARGET_SHARE: f32 = 0.35;
    const MIXED_TARGET_SHARE: f32 = 0.45;
    const REAL_TARGET_SHARE: f32 = 0.20;

    pub fn build_validation_dataset<const MM_SIZE: usize>(
        config: &EvalBuildConfig,
        real_traces: &[NamedTrace],
    ) -> Vec<PairwiseSample> {
        println!("🧪 Building validation dataset...");
        let mut rng = StdRng::seed_from_u64(config.seed);

        let (synthetic_traces, mixed_traces) = Self::build_validation_trace_groups(config, &mut rng);
        println!(
            "   Validation trace groups: synthetic={} mixed={} real={}",
            synthetic_traces.len(),
            mixed_traces.len(),
            real_traces.len()
        );

        let synthetic_samples = Self::materialize_samples::<MM_SIZE>(
            synthetic_traces,
            config,
            &mut rng,
            false,
        );
        let mixed_samples = Self::materialize_samples::<MM_SIZE>(
            mixed_traces,
            config,
            &mut rng,
            false,
        );
        let real_samples = Self::materialize_samples::<MM_SIZE>(
            real_traces.to_vec(),
            config,
            &mut rng,
            false,
        );

        Self::rebalance_groups(
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
        )
    }

    pub fn write_validation_csv<const MM_SIZE: usize, P: AsRef<Path>>(
        config: &EvalBuildConfig,
        real_traces: &[NamedTrace],
        output_path: P,
    ) -> std::io::Result<()> {
        let samples = Self::build_validation_dataset::<MM_SIZE>(config, real_traces);
        PairwiseCsvWriter::write_to_path(output_path, &samples, config.pairwise.decay_factors.len())
    }

    pub fn build_test_sets<const MM_SIZE: usize>(
        config: &EvalBuildConfig,
        real_traces: &[NamedTrace],
    ) -> HashMap<String, Vec<PairwiseSample>> {
        println!("🧪 Building test datasets...");
        let mut rng = StdRng::seed_from_u64(config.seed);

        let mut out = HashMap::new();

        let bursty = Self::make_test_bursty_traces();
        let looping = Self::make_test_looping_traces();
        let phase = Self::make_test_phase_traces();
        let zipf = Self::make_test_zipf_traces(config.seed);

        let pooled_components =
            [bursty.clone(), looping.clone(), phase.clone(), zipf.clone()].concat();

        let mixed_concat = vec![NamedTrace {
            name: "test_mixed_concat".to_string(),
            trace: Self::concat_traces(
                &pooled_components
                    .choose_multiple(&mut rng, 4)
                    .map(|nt| nt.trace.clone())
                    .collect::<Vec<_>>(),
            ),
        }];

        let mixed_interleaved = vec![NamedTrace {
            name: "test_mixed_interleaved".to_string(),
            trace: Self::interleave_traces_round_robin(
                &pooled_components
                    .choose_multiple(&mut rng, 4)
                    .map(|nt| nt.trace.clone())
                    .collect::<Vec<_>>(),
            ),
        }];

        out.insert(
            "test_bursty".to_string(),
            Self::materialize_samples::<MM_SIZE>(bursty.clone(), config, &mut rng, false),
        );
        out.insert(
            "test_looping".to_string(),
            Self::materialize_samples::<MM_SIZE>(looping.clone(), config, &mut rng, false),
        );
        out.insert(
            "test_phase".to_string(),
            Self::materialize_samples::<MM_SIZE>(phase.clone(), config, &mut rng, false),
        );
        out.insert(
            "test_zipf".to_string(),
            Self::materialize_samples::<MM_SIZE>(zipf.clone(), config, &mut rng, false),
        );
        out.insert(
            "test_mixed_concat".to_string(),
            Self::materialize_samples::<MM_SIZE>(mixed_concat.clone(), config, &mut rng, false),
        );
        out.insert(
            "test_mixed_interleaved".to_string(),
            Self::materialize_samples::<MM_SIZE>(
                mixed_interleaved.clone(),
                config,
                &mut rng,
                false,
            ),
        );

        let pooled_synthetic = [bursty, looping, phase, zipf].concat();
        let pooled_mixed = [mixed_concat, mixed_interleaved].concat();

        out.insert(
            "test_pooled".to_string(),
            Self::rebalance_groups(
                vec![
                    (
                        "synthetic",
                        Self::materialize_samples::<MM_SIZE>(
                            pooled_synthetic,
                            config,
                            &mut rng,
                            false,
                        ),
                        Self::SYNTHETIC_TARGET_SHARE,
                    ),
                    (
                        "mixed",
                        Self::materialize_samples::<MM_SIZE>(
                            pooled_mixed,
                            config,
                            &mut rng,
                            false,
                        ),
                        Self::MIXED_TARGET_SHARE,
                    ),
                    (
                        "real",
                        Self::materialize_samples::<MM_SIZE>(
                            real_traces.to_vec(),
                            config,
                            &mut rng,
                            false,
                        ),
                        Self::REAL_TARGET_SHARE,
                    ),
                ],
                config.max_samples_total,
                &mut rng,
            ),
        );

        for nt in real_traces {
            let samples = Self::materialize_samples::<MM_SIZE>(
                vec![nt.clone()],
                config,
                &mut rng,
                false,
            );
            out.insert(format!("wiki_{}", nt.name), samples);
        }

        if !real_traces.is_empty() {
            out.insert(
                "test_real_only".to_string(),
                Self::materialize_samples::<MM_SIZE>(
                    real_traces.to_vec(),
                    config,
                    &mut rng,
                    true,
                ),
            );
        }

        out
    }

    pub fn write_test_csvs<const MM_SIZE: usize, P: AsRef<Path>>(
        config: &EvalBuildConfig,
        real_traces: &[NamedTrace],
        output_dir: P,
    ) -> std::io::Result<()> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let datasets = Self::build_test_sets::<MM_SIZE>(config, real_traces);

        for (name, samples) in datasets {
            let path = output_dir.join(format!("{name}.csv"));
            println!("💾 Writing {}", path.display());
            PairwiseCsvWriter::write_to_path(path, &samples, config.pairwise.decay_factors.len())?;
        }

        Ok(())
    }

    fn materialize_samples<const MM_SIZE: usize>(
        traces: Vec<NamedTrace>,
        config: &EvalBuildConfig,
        rng: &mut StdRng,
        shuffle_and_cap_global: bool,
    ) -> Vec<PairwiseSample> {
        let mut all_samples = Vec::new();

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

        if shuffle_and_cap_global {
            all_samples.shuffle(rng);
            if all_samples.len() > config.max_samples_total {
                all_samples.truncate(config.max_samples_total);
            }
        }

        println!("   ✅ Final dataset size: {}", all_samples.len());
        all_samples
    }

    fn build_validation_trace_groups(
        config: &EvalBuildConfig,
        rng: &mut StdRng,
    ) -> (Vec<NamedTrace>, Vec<NamedTrace>) {
        let mut synthetic = Vec::new();
        synthetic.extend(Self::make_validation_bursty_traces());
        synthetic.extend(Self::make_validation_looping_traces());
        synthetic.extend(Self::make_validation_phase_traces());
        synthetic.extend(Self::make_validation_zipf_traces(config.seed));

        let mixed = Self::make_validation_mixed_traces(&synthetic, rng);

        (synthetic, mixed)
    }

    fn make_validation_bursty_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "val_bursty_a".to_string(),
                trace: Self::materialize_workload(BurstyWorkload::new(45, 35, 15, 24)),
            },
            NamedTrace {
                name: "val_bursty_b".to_string(),
                trace: Self::materialize_workload(BurstyWorkload::new(55, 80, 12, 36)),
            },
        ]
    }

    fn make_validation_looping_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "val_looping_a".to_string(),
                trace: Self::materialize_workload(LoopingWorkload::new((0..48).collect(), 2500)),
            },
            NamedTrace {
                name: "val_looping_b".to_string(),
                trace: Self::materialize_workload(LoopingWorkload::new((0..160).collect(), 5000)),
            },
        ]
    }

    fn make_validation_phase_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "val_phase_a".to_string(),
                trace: Self::materialize_workload(PhaseWorkload::new(7, 24, 350)),
            },
            NamedTrace {
                name: "val_phase_b".to_string(),
                trace: Self::materialize_workload(PhaseWorkload::new(9, 40, 450)),
            },
        ]
    }

    fn make_validation_zipf_traces(base_seed: u64) -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "val_zipf_a".to_string(),
                trace: Self::materialize_workload(ZipfWorkload::new(
                    (0..160).collect(),
                    4500,
                    0.9,
                    base_seed + 101,
                )),
            },
            NamedTrace {
                name: "val_zipf_b".to_string(),
                trace: Self::materialize_workload(ZipfWorkload::new(
                    (0..300).collect(),
                    5500,
                    1.25,
                    base_seed + 102,
                )),
            },
        ]
    }

    fn make_validation_mixed_traces(existing: &[NamedTrace], rng: &mut StdRng) -> Vec<NamedTrace> {
        let picked: Vec<RequestTrace> = existing
            .choose_multiple(rng, 4)
            .map(|nt| nt.trace.clone())
            .collect();

        vec![
            NamedTrace {
                name: "val_mixed_concat".to_string(),
                trace: Self::concat_traces(&picked),
            },
            NamedTrace {
                name: "val_mixed_interleaved".to_string(),
                trace: Self::interleave_traces_round_robin(&picked),
            },
        ]
    }

    fn make_test_bursty_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "test_bursty_a".to_string(),
                trace: Self::materialize_workload(BurstyWorkload::new(70, 30, 25, 28)),
            },
            NamedTrace {
                name: "test_bursty_b".to_string(),
                trace: Self::materialize_workload(BurstyWorkload::new(35, 120, 18, 48)),
            },
        ]
    }

    fn make_test_looping_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "test_looping_a".to_string(),
                trace: Self::materialize_workload(LoopingWorkload::new((0..40).collect(), 3000)),
            },
            NamedTrace {
                name: "test_looping_b".to_string(),
                trace: Self::materialize_workload(LoopingWorkload::new((0..192).collect(), 5200)),
            },
        ]
    }

    fn make_test_phase_traces() -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "test_phase_a".to_string(),
                trace: Self::materialize_workload(PhaseWorkload::new(10, 24, 400)),
            },
            NamedTrace {
                name: "test_phase_b".to_string(),
                trace: Self::materialize_workload(PhaseWorkload::new(12, 48, 600)),
            },
        ]
    }

    fn make_test_zipf_traces(base_seed: u64) -> Vec<NamedTrace> {
        vec![
            NamedTrace {
                name: "test_zipf_a".to_string(),
                trace: Self::materialize_workload(ZipfWorkload::new(
                    (0..180).collect(),
                    5000,
                    0.95,
                    base_seed + 201,
                )),
            },
            NamedTrace {
                name: "test_zipf_b".to_string(),
                trace: Self::materialize_workload(ZipfWorkload::new(
                    (0..320).collect(),
                    6000,
                    1.3,
                    base_seed + 202,
                )),
            },
        ]
    }

    fn materialize_workload<W: Workload>(mut workload: W) -> RequestTrace {
        let mut trace = RequestTrace::new();
        while let Some(key) = workload.next_request() {
            trace.push(Request::new(key));
        }
        trace
    }

    fn concat_traces(traces: &[RequestTrace]) -> RequestTrace {
        let mut out = RequestTrace::new();
        for trace in traces {
            for req in trace.requests() {
                out.push(*req);
            }
        }
        out
    }

    fn interleave_traces_round_robin(traces: &[RequestTrace]) -> RequestTrace {
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
        println!("   ✅ Final balanced dataset size: {}", balanced.len());
        balanced
    }
}

#[derive(Clone, Copy)]
enum RealTraceSplit {
    Validation,
    Test,
}

fn load_named_trace(arg: &str) -> Option<NamedTrace> {
    let path = std::path::PathBuf::from(arg);
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| arg.to_string());

    match wiki_trace_loader::load(&path) {
        Ok(wt) => {
            println!("📂 Loaded wiki trace '{}': {} requests", name, wt.trace.len());
            Some(NamedTrace {
                name,
                trace: wt.trace,
            })
        }
        Err(e) => {
            eprintln!("⚠️  Skipping '{}': {}", arg, e);
            None
        }
    }
}

fn main() {
    const MM_SIZE: usize = 200_000;

    let config = EvalBuildConfig {
        seed: 999,
        cache_sizes: vec![16, 32, 64],
        default_object_size: 128,
        max_samples_total: 60_000,
        max_samples_per_trace_run: 4_000,
        max_pairs_per_decision: 32,
        pairwise: PairwiseDatasetConfig {
            add_swapped_pairs: true,
            skip_ties: true,
            decay_factors: vec![0.5, 0.8, 0.95],
        },
    };

    let mut validation_real_traces = Vec::new();
    let mut test_real_traces = Vec::new();

    for arg in std::env::args().skip(1) {
        let (split, path) = if let Some(path) = arg.strip_prefix("val:") {
            (RealTraceSplit::Validation, path)
        } else if let Some(path) = arg.strip_prefix("test:") {
            (RealTraceSplit::Test, path)
        } else {
            (RealTraceSplit::Test, arg.as_str())
        };

        if let Some(named_trace) = load_named_trace(path) {
            match split {
                RealTraceSplit::Validation => validation_real_traces.push(named_trace),
                RealTraceSplit::Test => test_real_traces.push(named_trace),
            }
        }
    }

    if let Err(e) = EvalDatasetBuilder::write_validation_csv::<MM_SIZE, _>(
        &config,
        &validation_real_traces,
        "artifacts/datasets/pairwise_validation_dataset.csv",
    ) {
        eprintln!("Failed to write validation CSV: {}", e);
        std::process::exit(1);
    }

    if let Err(e) = EvalDatasetBuilder::write_test_csvs::<MM_SIZE, _>(
        &config,
        &test_real_traces,
        "artifacts/datasets/pairwise_test_datasets",
    ) {
        eprintln!("Failed to write test CSVs: {}", e);
        std::process::exit(1);
    }

    println!("🎉 Validation and test datasets complete.");
}

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
    pub fn build_pairwise_training_dataset<const MM_SIZE: usize>(
        config: &TrainingBuildConfig,
        real_traces: &[NamedTrace],
    ) -> Vec<PairwiseSample> {
        println!("🚀 Starting pairwise dataset generation...");

        let mut rng = StdRng::seed_from_u64(config.seed);

        let mut all_named_traces = Self::build_synthetic_trace_suite(config, &mut rng);
        println!("📊 Generated {} synthetic traces", all_named_traces.len());

        all_named_traces.extend(real_traces.iter().cloned());

        let mut all_samples = Vec::new();

        for named_trace in all_named_traces {
            println!(
                "\n🔹 Processing trace: {} (len = {})",
                named_trace.name,
                named_trace.trace.len()
            );

            for &cache_size in &config.cache_sizes {
                println!("   → Cache size: {}", cache_size);

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

                println!("      Generated {} raw samples", samples.len());

                let sampled = Self::downsample_trace_run_samples(
                    samples,
                    config.max_pairs_per_decision,
                    config.max_samples_per_trace_run,
                    &mut rng,
                );

                println!("      Kept {} samples after downsampling", sampled.len());

                all_samples.extend(sampled);

                println!("      Total dataset size so far: {}", all_samples.len());
            }
        }

        println!("\n🔀 Shuffling dataset...");
        all_samples.shuffle(&mut rng);

        if all_samples.len() > config.max_samples_total {
            println!(
                "✂️ Truncating dataset to {} samples",
                config.max_samples_total
            );
            all_samples.truncate(config.max_samples_total);
        }

        println!("\n✅ Final dataset size: {} samples", all_samples.len());

        all_samples
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

    fn build_synthetic_trace_suite(
        config: &TrainingBuildConfig,
        rng: &mut StdRng,
    ) -> Vec<NamedTrace> {
        let mut traces = Vec::new();

        if config.enable_bursty {
            traces.extend(Self::make_bursty_traces());
        }
        if config.enable_looping {
            traces.extend(Self::make_looping_traces());
        }
        if config.enable_phase {
            traces.extend(Self::make_phase_traces());
        }
        if config.enable_zipf {
            traces.extend(Self::make_zipf_traces(config.seed));
        }
        if config.enable_mixed {
            let mixed = Self::make_mixed_traces(&traces, rng);
            traces.extend(mixed);
        }

        traces
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
        "pairwise_training_dataset.csv",
    ) {
        Ok(()) => println!("\n🎉 Done! Dataset ready."),
        Err(e) => {
            eprintln!("❌ Failed: {}", e);
            std::process::exit(1);
        }
    }
}

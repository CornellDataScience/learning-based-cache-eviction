use std::env;
use std::process;

use lbce::core::cache::Cache;
use lbce::core::mainmemory::{MainMemory, MemoryObject};
use lbce::core::policy::{CacheKey, Policy};
use lbce::core::replay_engine::ReplayResult;
use lbce::core::trace::{CacheEvent, RequestTrace};
use lbce::policies::{
    fifo::FifoPolicy, learnedpolicy::LearnedPolicy, naivelru::LruPolicy, optimal::OptimalPolicy,
};
use lbce::workloads::bursty::BurstyWorkload;
use lbce::workloads::looping::LoopingWorkload;
use lbce::workloads::phase::PhaseWorkload;
use lbce::workloads::workload::collect_requests;
use lbce::workloads::zipf::ZipfWorkload;

const DEFAULT_MM_SIZE: usize = 65_536;
const DEFAULT_DECAY_FACTORS: [f32; 3] = [0.5, 0.8, 0.95];
const DEFAULT_POLICY: &str = "lru";
const DEFAULT_WORKLOAD: &str = "looping";
const DEFAULT_CACHE_CAPACITY: usize = 64;
const DEFAULT_TOTAL_REQUESTS: usize = 10_000;
const DEFAULT_KEY_SPACE: usize = 128;
const DEFAULT_MODEL_PATH: &str = "eviction_mlp.pt";
const DEFAULT_SHORTLIST_K: usize = 4;
const DEFAULT_ZIPF_SKEW: f64 = 1.2;
const DEFAULT_ZIPF_SEED: u64 = 7;
const DEFAULT_BURSTY_CYCLES: usize = 50;
const DEFAULT_BURSTY_QUIET: usize = 32;
const DEFAULT_BURSTY_BURST: usize = 16;
const DEFAULT_BACKGROUND_KEYS: usize = 32;
const DEFAULT_PHASES: usize = 4;
const DEFAULT_KEYS_PER_PHASE: usize = 32;

#[derive(Debug, Clone)]
struct RunnerConfig {
    policy: String,
    workload: String,
    cache_capacity: usize,
    total_requests: usize,
    key_space: usize,
    model_path: String,
    shortlist_k: usize,
    zipf_skew: f64,
    zipf_seed: u64,
    bursty_cycles: usize,
    bursty_quiet: usize,
    bursty_burst: usize,
    background_keys: usize,
    phase_count: usize,
    keys_per_phase: usize,
    verbose: bool,
    show_events: Option<usize>,
    debug_learned: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            policy: DEFAULT_POLICY.to_string(),
            workload: DEFAULT_WORKLOAD.to_string(),
            cache_capacity: DEFAULT_CACHE_CAPACITY,
            total_requests: DEFAULT_TOTAL_REQUESTS,
            key_space: DEFAULT_KEY_SPACE,
            model_path: DEFAULT_MODEL_PATH.to_string(),
            shortlist_k: DEFAULT_SHORTLIST_K,
            zipf_skew: DEFAULT_ZIPF_SKEW,
            zipf_seed: DEFAULT_ZIPF_SEED,
            bursty_cycles: DEFAULT_BURSTY_CYCLES,
            bursty_quiet: DEFAULT_BURSTY_QUIET,
            bursty_burst: DEFAULT_BURSTY_BURST,
            background_keys: DEFAULT_BACKGROUND_KEYS,
            phase_count: DEFAULT_PHASES,
            keys_per_phase: DEFAULT_KEYS_PER_PHASE,
            verbose: false,
            show_events: None,
            debug_learned: false,
        }
    }
}

impl RunnerConfig {
    fn from_env() -> Result<Self, String> {
        let mut config = Self::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--policy" => config.policy = next_string(&mut args, "--policy")?,
                "--workload" => config.workload = next_string(&mut args, "--workload")?,
                "--cache-capacity" => {
                    config.cache_capacity = next_parse(&mut args, "--cache-capacity")?
                }
                "--total-requests" => {
                    config.total_requests = next_parse(&mut args, "--total-requests")?
                }
                "--key-space" => config.key_space = next_parse(&mut args, "--key-space")?,
                "--model" => config.model_path = next_string(&mut args, "--model")?,
                "--shortlist-k" => config.shortlist_k = next_parse(&mut args, "--shortlist-k")?,
                "--zipf-skew" => config.zipf_skew = next_parse(&mut args, "--zipf-skew")?,
                "--zipf-seed" => config.zipf_seed = next_parse(&mut args, "--zipf-seed")?,
                "--bursty-cycles" => {
                    config.bursty_cycles = next_parse(&mut args, "--bursty-cycles")?
                }
                "--bursty-quiet" => config.bursty_quiet = next_parse(&mut args, "--bursty-quiet")?,
                "--bursty-burst" => config.bursty_burst = next_parse(&mut args, "--bursty-burst")?,
                "--background-keys" => {
                    config.background_keys = next_parse(&mut args, "--background-keys")?
                }
                "--phase-count" => config.phase_count = next_parse(&mut args, "--phase-count")?,
                "--keys-per-phase" => {
                    config.keys_per_phase = next_parse(&mut args, "--keys-per-phase")?
                }
                "--verbose" => config.verbose = true,
                "--show-events" => {
                    config.show_events = Some(next_parse(&mut args, "--show-events")?)
                }
                "--debug-learned" => config.debug_learned = true,
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                other => return Err(format!("unrecognized argument: {other}")),
            }
        }

        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.cache_capacity == 0 {
            return Err("cache capacity must be > 0".to_string());
        }
        if self.total_requests == 0 {
            return Err("total requests must be > 0".to_string());
        }
        if self.shortlist_k == 0 {
            return Err("shortlist-k must be > 0".to_string());
        }

        match self.policy.as_str() {
            "fifo" | "lru" | "learned" | "optimal" => {}
            other => {
                return Err(format!(
                    "unsupported policy '{other}'. expected one of: fifo, lru, learned, optimal"
                ));
            }
        }

        match self.workload.as_str() {
            "looping" | "zipf" | "bursty" | "phase" => {}
            other => {
                return Err(format!(
                    "unsupported workload '{other}'. expected one of: looping, zipf, bursty, phase"
                ));
            }
        }

        Ok(())
    }
}

fn next_string(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn next_parse<T>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = next_string(args, flag)?;
    raw.parse::<T>()
        .map_err(|err| format!("invalid value for {flag}: {err}"))
}

fn build_main_memory<const MM_SIZE: usize>(key_space: usize) -> MainMemory<MM_SIZE> {
    let mut mm = MainMemory::new();
    for key in 0..key_space {
        mm.insert(MemoryObject::new(key as CacheKey, vec![0]));
    }
    mm
}

fn run_with_policy<P: Policy>(config: &RunnerConfig, policy: P) -> ReplayResult {
    let trace = build_trace(config);
    let main_memory = build_main_memory::<DEFAULT_MM_SIZE>(trace_key_space(config));
    let mut cache = Cache::new(config.cache_capacity, policy, main_memory);
    replay_with_logging(config, &trace, &mut cache)
}

fn run_optimal_policy(config: &RunnerConfig) -> ReplayResult {
    let trace = build_trace(config);
    let main_memory = build_main_memory::<DEFAULT_MM_SIZE>(trace_key_space(config));
    let policy = OptimalPolicy::from_trace(&trace, config.cache_capacity);
    let mut cache = Cache::new(config.cache_capacity, policy, main_memory);
    replay_with_logging(config, &trace, &mut cache)
}

fn replay_with_logging<P: Policy, const MM_SIZE: usize>(
    config: &RunnerConfig,
    trace: &RequestTrace,
    cache: &mut Cache<P, MM_SIZE>,
) -> ReplayResult {
    let mut printed_events = 0usize;

    for (request_index, request) in trace.requests().iter().enumerate() {
        if config.verbose {
            println!(
                "[access] index={} key={} cache_size_before={}",
                request_index,
                request.key,
                cache.store.len()
            );
        }

        cache.access(request.key);

        if config.verbose {
            let events = cache.event_trace.events();
            for event in &events[printed_events..] {
                println!("{}", format_event(event));
            }
            printed_events = events.len();
        }
    }

    if let Some(limit) = config.show_events {
        let events = cache.event_trace.events();
        let start = events.len().saturating_sub(limit);
        println!("event_log_tail={}", events.len() - start);
        for event in &events[start..] {
            println!("{}", format_event(event));
        }
    }

    ReplayResult::from_metrics(&cache.metrics)
}

fn format_event(event: &CacheEvent) -> String {
    match event {
        CacheEvent::Hit { key, tick } => format!("[hit] tick={} key={}", tick, key),
        CacheEvent::Miss { key, tick } => format!("[miss] tick={} key={}", tick, key),
        CacheEvent::Insert {
            key,
            size_bytes,
            tick,
        } => format!(
            "[insert] tick={} key={} size_bytes={}",
            tick, key, size_bytes
        ),
        CacheEvent::Evict {
            key,
            size_bytes,
            tick,
        } => format!(
            "[evict] tick={} key={} size_bytes={}",
            tick, key, size_bytes
        ),
    }
}

fn trace_key_space(config: &RunnerConfig) -> usize {
    match config.workload.as_str() {
        "bursty" => config.background_keys + 1,
        "phase" => config.phase_count * config.keys_per_phase,
        _ => config.key_space,
    }
}

fn build_trace(config: &RunnerConfig) -> lbce::core::trace::RequestTrace {
    match config.workload.as_str() {
        "looping" => {
            let keys: Vec<CacheKey> = (0..config.key_space as CacheKey).collect();
            let mut workload = LoopingWorkload::new(keys, config.total_requests);
            collect_requests(&mut workload)
        }
        "zipf" => {
            let keys: Vec<CacheKey> = (0..config.key_space as CacheKey).collect();
            let mut workload = ZipfWorkload::new(
                keys,
                config.total_requests,
                config.zipf_skew,
                config.zipf_seed,
            );
            collect_requests(&mut workload)
        }
        "bursty" => {
            let mut workload = BurstyWorkload::new(
                config.bursty_cycles,
                config.bursty_quiet,
                config.bursty_burst,
                config.background_keys,
            );
            collect_requests(&mut workload)
        }
        "phase" => {
            let mut workload = PhaseWorkload::new(
                config.phase_count,
                config.keys_per_phase,
                requests_per_phase(config),
            );
            collect_requests(&mut workload)
        }
        other => panic!("unsupported workload after validation: {other}"),
    }
}

fn requests_per_phase(config: &RunnerConfig) -> usize {
    let base = config.total_requests / config.phase_count.max(1);
    base.max(1)
}

fn print_usage() {
    println!(
        "\
Usage:
  cargo run -- --policy <fifo|lru|learned|optimal> --workload <looping|zipf|bursty|phase> [options]

Core options:
  --policy <name>           Policy to run. Default: lru
  --workload <name>         Workload generator. Default: looping
  --cache-capacity <n>      Cache capacity. Default: 64
  --total-requests <n>      Number of requests to generate. Default: 10000
  --key-space <n>           Distinct keys for looping/zipf. Default: 128

Learned-policy options:
  --model <path>            Model checkpoint path. Default: eviction_mlp.pt
  --shortlist-k <n>         LRU shortlist size before pairwise voting. Default: 4
  --debug-learned           Print shortlist and pairwise vote details

Zipf options:
  --zipf-skew <f64>         Zipf skew parameter. Default: 1.2
  --zipf-seed <u64>         Zipf RNG seed. Default: 7

Bursty options:
  --bursty-cycles <n>       Number of cycles. Default: 50
  --bursty-quiet <n>        Quiet requests per cycle. Default: 32
  --bursty-burst <n>        Burst requests per cycle. Default: 16
  --background-keys <n>     Number of background keys. Default: 32

Phase options:
  --phase-count <n>         Number of phases. Default: 4
  --keys-per-phase <n>      Distinct keys in each phase. Default: 32

Logging options:
  --verbose                 Print per-access and per-event logs during replay
  --show-events <n>         Print the last n cache events after replay
"
    );
}

fn main() {
    let config = match RunnerConfig::from_env() {
        Ok(config) => config,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            process::exit(2);
        }
    };

    let result = match config.policy.as_str() {
        "fifo" => run_with_policy(&config, FifoPolicy::new(config.cache_capacity)),
        "lru" => run_with_policy(&config, LruPolicy::new(config.cache_capacity)),
        "learned" => run_with_policy(
            &config,
            LearnedPolicy::with_config_and_debug(
                &config.model_path,
                DEFAULT_DECAY_FACTORS.to_vec(),
                config.shortlist_k,
                config.debug_learned,
            ),
        ),
        "optimal" => run_optimal_policy(&config),
        other => {
            eprintln!("unsupported policy after validation: {other}");
            process::exit(2);
        }
    };

    println!("policy={}", config.policy);
    println!("workload={}", config.workload);
    if config.policy == "learned" {
        println!("model={}", config.model_path);
        println!("shortlist_k={}", config.shortlist_k);
        println!("debug_learned={}", config.debug_learned);
    }
    println!("verbose={}", config.verbose);
    println!("show_events={:?}", config.show_events);
    println!("{result}");
}

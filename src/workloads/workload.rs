// at some point, we might want to access one of these
use crate::core::metrics::Metrics;
use crate::core::time::Clock;
use crate::core::trace::CacheTrace;

/* 
looping workload: simulate repeated access to a small set of items to test cache performance under high locality.
phase workload: the program proceeds in distinct phases, where each phase accesses a set of items more often. 
bursty workload: periods of intense access to certain items followed by periods of low activity. 
zipf workload: frequencies are inversely proportional to ranks
*/

// IMO: we have too many files and could just condense all the workloads into one file but...
pub trait Workload {
    fn next_request(&mut self) -> Option<u64>; // returns the next key to access, or None if workload is complete
    fn is_complete(&self) -> bool; // indicates whether the workload has finished generating requests
}
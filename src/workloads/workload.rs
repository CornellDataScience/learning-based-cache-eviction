use crate::core::policy::CacheKey;

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

// Collects all requests from a workload into a Vec<CacheKey>.
pub fn collect_requests(workload: &mut impl Workload) -> Vec<CacheKey> {
    let mut requests = Vec::new();
    while let Some(key) = workload.next_request() {
        requests.push(key);
    }
    requests
}
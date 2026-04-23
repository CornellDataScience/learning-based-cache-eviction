use crate::core::policy::CacheKey;
use crate::core::trace::{Request, RequestTrace};

pub trait Workload {
    fn next_request(&mut self) -> Option<CacheKey>;
    fn is_complete(&self) -> bool;
}

// Collect all requests from a workload into a RequestTrace.
pub fn collect_requests(workload: &mut impl Workload) -> RequestTrace {
    let mut trace = RequestTrace::new();

    while let Some(key) = workload.next_request() {
        trace.push(Request::new(key));
    }

    trace
}

use crate::core::policy::CacheKey;
use std::collections::HashMap;
use crate::core::trace::RequestTrace;

pub struct DataEntry {
    time_between_accesses: f64,
    exp_decay_counters: Vec<f64>,
    access_cnt: f64,
    recent_rank: f64,
    first_access_time: f64,
    last_access_time: f64,
    avg_time_between_accesses: f64,
}
pub struct CacheState {
    entries: HashMap<CacheKey, DataEntry>,
}

impl CacheState {
    pub fn new() -> Self {
        CacheState {
            entries: HashMap::new(),
        }
    }
}

pub struct Generator {
}

impl Generator {
    pub fn calc_next_use(trace: RequestTrace){
        let len = trace.len();
        let mut next_use = vec![usize::MAX; len];
        let mut next_pos = HashMap::new();

        for i in (0..len).rev(){
            let key = trace.requests[i].key;
            if let Some(&next_index) = next_pos.get(&key) {
                next_use[i] = next_index;
            }

            next_pos.insert(key, i);
        }
        next_use
    }
    pub fn make_dataset(rt: RequestTrace) -> Vec<CacheState> {
        let mut dataset = Vec::new();
        let mut state = CacheState::new();

        let mut next_use = Self::calc_next_use(rt.clone()); //backward pass for next_use when calculating the forward pass

        for request in rt.requests {
            // if it's time to evict, then do the following
            // time between accesses shift left + add new entry
            // update exp_decay_counters
            // avg_time_between_accesses = (avg * access_cnt + t) / access_cnt + 1
            // add 1 to access_cnt
            // recent_rank - update location of key in concurrent LRU queue before update
            // if first time accessing key, create first_access_time
            // write last_access_time
            let key = request.key.clone();
        
            if let Some(entry) = state.entries.get_mut(&key) {
                // Calculate time between accesses
                entry.time_between_accesses = current_time - entry.last_access_time;
                
                // Update exponential decay counters (shift left and add new value)
                entry.exp_decay_counters.remove(0);
                entry.exp_decay_counters.push(entry.time_between_accesses);
                
                // Update average time between accesses
                entry.avg_time_between_accesses = 
                    (entry.avg_time_between_accesses * entry.access_cnt + entry.time_between_accesses) 
                    / (entry.access_cnt + 1.0);
                
                // Increment access count
                entry.access_cnt += 1.0;
                
                // Update recent rank (lower is more recent)
                entry.recent_rank += 1.0;
            } else {
                // First time accessing this key
                state.entries.insert(
                    key,
                    DataEntry {
                        time_between_accesses: 0.0,
                        exp_decay_counters: Vec::new(),
                        access_cnt: 1.0,
                        recent_rank: 0.0,
                        first_access_time: current_time,
                        last_access_time: current_time,
                        avg_time_between_accesses: 0.0,
                    },
                );
            }
            
            // Update last access time for existing entry
            if let Some(entry) = state.entries.get_mut(&key) {
                entry.last_access_time = current_time;
                entry.recent_rank = 0.0; // Reset rank since it was just accessed
            }
            
            current_time += 1.0;
            // add new entry to dataset
        }    

        return dataset
    }
    dataset.push(state);
    dataset
}




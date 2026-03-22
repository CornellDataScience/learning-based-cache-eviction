use crate::core::entry::Entry;

// RequestTrace gives us the list of cache keys that were accessed, but i think we need a new struct EntryRqTrace
// so that we can get the full details of the entry that encompasses the key since we need those metrics for
// our dataset
use crate::core::trace::EntryRqTrace; 

struct Record {
    entry0: Entry,
    entry1: Entry,
    evict: u8 // 0 if entry0 should be evicted, 1 if entry1 should be evicted
}

impl Record {
    pub fn new(entry0: Entry, entry1: Entry, evict: u8) -> Self {
        Self {
            entry0,
            entry1,
            evict,
        }
    }
}

struct PairwiseDataset {
    entries: Vec<Entry>,
    time_to_next_access: Vec<u32>,
}

impl PairwiseDataset {
    pub fn new(trace: &EntryRqTrace) -> Self {
        let mut entries = Vec::new();
        let mut time_to_next_access = Vec::new();
        for (i, entry) in trace.entryrqs().iter().enumerate() {
            entries.push(entry.clone()); // do this because entry will mutate as trace is being replayed
            let next_access_time = trace.entryrqs().iter().skip(i + 1).find(|e| e.key == entry.key)
                                    .map(|ttn| ttn.tick - entry.tick)
                                    .unwrap_or(u32::MAX); // use max if the entry isn't accessed again in the trace
            time_to_next_access.push(next_access_time);
        }
        Self {
            entries,
            time_to_next_access,
        }
    }

    pub fn generate(&self) -> Vec<Record> {
        let mut records = Vec::new();
        for (i, entry) in self.entries.iter().enumerate() {
            for (j, next_entry) in self.entries[i + 1..].iter().enumerate() {
                if entry.key != next_entry.key { // make sure we're not comparing same cache keys
                    let evict = if (self.time_to_next_access[i] < self.time_to_next_access[i + 1 + j]) { 1 } else { 0 };
                    records.push(Record {
                        entry0: entry.clone(),
                        entry1: next_entry.clone(),
                        evict,
                    });
                } else {
                    break; // break if we encounter the same key bc we want to compare to the most recent version of the entry
                }
            }
        }
        records
    }
}
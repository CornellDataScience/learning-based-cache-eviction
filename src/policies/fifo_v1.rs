use std::collections::{HashSet, VecDeque};
use crate::core::policy::{Policy, CacheKey};

struct FifoEntry {
    key: CacheKey,
    freq: u8, // 2-bit CLOCK counter, capped at 3
}

pub struct Fifo {
    // Probationary FIFO (Quick Demotion) — 10% of capacity
    probation: VecDeque<FifoEntry>,
    probation_members: HashSet<CacheKey>,
    probation_capacity: usize,

    // Main cache (Lazy Promotion via 2-bit CLOCK) — 90% of capacity
    main: VecDeque<FifoEntry>,
    main_members: HashSet<CacheKey>,
    main_capacity: usize,

    // Ghost queue — remembers recently quick-demoted keys
    ghost: VecDeque<CacheKey>,
    ghost_members: HashSet<CacheKey>,
    ghost_capacity: usize,
}

impl Fifo {
    pub fn new(total_capacity: usize) -> Self {
        let probation_capacity = (total_capacity / 10).max(1);
        let main_capacity = total_capacity - probation_capacity;
        Fifo {
            probation: VecDeque::new(),
            probation_members: HashSet::new(),
            probation_capacity,
            main: VecDeque::new(),
            main_members: HashSet::new(),
            main_capacity,
            ghost: VecDeque::new(),
            ghost_members: HashSet::new(),
            ghost_capacity: main_capacity,
        }
    }

    fn increment_freq(&mut self, key: CacheKey) {
        for entry in self.probation.iter_mut() {
            if entry.key == key {
                if entry.freq < 3 { entry.freq += 1; }
                return;
            }
        }
        for entry in self.main.iter_mut() {
            if entry.key == key {
                if entry.freq < 3 { entry.freq += 1; }
                return;
            }
        }
    }

    // Evicts the front of probation.
    // If it has been accessed (freq > 0), graduates it to main — returns None.
    // If unpopular (freq == 0), quick-demotes it to ghost — returns the evicted key.
    fn evict_from_probation(&mut self) -> Option<CacheKey> {
        let entry = self.probation.pop_front()?;
        self.probation_members.remove(&entry.key);
        if entry.freq > 0 {
            self.insert_into_main(entry.key);
            None
        } else {
            let evicted = entry.key;
            self.add_to_ghost(evicted);
            Some(evicted)
        }
    }

    // 2-bit CLOCK eviction from main: scan front-to-back, decrementing freq.
    // First entry with freq == 0 is evicted (lazy promotion).
    pub(crate) fn evict_from_main(&mut self) -> Option<CacheKey> {
        let len = self.main.len();
        for _ in 0..len {
            let mut entry = self.main.pop_front()?;
            if entry.freq == 0 {
                self.main_members.remove(&entry.key);
                return Some(entry.key);
            }
            entry.freq -= 1;
            self.main.push_back(entry);
        }
        // Fallback: everything had freq > 0, evict front anyway
        let entry = self.main.pop_front()?;
        self.main_members.remove(&entry.key);
        Some(entry.key)
    }

    pub(crate) fn insert_into_main(&mut self, key: CacheKey) {
        if self.main.len() >= self.main_capacity {
            self.evict_from_main();
        }
        self.main_members.insert(key);
        self.main.push_back(FifoEntry { key, freq: 0 });
    }

    fn add_to_ghost(&mut self, key: CacheKey) {
        if self.ghost.len() >= self.ghost_capacity {
            if let Some(old) = self.ghost.pop_front() {
                self.ghost_members.remove(&old);
            }
        }
        self.ghost_members.insert(key);
        self.ghost.push_back(key);
    }
}

impl Policy for Fifo {
    fn on_hit(&mut self, key: CacheKey) {
        // Lazy promotion: just mark accessed, no pointer shuffling
        self.increment_freq(key);
    }

    fn on_miss(&mut self, key: CacheKey) -> Option<CacheKey> {
        // Decide where the new key goes, and whether anything needs evicting

        if self.ghost_members.contains(&key) {
            // Key was quick-demoted but came back — proven popular, skip probation
            self.ghost_members.remove(&key);
            self.ghost.retain(|k| *k != key);

            if self.main.len() >= self.main_capacity {
                let victim = self.evict_from_main();
                self.insert_into_main(key);
                return victim;
            }
            self.insert_into_main(key);
            return None;
        }

        // Brand new key — send to probation
        if self.probation.len() >= self.probation_capacity {
            // Probation full: try to quick-demote the oldest probation entry
            let evicted = self.evict_from_probation();
            self.probation_members.insert(key);
            self.probation.push_back(FifoEntry { key, freq: 0 });
            return evicted; // Some(key) if unpopular was evicted, None if graduated
        }

        self.probation_members.insert(key);
        self.probation.push_back(FifoEntry { key, freq: 0 });
        None
    }

    fn insert(&mut self, key: CacheKey) {
        // Called after the cache has confirmed insertion.
        // on_miss already placed the key into the right queue,
        // so this is a no-op — but you could add assertions here
        // to catch bugs (e.g. key should now be in probation or main).
        debug_assert!(
            self.probation_members.contains(&key) || self.main_members.contains(&key),
            "insert() called for key not tracked by policy"
        );
    }

    fn remove(&mut self, key: CacheKey) {
        if self.probation_members.remove(&key) {
            self.probation.retain(|e| e.key != key);
        } else if self.main_members.remove(&key) {
            self.main.retain(|e| e.key != key);
        }
        // Also clean ghost if present (e.g. explicit cache invalidation)
        if self.ghost_members.remove(&key) {
            self.ghost.retain(|k| *k != key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::policy::Policy;

    // helper: Fifo with probation=2, main=18
    fn make_fifo() -> Fifo {
        Fifo::new(20)
    }

    // --- Insertion routing ---

    #[test]
    fn new_object_goes_to_probation() {
        let mut f = make_fifo();
        f.on_miss(1);
        assert!(f.probation_members.contains(&1));
        assert!(!f.main_members.contains(&1));
    }

    #[test]
    fn ghost_hit_goes_directly_to_main() {
        let mut f = make_fifo();
        // fill + overflow probation so key 1 gets quick-demoted to ghost
        f.on_miss(1); // probation: [1]
        f.on_miss(2); // probation: [1, 2]
        f.on_miss(3); // probation full: evict 1 (freq=0) → ghost
        assert!(f.ghost_members.contains(&1));
        // now re-request key 1
        f.on_miss(1);
        assert!(f.main_members.contains(&1));
        assert!(!f.probation_members.contains(&1));
        assert!(!f.ghost_members.contains(&1));
    }

    // --- Quick Demotion ---

    #[test]
    fn unpopular_object_quick_demoted_to_ghost() {
        let mut f = make_fifo();
        f.on_miss(1);
        f.on_miss(2);
        let evicted = f.on_miss(3); // triggers eviction of key 1 (freq=0)
        assert_eq!(evicted, Some(1));
        assert!(f.ghost_members.contains(&1));
        assert!(!f.probation_members.contains(&1));
        assert!(!f.main_members.contains(&1));
    }

    #[test]
    fn popular_object_graduates_to_main() {
        let mut f = make_fifo();
        f.on_miss(1);
        f.on_hit(1);              // freq → 1, mark as popular
        f.on_miss(2);
        let evicted = f.on_miss(3); // evict key 1 from probation: freq>0 → graduate
        assert_eq!(evicted, None); // no eviction returned, key graduated instead
        assert!(f.main_members.contains(&1));
        assert!(!f.ghost_members.contains(&1));
    }

    // --- Lazy Promotion (2-bit CLOCK in main) ---

    #[test]
    fn clock_skips_recently_accessed_entries() {
        let mut f = make_fifo();
        // put two keys directly in main
        f.insert_into_main(10);
        f.insert_into_main(20);
        f.on_hit(10); // freq[10] → 1, freq[20] stays 0
        // evict from main: should skip 10 (freq=1), evict 20... 
        // wait — 10 is at front. CLOCK decrements it to 0 and requeues,
        // then evicts 20 (freq=0)
        let victim = f.evict_from_main();
        assert_eq!(victim, Some(20));
        assert!(f.main_members.contains(&10));
    }

    #[test]
    fn freq_capped_at_3() {
        let mut f = make_fifo();
        f.on_miss(1);
        for _ in 0..10 {
            f.on_hit(1);
        }
        let entry = f.probation.front().unwrap();
        assert_eq!(entry.freq, 3);
    }

    // --- Remove ---

    #[test]
    fn remove_cleans_up_probation() {
        let mut f = make_fifo();
        f.on_miss(1);
        f.remove(1);
        assert!(!f.probation_members.contains(&1));
        assert!(f.probation.is_empty());
    }

    #[test]
    fn remove_cleans_up_main() {
        let mut f = make_fifo();
        f.insert_into_main(99);
        f.remove(99);
        assert!(!f.main_members.contains(&99));
    }

    // --- Capacity boundaries ---

    #[test]
    fn no_eviction_when_cache_not_full() {
        let mut f = make_fifo(); // probation capacity = 2
        assert_eq!(f.on_miss(1), None);
        assert_eq!(f.on_miss(2), None); // probation now full but not over
    }

    #[test]
    fn ghost_does_not_grow_unbounded() {
        let mut f = Fifo::new(10); // ghost_capacity = 9
        // flood with unique keys to overflow ghost
        for i in 0..100 {
            f.on_miss(i);
        }
        assert!(f.ghost.len() <= f.ghost_capacity);
    }
}
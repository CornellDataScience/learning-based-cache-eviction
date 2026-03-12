use lbce::core::cache::Cache;
use lbce::core::mainmemory::{MainMemory, MemoryObject};
use lbce::policies::fifo::FifoPolicy;
use lbce::policies::naivelru::LruPolicy;

fn populated_main_memory<const N: usize>(keys: &[u64]) -> MainMemory<N> {
    let mut mm = MainMemory::new();
    for &key in keys {
        mm.insert(MemoryObject::new(key, 1));
    }
    mm
}

fn fifo_cache(capacity: usize) -> Cache<FifoPolicy, 1024> {
    let mm = populated_main_memory::<1024>(&[1, 2, 3, 4, 5, 6, 10]);
    Cache::new(capacity, FifoPolicy::new(capacity), mm)
}

fn lru_cache(capacity: usize) -> Cache<LruPolicy, 1024> {
    let mm = populated_main_memory::<1024>(&[1, 2, 3, 4, 5, 6, 10]);
    Cache::new(capacity, LruPolicy::new(capacity), mm)
}

// --- Backing store / insertion semantics ---

#[test]
fn miss_inserts_only_if_key_exists_in_main_memory() {
    let mut mm = MainMemory::<1024>::new();
    mm.insert(MemoryObject::new(1, 1));

    let mut cache = Cache::new(4, FifoPolicy::new(4), mm);

    cache.access(1); // present in backing store
    assert!(cache.store.contains_key(&1));

    cache.access(999); // not present in backing store
    assert!(!cache.store.contains_key(&999));
}

#[test]
fn zero_capacity_cache_never_stores_anything() {
    let mm = populated_main_memory::<1024>(&[1, 2, 3]);
    let mut cache = Cache::new(0, FifoPolicy::new(1), mm);

    cache.access(1);
    cache.access(2);

    assert_eq!(cache.store.len(), 0);
    assert_eq!(cache.metrics.request_count, 2);
    assert_eq!(cache.metrics.hit_count, 0);
}

// --- Clock / time ---

#[test]
fn clock_advances_each_access() {
    let mut cache = fifo_cache(4);
    assert_eq!(cache.clock.get_tick(), 0);
    
    cache.access(1);
    assert_eq!(cache.clock.get_tick(), 1);
    
    cache.access(2);
    assert_eq!(cache.clock.get_tick(), 2);
    
    cache.access(1); // hit
    assert_eq!(cache.clock.get_tick(), 3);
}

// --- Entry ---

#[test]
fn entry_insertion_tick_is_set_on_miss() {
    let mut cache = fifo_cache(4);
    cache.access(10); // tick becomes 1, entry inserted at tick 1
    let entry = cache.store.get(&10).unwrap();
    
    assert_eq!(entry.insertion_tick, 1);
    assert_eq!(entry.last_access_tick, 1);
    assert_eq!(entry.access_count, 1);
}

#[test]
fn entry_updates_on_hit() {
    let mut cache = fifo_cache(4);
    
    cache.access(10); // tick=1, inserted
    cache.access(10); // tick=2, hit
    cache.access(10); // tick=3, hit
    
    let entry = cache.store.get(&10).unwrap();
    assert_eq!(entry.insertion_tick, 1);
    assert_eq!(entry.last_access_tick, 3);
    assert_eq!(entry.access_count, 3);
}

#[test]
fn entry_frequency_decreases_over_time() {
    let mut cache = fifo_cache(4);
    
    cache.access(5); // tick=1, access_count=1, freq = 1/(1-1+1) = 1.0
    cache.access(5); // tick=2, access_count=2, freq = 2/(2-1+1) = 1.0
    cache.access(6); // tick=3, miss — key 5 not accessed this tick
    
    // key 5: access_count=2, insertion_tick=1, freq at tick=3 => 2/(3-1+1) = 2/3
    let entry = cache.store.get(&5).unwrap();
    let freq = entry.frequency(cache.clock.get_tick());
    assert!((freq - 2.0 / 3.0).abs() < 1e-9);
}

// --- Metrics ---

#[test]
fn metrics_all_misses() {
    let mut cache = fifo_cache(4);
    
    cache.access(1);
    cache.access(2);
    cache.access(3);
    
    assert_eq!(cache.metrics.hit_count, 0);
    assert_eq!(cache.metrics.request_count, 3);
    assert_eq!(cache.metrics.hit_rate(), 0.0);
    assert_eq!(cache.metrics.miss_rate(), 1.0);
}

#[test]
fn metrics_mixed_hits_and_misses() {
    let mut cache = fifo_cache(4);
    
    cache.access(1); // miss
    cache.access(2); // miss
    cache.access(1); // hit
    cache.access(2); // hit
    
    assert_eq!(cache.metrics.request_count, 4);
    assert_eq!(cache.metrics.hit_count, 2);
    assert_eq!(cache.metrics.hit_rate(), 0.5);
    assert_eq!(cache.metrics.miss_rate(), 0.5);
}

#[test]
fn metrics_eviction_count() {
    let mut cache = fifo_cache(2);
    
    cache.access(1);
    cache.access(2);
    cache.access(3); // eviction
    cache.access(4); // eviction
   
    assert_eq!(cache.metrics.eviction_count, 2);
}

#[test]
fn metrics_empty_cache_hit_rate_is_zero() {
    let cache = fifo_cache(4);
    assert_eq!(cache.metrics.hit_rate(), 0.0);
}

// --- Capacity invariants ---

#[test]
fn store_len_never_exceeds_capacity_fifo() {
    let mut cache = fifo_cache(2);

    cache.access(1);
    cache.access(2);
    cache.access(3);
    cache.access(4);
    cache.access(5);

    assert!(cache.store.len() <= 2);
}

#[test]
fn store_len_never_exceeds_capacity_lru() {
    let mut cache = lru_cache(2);

    cache.access(1);
    cache.access(2);
    cache.access(3);
    cache.access(4);
    cache.access(5);

    assert!(cache.store.len() <= 2);
}

// --- FIFO policy behavior ---

#[test]
fn fifo_hit_on_second_access() {
    let mut cache = fifo_cache(4);
    
    cache.access(1); // miss
    cache.access(1); // hit
    
    assert_eq!(cache.metrics.hit_count, 1);
}

#[test]
fn fifo_no_eviction_below_capacity() {
    let mut cache = fifo_cache(3);
    
    cache.access(1);
    cache.access(2);
    cache.access(3);
    
    assert_eq!(cache.metrics.eviction_count, 0);
    assert_eq!(cache.store.len(), 3);
}

#[test]
fn fifo_evicts_oldest_inserted() {
    let mut cache = fifo_cache(2);

    cache.access(1); // oldest
    cache.access(2);
    cache.access(3); // at capacity: evict 1
    
    assert!(!cache.store.contains_key(&1));
    assert!(cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
}

#[test]
fn fifo_hit_does_not_protect_key_from_eviction() {
    // FIFO never reorders; hitting key 1 doesn't save it
    let mut cache = fifo_cache(2);
    
    cache.access(1); // inserted first
    cache.access(2);
    cache.access(1); // hit — FIFO doesn't move 1
    cache.access(3); // still evicts 1 (oldest inserted)
    
    assert!(!cache.store.contains_key(&1));
    assert!(cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
}

#[test]
fn fifo_eviction_sequence_is_insertion_order() {
    let mut cache = fifo_cache(2);
    
    cache.access(1);
    cache.access(2);
    cache.access(3); // evict 1
    cache.access(4); // evict 2
    
    assert!(!cache.store.contains_key(&1));
    assert!(!cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
    assert!(cache.store.contains_key(&4));
    assert_eq!(cache.metrics.eviction_count, 2);
}

// --- LRU policy behavior ---

#[test]
fn lru_hit_on_second_access() {
    let mut cache = lru_cache(4);

    cache.access(1); // miss
    cache.access(1); // hit

    assert_eq!(cache.metrics.hit_count, 1);
}

#[test]
fn lru_no_eviction_below_capacity() {
    let mut cache = lru_cache(3);

    cache.access(1);
    cache.access(2);
    cache.access(3);

    assert_eq!(cache.metrics.eviction_count, 0);
    assert_eq!(cache.store.len(), 3);
}

#[test]
fn lru_without_hits_evicts_oldest() {
    // No hits means insertion order == recency order, same as FIFO
    let mut cache = lru_cache(2);
    
    cache.access(1);
    cache.access(2);
    cache.access(3); // evict 1 (least recently used)
    
    assert!(!cache.store.contains_key(&1));
    assert!(cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
}

#[test]
fn lru_hit_protects_key_from_eviction() {
    let mut cache = lru_cache(2);
    
    cache.access(1); // insert 1 (LRU)
    cache.access(2); // insert 2 (MRU)
    cache.access(1); // hit — 1 becomes MRU, 2 becomes LRU
    cache.access(3); // evict 2 (now LRU)
    
    assert!(cache.store.contains_key(&1));
    assert!(!cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&3));
}

#[test]
fn lru_eviction_sequence_tracks_recency() {
    let mut cache = lru_cache(3);
    cache.access(1);
    cache.access(2);
    cache.access(3);
    // order: [1(LRU), 2, 3(MRU)]
    cache.access(2); // hit — order: [1(LRU), 3, 2(MRU)]
    cache.access(4); // evict 1
    assert!(!cache.store.contains_key(&1));

    cache.access(5); // evict 3
    assert!(!cache.store.contains_key(&3));

    assert!(cache.store.contains_key(&2));
    assert!(cache.store.contains_key(&4));
    assert!(cache.store.contains_key(&5));
}

// --- FIFO vs LRU divergence ---

#[test]
fn fifo_and_lru_diverge_on_canonical_sequence() {
    let mut fifo = fifo_cache(2);
    let mut lru = lru_cache(2);

    for key in [1, 2, 1, 3] {
        fifo.access(key);
        lru.access(key);
    }

    // FIFO: evicts 1 (oldest inserted)
    assert!(!fifo.store.contains_key(&1));
    assert!(fifo.store.contains_key(&2));
    assert!(fifo.store.contains_key(&3));

    // LRU: evicts 2 (least recently used)
    assert!(lru.store.contains_key(&1));
    assert!(!lru.store.contains_key(&2));
    assert!(lru.store.contains_key(&3));
}

//represents key stored in cache
pub type CacheKey = u64;

pub trait Policy {
    // called when a key is accessed and found in cache
    // policy updates internal ordering
    fn on_hit(&mut self, key : CacheKey);

    // called on a miss, policy inspects current candidates and returns 
    // which one should be evicted(if any)
    fn on_miss(&mut self, key: CacheKey) -> Option<CacheKey>;

    // adds key to policy to track after cache insertion
    fn insert(&mut self, key: CacheKey);

    // removes key from policy track
    // usage: after something removed from cache, update policy
    // to refresh tracking
    fn remove(&mut self, key: CacheKey);
}
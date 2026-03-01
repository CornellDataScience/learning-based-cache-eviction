pub type CacheKey = u64;

pub trait Policy {
    fn on_hit(&mut self, key: CacheKey);
    fn on_miss(&mut self, key: CacheKey) -> Option<CacheKey>;
    fn insert(&mut self, key: CacheKey);
    fn remove(&mut self, key: CacheKey);
}
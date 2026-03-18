use std::fmt;

#[derive(Clone)]
pub struct Entry<T> {
    pub key: T,
    pub size_in_bytes: usize,
    pub insertion_tick: u64,
    pub last_access_tick: u64,
    pub access_count: u64,
    pub bytes: Vec<u8>,
}

impl<T, const N : usize> Entry<T, N> {
    pub fn new(key: T, size_in_bytes: usize, insertion_tick: u64, bytes: Vec<u8>) -> Self {
        Self {
            key,
            size_in_bytes,
            insertion_tick,
            last_access_tick: insertion_tick,
            access_count: 1,
            bytes,
        }
    }

    pub fn on_access(&mut self, current_tick: u64) {
        self.last_access_tick = current_tick;
        self.access_count = self.access_count.saturating_add(1);
        
    }

    pub fn frequency(&self, current_tick: u64) -> f64 {
        let age = current_tick.saturating_sub(self.insertion_tick) + 1;
        self.access_count as f64 / age as f64
    }
}

impl<T: fmt::Display, const N : usize> fmt::Display for Entry<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Entry:\nkey: {}\nsize_in_bytes: {}\ninsertion_tick: {}\nlast_access_tick: {}\naccess_count: {}",
            self.key,
            self.size_in_bytes,
            self.insertion_tick,
            self.last_access_tick,
            self.access_count,
        )
    }
}
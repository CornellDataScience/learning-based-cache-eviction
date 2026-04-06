use crate::core::mainmemory::{MainMemory, MemoryObject};
use crate::core::policy::CacheKey;
use crate::core::trace::RequestTrace;

pub struct MainMemoryBuilder;

impl MainMemoryBuilder {
    /// Build main memory from a trace using a fixed object size.
    pub fn from_trace<const MM_SIZE: usize>(
        trace: &RequestTrace,
        default_object_size: usize,
    ) -> MainMemory<MM_SIZE> {
        let mut mm = MainMemory::<MM_SIZE>::new();

        for req in trace.requests() {
            if !mm.contains(&req.key) {
                mm.insert(MemoryObject::new(
                    req.key,
                    vec![0u8; default_object_size],
                ));
            }
        }

        mm
    }

    /// Build main memory from a trace with object bytes decided by a callback.
    pub fn from_trace_with<const MM_SIZE: usize, F>(
        trace: &RequestTrace,
        mut bytes_fn: F,
    ) -> MainMemory<MM_SIZE>
    where
        F: FnMut(CacheKey) -> Vec<u8>,
    {
        let mut mm = MainMemory::<MM_SIZE>::new();

        for req in trace.requests() {
            if !mm.contains(&req.key) {
                let bytes = bytes_fn(req.key);
                mm.insert(MemoryObject::new(req.key, bytes));
            }
        }

        mm
    }
}
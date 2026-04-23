use std::fmt;

pub struct Clock {
    tick: u64,
}

impl Clock {
    pub fn new() -> Self {
        Self { tick: 0 }
    }

    pub fn tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    pub fn get_tick(&self) -> u64 {
        self.tick
    }

    pub fn reset(&mut self) {
        self.tick = 0;
    }
}

impl fmt::Display for Clock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Current tick: {}", self.tick)
    }
}

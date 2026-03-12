use std::fmt;

pub struct Clock {
    pub tick: u64
}

impl Clock {
    pub fn new() -> Self {
        Clock {
            tick: 0
        }
    }

    pub fn tick_up(&mut self) {
        self.tick += 1;
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
        write!(
            f,
            "Current tick: {}",
            self.tick
        )
    }
}
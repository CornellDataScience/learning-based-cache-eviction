use lbce::core::time::Clock;

#[test]
fn test_clock_init() {
    let clock = Clock::new();
    assert_eq!(clock.get_tick(), 0);
}

#[test]
fn test_clock_tick_up() {
    let mut clock = Clock::new();
    for _ in 1..=10 {
        clock.tick_up();
    }
    assert_eq!(clock.get_tick(), 10);
}

#[test]
fn test_reset_after_ticks() {
    let mut clock = Clock::new();
    for _ in 1..=5 {
        clock.tick_up();
    }
    clock.reset();
    assert_eq!(clock.get_tick(), 0);
    for _ in 1..=5 {
        clock.tick_up();
    }
    assert_eq!(clock.get_tick(), 5);
}
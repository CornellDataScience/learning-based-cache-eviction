use lbce::core::entry::Entry;

#[test]
fn test_entry_init() {
    let entry = Entry::new("init key", 128, 5);
    assert_eq!(entry.key, "init key");
    assert_eq!(entry.size_in_bytes, 128);
    assert_eq!(entry.insertion_tick, 5);
    assert_eq!(entry.last_access_tick, 5);
    assert_eq!(entry.access_count, 1);
}

#[test]
fn test_on_access() {
    let mut entry = Entry::new("access key", 64, 3);
    assert_eq!(entry.key, "access key");
    assert_eq!(entry.size_in_bytes, 64);
    assert_eq!(entry.insertion_tick, 3);
    assert_eq!(entry.last_access_tick, 3);
    assert_eq!(entry.access_count, 1);
    entry.on_access(10);
    assert_eq!(entry.last_access_tick, 10);
    assert_eq!(entry.access_count, 2);
    entry.on_access(17);
    assert_eq!(entry.last_access_tick, 17);
    assert_eq!(entry.access_count, 3);
    entry.on_access(172);
    assert_eq!(entry.last_access_tick, 172);
    assert_eq!(entry.access_count, 4);
}

#[test]
fn test_frequency() {
    let mut entry = Entry::new("freq key", 32, 9);
    assert_eq!(entry.key, "freq key");
    assert_eq!(entry.size_in_bytes, 32);
    assert_eq!(entry.insertion_tick, 9);
    assert_eq!(entry.last_access_tick, 9);
    assert_eq!(entry.access_count, 1);
    entry.on_access(10);
    assert_eq!(entry.last_access_tick, 10);
    assert_eq!(entry.access_count, 2);
    assert_eq!(entry.frequency(10), 1.0);
    assert_eq!(entry.frequency(15), 2.0 / 7.0);
    entry.on_access(24);
    assert_eq!(entry.last_access_tick, 24);
    assert_eq!(entry.access_count, 3);
    assert_eq!(entry.frequency(24), 3.0 / (24.0 - 9.0 + 1.0));
    assert_eq!(entry.frequency(38), 3.0 / (38.0 - 9.0 + 1.0));
    entry.on_access(212);
    assert_eq!(entry.last_access_tick, 212);
    assert_eq!(entry.access_count, 4);
    assert_eq!(entry.frequency(212), 4.0 / (212.0 - 9.0 + 1.0));
    assert_eq!(entry.frequency(1000), 4.0 / (1000.0 - 9.0 + 1.0));
}

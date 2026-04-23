use lbce::core::policy::Policy;
use lbce::policies::lru::LruPolicy;

#[test]
fn test_lru() {
    let mut policy = LruPolicy::new(3);

    policy.insert(1);
    policy.insert(2);
    policy.insert(3);

    policy.on_hit(1);

    let evicted = policy.on_miss(4);
    assert_eq!(evicted, Some(2));

    policy.remove(2);
    policy.insert(4);

    let evicted = policy.on_miss(5);
    assert_eq!(evicted, Some(3));
}

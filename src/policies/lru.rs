use crate::core::policy::{Policy, CacheKey}; //bringing policy trait into scope
use std::collections::HashMap;

struct Node {
    key: CacheKey,
    next: Option<usize>,
    prev: Option<usize>
}

pub struct LruPolicy{
    nodes: Vec<Node>,
    map: HashMap<CacheKey, usize>,
    head: Option<usize>, //least recent O(1) access
    tail: Option<usize>, //most recent
    free: Vec<usize>, //recyclable indices
}

impl LruPolicy{
    //returns a new LRU policy
    pub fn new(capacity: usize) -> Self{
        //initialize LRU policy
        let nodes = (0..capacity)
            .map(|_| Node {
                key: 0,
                next: None,
                prev: None
            })
            .collect();

        Self{
            nodes: nodes,
            map: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            free: (0..capacity).rev().collect(), //indices [ capacity-1, ..., 0 ]
        }
    }
}

impl Policy for LruPolicy{
    fn on_hit(&mut self, key: CacheKey){
        let idx = *self.map.get(&key).expect("Key not found in LRU policy");
        
        //already most recent (tail), nothing to do
        if Some(idx) == self.tail{
            return; 
        }
        
        //extract prev, next nodes
        let (prev, next) = {
            let node = &self.nodes[idx];
            (node.prev, node.next)
        };

        match prev{
            Some(p) => self.nodes[p].next = next,
            None => self.head = next //current index is the head
        }

        if let Some(next_idx) = next{
            self.nodes[next_idx].prev = prev;
        }

        if let Some(tail_idx) = self.tail{
            self.nodes[tail_idx].next = Some(idx);
        }

        self.nodes[idx].prev = self.tail;
        self.nodes[idx].next = None;
        self.tail = Some(idx);
    }

    fn on_miss(&mut self, _key: CacheKey) {}

    fn insert(&mut self, key: CacheKey){
        debug_assert!(!self.map.contains_key(&key));
        
        let idx = self.free.pop().expect("No free nodes available");

        //create new node
        self.nodes[idx] = Node {
            key, 
            next: None, 
            prev: self.tail,
        };

        if let Some(tail_idx) = self.tail{
            self.nodes[tail_idx].next = Some(idx);
        } 
        else{
            self.head = Some(idx); //first node
        }

        self.tail = Some(idx);
        self.map.insert(key, idx);
    }

    fn remove(&mut self, key: CacheKey){
        let idx = self.map.remove(&key).expect("Key not found in LRU policy");

        let (prev, next) = {
            let node = &self.nodes[idx];
            (node.prev, node.next)
        };

        match prev{
            Some(p) => self.nodes[p].next = next,
            None => self.head = next //current index is the head
        }

        match next{
            Some(n) => self.nodes[n].prev = prev,
            None => self.tail = prev //current index is tail
        }

        self.free.push(idx); //recycle index
    }

    fn victim(&mut self) -> Option<CacheKey> {
        self.head.map(|head_idx| self.nodes[head_idx].key)
    }
}

impl LruPolicy {
    /// Walk from tail (most recent, rank 0) to head (least recent),
    /// returning (key, rank) pairs for every entry in the list.
    pub fn ranks(&self) -> Vec<(CacheKey, usize)> {
        let mut result = Vec::new();
        let mut rank = 0;
        let mut current = self.tail;
        while let Some(idx) = current {
            result.push((self.nodes[idx].key, rank));
            rank += 1;
            current = self.nodes[idx].prev;
        }
        result
    }
}
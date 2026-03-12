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
    capacity: usize
}

impl LruPolicy{
    //returns a new LRU policy
    pub fn new(capacity: usize) -> Self{
        //initialize LRU policy
        let nodes = (0..capacity).map(|_| Node {key: CacheKey::default(), next: None, prev: None}).collect();
        Self{
            nodes: nodes,
            map: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            free: (0..capacity).rev().collect(), //indices [ capacity-1, ..., 0 ]
            capacity
        }
    }
}

impl Policy for LruPolicy{
    fn on_hit(&mut self, key: CacheKey){
        let idx = self.map[&key];
        //if head, nothing to do
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
        };
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

    fn on_miss(&mut self, _key: CacheKey) -> Option<CacheKey>{
        if self.map.len() >= self.capacity{
            match self.head{
                Some(head_idx) => Some(self.nodes[head_idx].key),
                None => None
            }
        }else{
            None
        }
    }

    fn insert(&mut self, key: CacheKey){
        let idx = self.free.pop().expect("No free nodes available");

        //create new node
        self.nodes[idx] = Node {key: key.clone(), next:None, prev: self.tail};

        if let Some(tail_idx) = self.tail{
            self.nodes[tail_idx].next = Some(idx);
        }else{
            self.head = Some(idx); //first node
        }

        self.tail = Some(idx);
        self.map.insert(key, idx);
    }

    fn remove(&mut self, key: CacheKey){
        let idx = self.map.remove(&key).expect("Key not found in policy");

        let (prev, next) = {
            let node = &self.nodes[idx];
            (node.prev, node.next)
        };
        match prev{
            Some(p) => self.nodes[p].next = next,
            None => self.head = next //current index is the head
        };

        match next{
            Some(n) => self.nodes[n].prev = prev,
            None => self.tail = prev //current index is tail
        }

        self.free.push(idx); //recycle index
    }
}
use std::collections::HashMap;

// block size
const block_size: u32 = 4;

// equivalent to block
pub struct CacheLine<const BLOCKSIZE: usize> {
    pub valid: bool, // if the data is valid
    pub dirty: bool, // this is for a non-write through cache which we haven't implemented yet lmao
    pub tag: u32,
    pub index: u32,
    pub size: u32,
    pub data: [u8; BLOCKSIZE]
}

pub struct Set<const WAYS: usize, const BLOCKSIZE: usize> {
    pub lines: HashMap
}

// here we write back to MM see comment below this is why we need MM
pub struct WriteThroughCache {
    pub block_size: u32,
    pub num_ways: u32,
    pub num_sets: u32,
    pub policy: Policy,
    pub sets: [Set; num_sets]
}

/* 
we lowkey need main memory so that we know what
 blocks to fetch when we have a cache miss

 theoretically when new data is created the whole thing should be written to main memory, but that will
 happen via cache i guess?
*/

pub impl WriteThroughCache {
    fn new(block_size : u32, num_ways: u32, num_sets : u32, policy: Policy) -> Self {
        Self {            
            sets: [Set; num_sets] = std::array::from_fn(|_| 
            Set {
                lines: HashMap::with_capacity(num_ways)
            });
            block_size, num_ways, num_sets, policy
        }
    }

    
    fn miss(address) {}{
        u32 offset_bits = block_size.ilog2();
        u32 index_bits = num_sets.ilog2();
        u32 tag_bits = 32 - offset_bits - index_bits;
        
        u32 offset_mask = block_size - 1;
        u32 index_mask = (num_sets - 1) << offset_bits;
        u32 tag_mask = (-1) << (offset_bits + index_bits);

        u32 offset = address & offset_mask;
        u32 index = address & index_mask >> offset_bits;
        u32 tag = address & tag_mask >> (offset_bits + index_bits);

        // maybe we also need to tell policy that we had a miss on a store vs a load
        evict_choice = POLICY.evict_candidate(sets[index]) // policy tells us which one to evict out of the stuff in the sets
        

        // create the new block
        let new_line = CacheLine::<block_size> {
            valid: false,
            dirty: false,
            tag: tag,
            index: index,
            size: block_size, 
            data: [u8; 64]
        }
        

        // now we pull the entire corresponding block for the missed address
        u32 offset_bits = block_size.ilog2()
        u32 block_address = (address >> offset_bits) << offset_bits
        for index in 0..block_size {
            new_line.data[index] = MAINMEMORY[(block_address << offset_bits) | index]
        }

        sets[index][evict_choice] = new_line;
    }


    fn find(address : u32) -> Option<u8> {
        u32 offset_bits = block_size.ilog2();
        u32 index_bits = num_sets.ilog2();
        u32 tag_bits = 32 - offset_bits - index_bits;
        
        u32 offset_mask = block_size - 1;
        u32 index_mask = (num_sets - 1) << offset_bits;
        u32 tag_mask = (0xFFFFFFFF) << (offset_bits + index_bits);

        u32 offset = address & offset_mask;
        u32 index = address & index_mask >> offset_bits;
        u32 tag = address & tag_mask >> (offset_bits + index_bits);
        
        if tag in sets[index] {
            return Some sets[index][tag].data[offset]
        }
        return None
    }

    fn read(address : u32) -> u8 {
        match find(address) {
            Some(byte) => byte // on read, we good. MIGHT NEED TO TELL POLICY WE HAD A HIT
            None => { // store if needed, then return from mem
                miss(address);
                mem[address];
            }
        }
    }

    fn write(address : u32, val : u8) -> {
        mem[address] = val; // store always
        match find(address) {
            Some(byte) => byte { // on hit, we st
                sets[index][tag].data[offset] = val; // MIGHT NEED TO TELL POLICY WE HAD A HIT
            }
            None => {
                miss(address)
            }
        }
    }
}
use crate::core::policy::Policy;
use crate::core::policy::CacheKey;
use crate::deployed::EvictionMLP;
use crate::core::time::Clock;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::tensor::backend::Backend;

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::{FullPrecisionSettings, Recorder},
};

pub struct LearnedPolicy {
    model: EvictionMLP::<MyBackend>, //change
    metadata: HashMap<CacheKey, Vec<f64>>,
    currentElements: Set<CacheKey>,
}

FEATURE_COLS = [
    "resident_age_diff",
    "resident_time_since_last_diff",
    "resident_access_count_diff",
    "resident_frequency_diff",
    "global_age_since_first_request_diff",
    "global_time_since_last_request_diff",
    "global_total_request_count_diff",
    "last_interarrival_diff",
    "avg_interarrival_diff",
    "gap_count_diff",
    // "decay_0_diff",
    // "decay_1_diff",
    // "decay_2_diff",
]

// 0. is it resident in the cache, 0/1
// 1. what tick was it added
// 2. prev access tick
// 3. 2nd prev access tick
// 4. how many times as it been accessed since it was added
// 5. first request tick
// 6. how many total requests its had


// last_interarrival_diff - Difference in the most recent gap between requests for each object.
// avg_interarrival_diff - Difference in average time between requests.
// gap_count_diff - Difference in number of “gaps” (periods of inactivity or large interarrival times).



impl LearnedPolicy {

    pub fn new -> Self()
    {
        let device = Default::default();

        // File is {"state_dict": model.state_dict(), ...}; Sequential keys are net.0, net.3, net.6
        let args = LoadArgs::new("eviction_mlp.pt".into())
            .with_top_level_key("state_dict")
            .with_key_remap("net\\.0\\.(.*)", "fc1.$1")
            .with_key_remap("net\\.3\\.(.*)", "fc2.$1")
            .with_key_remap("net\\.6\\.(.*)", "fc3.$1");

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(args, &device)
            .expect("Should decode state successfully");
        Self {
            model : EvictionMLP::<MyBackend>::init(&device).load_record(record);
            metadata : HashMap::new(),
            currentElements: Set::new()
        }
    }

}

impl Policy for LearnedPolicy {
    fn on_hit(&mut self, _key: CacheKey, tick: u64) {
        temp = metadatap[_key][2];
        metadata.get_mut(_key)[2] = tick;
        metadata.get_mut(_key)[3] = temp;
        metadata.get_mut(_key)[4]++; 
        metadata.get_mut(_key)[6]++;

    }

    fn on_miss(&mut self, _key: CacheKe, tick : u64) {
        if (!metadata.contains_key(_key)) {
            metadata.insert(_key, vec![0, 0, 0, 0, 0, 0, 0]);
            metadata.get_mut(_key)[5] = tick;
        }
        metadata.get_mut(_key)[0] = 1;
        metadata.get_mut(_key)[1] = tick;
        metadata.get_mut(_key)[2] = tick;
        metadata.get_mut(_key)[3] = ??;
        metadata.get_mut(_key)[4]++;
        metadata.get_mut(_key)[6]++;
        currentElements.insert(_key);
    }

    fn insert(&mut self, key: CacheKey, u64 tick: u64) {
        
    }

    fn remove(&mut self, key: CacheKey, tick : u64) {
        currentElements.remove(key);
        metadata.get_mut(_key)[0] = 0;
    }

    fn victim(&mut self) -> Option<CacheKey> {
        // here we use the model
        let mut pq = BinaryHeap::new();
        pq.push(Reverse(10));
        for (CacheKey k : currentElements) {
            pq.push(Reverse((metadata.get(k)[1], k)))
        }

        Vec<CacheKey> victims = Vec::new();
        
        for (i in 0 .. 4) {
            victims.append(pq.pop());
        }

        feature_0_1 = Vec::new();

        let model = EvictionMLPNormalized::<MyBackend>::load("eviction_mlp.pt", &device);
        // DATASET HAS MANY ISSUES ........ 

        // first key minus second key or second minus first??? assuming first minus second
        feature_0_1.append(metadata.get(victims[0])[1] - metadata.get(victims[1])[1]); //resident_age_diff
        feature_0_1.append(metadata.get(victims[0])[2] - metadata.get(victims[1])[2]); //resident_time_since_last_diff
        feature_0_1.append(metadata.get(victims[0])[4] - metadata.get(victims[1])[4]); //resident_access_count_diff
        feature_0_1.append(metadata.get(victims[0])[4]/(tick - metadata.get(victims[0])[1]) - metadata.get(victims[1])[4]/(tick - metadata.get(victims[1])[1])); //resident_frequency_diff
        feature_0_1.append(metadata.get(victims[0])[5] - metadata.get(victims[1])[5]); //global_age_since_first_request_diff
        feature_0_1.append(metadata.get(victims[0])[2] - metadata.get(victims[1])[2]); //global_time_since_last_request_diff THIS IS WRONG AND IS THE SAME AS THE OTHER ONE
        feature_0_1.append(metadata.get(victims[0])[6] - metadata.get(victims[1])[6]); //global_total_request_count_diff
        feature_0_1.append((metadata.get(victims[0])[2] - metadata.get(victims[0])[3]) - (metadata.get(victims[1])[2] - metadata.get(victims[1])[3])) //last_interarrival_diff
        feature_0_1.append((metadata.get(victims[0])[2] - metadata.get(victims[0])[5])/(metadata.get(victims[0])[4]) - (metadata.get(victims[1])[2] - metadata.get(victims[1])[5])/metadata.get(victims[1])[4]) // average interrival
        feature_0_1.append(metadata.get(victims[0])[6] - 1 - metadata.get(victims[1])[6] - 1) //gap_count_diff      
        
        let feature_0_1_input = Tensor::<MyBackend, 1>::from_floats(feature_0_1.as_slice(), &device)
            .unsqueeze::<2>();
        let index1 = model.forward(feature_0_1_inp).into_scalar();; // index for final comp

        feature_2_3.append(metadata.get(victims[2])[1] - metadata.get(victims[3])[1]); //resident_age_diff
        feature_2_3.append(metadata.get(victims[2])[2] - metadata.get(victims[3])[2]); //resident_time_since_last_diff
        feature_2_3.append(metadata.get(victims[2])[4] - metadata.get(victims[3])[4]); //resident_access_count_diff
        feature_2_3.append(metadata.get(victims[2])[4]/(tick - metadata.get(victims[2])[1]) - metadata.get(victims[3])[4]/(tick - metadata.get(victims[3])[1])); //resident_frequency_diff
        feature_2_3.append(metadata.get(victims[2])[5] - metadata.get(victims[3])[5]); //global_age_since_first_request_diff
        feature_2_3.append(metadata.get(victims[2])[2] - metadata.get(victims[3])[2]); //global_time_since_last_request_diff THIS IS WRONG AND IS THE SAME AS THE OTHER ONE
        feature_2_3.append(metadata.get(victims[2])[6] - metadata.get(victims[3])[6]); //global_total_request_count_diff
        feature_2_3.append((metadata.get(victims[2])[2] - metadata.get(victims[2])[3]) - (metadata.get(victims[3])[2] - metadata.get(victims[3])[3])) //last_interarrival_diff
        feature_2_3.append((metadata.get(victims[2])[2] - metadata.get(victims[2])[5])/(metadata.get(victims[2])[4]) - (metadata.get(victims[3])[2] - metadata.get(victims[3])[5])/metadata.get(victims[3])[4]) // average interrival
        feature_2_3.append(metadata.get(victims[2])[6] - 1 - metadata.get(victims[3])[6] - 1) //gap_count_diff      

        let feature_2_3_input = Tensor::<MyBackend, 1>::from_floats(feature_2_3.as_slice(), &device)
            .unsqueeze::<2>();
        let index2 = model.forward(feature_2_3_inp).into_scalar();; // index for final comp

        feature_final.append(metadata.get(victims[index1])[1] - metadata.get(victims[index2])[1]); //resident_age_diff
        feature_final.append(metadata.get(victims[index1])[2] - metadata.get(victims[index2])[2]); //resident_time_since_last_diff
        feature_final.append(metadata.get(victims[index1])[4] - metadata.get(victims[index2])[4]); //resident_access_count_diff
        feature_final.append(metadata.get(victims[index1])[4]/(tick - metadata.get(victims[nindex1])[1]) - metadata.get(victims[index2])[4]/(tick - metadata.get(victims[index2])[1])); //resident_frequency_diff
        feature_final.append(metadata.get(victims[index1])[5] - metadata.get(victims[index2])[5]); //global_age_since_first_request_diff
        feature_final.append(metadata.get(victims[index1])[2] - metadata.get(victims[index2])[2]); //global_time_since_last_request_diff THIS IS WRONG AND IS THE SAME AS THE OTHER ONE
        feature_final.append(metadata.get(victims[index1])[6] - metadata.get(victims[index2])[6]); //global_total_request_count_diff
        feature_final.append((metadata.get(victims[index1])[2] - metadata.get(victims[nindex1])[3]) - (metadata.get(victims[index2])[2] - metadata.get(victims[index2])[3])) //last_interarrival_diff
        feature_final.append((metadata.get(victims[index1])[2] - metadata.get(victims[nindex1])[5])/(metadata.get(victimsn[index1])[4]) - (metadata.get(victims[index2])[2] - metadata.get(victims[index2])[5])/metadata.get(victims[index2])[4]) // average interrival
        feature_final.append(metadata.get(victims[index1])[6] - 1 - metadata.get(victims[index2])[6] - 1) //gap_count_diff      

        let feature_final_input = Tensor::<MyBackend, 1>::from_floats(feature_final.as_slice(), &device)
            .unsqueeze::<2>();
        let index3 = model.forward(feature_final_input).into_scalar();; // index for final comp

        return Some(victims[index3]);

    }
}
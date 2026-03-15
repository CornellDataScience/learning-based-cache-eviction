// reading and exporting trace data

use crate::core::policy::CacheKey;
use crate::core::trace::{CacheTrace, CacheEvent};
use csv::Reader;
use std::fs::File;
use stf::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

fn hash_key(key: &str) -> CacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

pub fn load_trace_csv(path: &str) -> Vec<(CacheKey, usize)> {

    let file = File::open(path).expect("Failed to open trace file");
    let mut reader = Reader::from_reader(file);

    let headers = reader.headers().expect("Missing headers");

    let key_index = headers.iter().position(|h| h == "key").expect("Column 'key' not found"); //name of the column with the key can differ

    let value_index = headers.iter().position(|h| h == "value_size").expect("Column 'value_size' not found"); //same here

    let mut requests = Vec::new();

    for result in reader.records() {

        let record = result.expect("Failed to read record");

        let key_str = &record[key_index];
        let value_size: usize = &record[value_index].parse().expect("Invalid value size");

        let key = hash_key(key_str);

        requests.push((key, value_size));
    }
    requests
}

pub fn write_trace_csv(path: &str, trace: &CacheTrace){
    let file = File::create(path).expect("Failed to create file");
    let mut writer = Writer::from_writer(file);

    writer
        .write_record(&["event", "key", "size_bytes", "tick"])
        .expect("Failed to write headers");
    
    for event in trace.events() {
        match event {
            CacheEvent::Hit { key, tick} =>{
                writer.write_record(&[
                    "hit",
                    &key.to_string(),
                    "",
                    &tick.to_string()
                ]).expect("Failed to write hit event");
            }
            CacheEvent::Miss{key, tick} => {
                writer.write_record(&[
                    "miss",
                    &key.to_string(),
                    "",
                    &tick.to_string()
                ]).expect("Failed to write miss event");
            }
            CacheEvent::Insert{key, size_bytes, tick} => {
                writer.write_record(&[
                    "insert",
                    &key.to_string(),
                    &size_bytes.to_string(),
                    &tick.to_string()
                ]).expect("Failed to write insert event");
            }
            CacheEvent::Evict{key, size_bytes, tick} => {
                writer.write_record(&[
                    "evict",
                    &key.to_string(),
                    &size_bytes.to_string(),
                    &tick.to_string()
                ]).expect("Failed to write evict event");
            }
        }
    }

    writer.flush().expect("Failed to flush writer");
}
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};

use csv::{Reader, Writer};

use crate::core::policy::CacheKey;
use crate::core::trace::{Request, RequestTrace};

fn hash_key(key: &str) -> CacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

pub fn load_request_trace_csv(path: &str) -> RequestTrace {

    let file = File::open(path).expect("Failed to open request trace file");
    let mut reader = Reader::from_reader(file);

    let headers = reader.headers().expect("Missing headers").clone();

    let key_index = headers.iter().position(|h| h == "key").expect("Column 'key' not found");

    let mut trace = RequestTrace::new();

    for result in reader.records() {

        let record: csv::StringRecord = result.expect("Failed to read record");

        let key_str = &record[key_index];

        let key = hash_key(key_str);

        trace.push(Request::new(key));
    }
    trace
}

pub fn write_request_trace_csv(path: &str, trace: &RequestTrace){
    let file = File::create(path).expect("Failed to create request trace file");
    let mut writer = Writer::from_writer(file);

    writer
        .write_record(["key"])
        .expect("Failed to write request trace headers");
    
    for request in trace.requests() {
        writer
            .write_record([request.key.to_string()])
            .expect("Failed to write request trace row");
    }

    writer.flush().expect("Failed to flush request trace writer");
}
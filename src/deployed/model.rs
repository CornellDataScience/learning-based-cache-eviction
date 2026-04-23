use std::fs;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::{FullPrecisionSettings, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use serde::Deserialize;

pub const FEATURE_DIM: usize = 13;

#[derive(Module, Debug)]
pub struct EvictionMLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> EvictionMLP<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(FEATURE_DIM, 64).init(device),
            fc2: LinearConfig::new(64, 32).init(device),
            fc3: LinearConfig::new(32, 1).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = self.relu.forward(self.fc1.forward(x));
        let x = self.relu.forward(self.fc2.forward(x));
        self.fc3.forward(x).squeeze(1)
    }
}

#[derive(Debug, Deserialize)]
struct ModelMetadata {
    mean: Vec<f32>,
    std: Vec<f32>,
}

pub struct EvictionMLPNormalized<B: Backend> {
    model: EvictionMLP<B>,
    mean: [f32; FEATURE_DIM],
    std: [f32; FEATURE_DIM],
    device: B::Device,
}

impl<B: Backend<FloatElem = f32>> EvictionMLPNormalized<B> {
    pub fn load(path: impl AsRef<Path>, device: &B::Device) -> Result<Self, String> {
        let path = path.as_ref().to_string_lossy().into_owned();

        let model_args = LoadArgs::new(path.clone().into())
            .with_top_level_key("state_dict")
            .with_key_remap("net\\.0\\.(.*)", "fc1.$1")
            .with_key_remap("net\\.2\\.(.*)", "fc2.$1")
            .with_key_remap("net\\.4\\.(.*)", "fc3.$1")
            .with_key_remap("net\\.3\\.(.*)", "fc2.$1")
            .with_key_remap("net\\.6\\.(.*)", "fc3.$1");

        let model_record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(model_args, device)
            .map_err(|err| format!("failed to load model weights from {path}: {err}"))?;
        let model = EvictionMLP::<B>::init(device).load_record(model_record);

        let metadata = Self::load_metadata(&path)?;

        Ok(Self {
            model,
            mean: metadata.mean,
            std: metadata.std,
            device: device.clone(),
        })
    }

    fn load_metadata(model_path: &str) -> Result<ModelMetadataFixed, String> {
        let metadata_path = metadata_path_for(model_path);
        let raw = fs::read_to_string(&metadata_path).map_err(|err| {
            format!(
                "failed to read normalization metadata from {}: {err}",
                metadata_path.display()
            )
        })?;
        let metadata: ModelMetadata = serde_json::from_str(&raw).map_err(|err| {
            format!(
                "failed to parse normalization metadata from {}: {err}",
                metadata_path.display()
            )
        })?;
        ModelMetadataFixed::try_from(metadata).map_err(|err| {
            format!(
                "invalid normalization metadata in {}: {err}",
                metadata_path.display()
            )
        })
    }

    pub fn predict_pair(&self, raw_features: &[f32; FEATURE_DIM]) -> f32 {
        let input =
            Tensor::<B, 1>::from_floats(raw_features.as_slice(), &self.device).unsqueeze::<2>();

        let sign = input.clone().sign();
        let transformed = sign * (input.abs() + 1.0f32).log();

        let mean = Tensor::<B, 1>::from_floats(self.mean.as_slice(), &self.device).unsqueeze::<2>();
        let std = Tensor::<B, 1>::from_floats(self.std.as_slice(), &self.device).unsqueeze::<2>();
        let normalized = (transformed - mean) / std;

        self.model.forward(normalized).into_scalar()
    }
}

struct ModelMetadataFixed {
    mean: [f32; FEATURE_DIM],
    std: [f32; FEATURE_DIM],
}

impl TryFrom<ModelMetadata> for ModelMetadataFixed {
    type Error = String;

    fn try_from(value: ModelMetadata) -> Result<Self, Self::Error> {
        if value.mean.len() != FEATURE_DIM {
            return Err(format!(
                "expected {} mean values, found {}",
                FEATURE_DIM,
                value.mean.len()
            ));
        }
        if value.std.len() != FEATURE_DIM {
            return Err(format!(
                "expected {} std values, found {}",
                FEATURE_DIM,
                value.std.len()
            ));
        }

        Ok(Self {
            mean: value
                .mean
                .try_into()
                .map_err(|_| "failed to convert mean vector to fixed-size array".to_string())?,
            std: value
                .std
                .try_into()
                .map_err(|_| "failed to convert std vector to fixed-size array".to_string())?,
        })
    }
}

fn metadata_path_for(model_path: &str) -> PathBuf {
    let path = PathBuf::from(model_path);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("eviction_mlp");
    let metadata_name = format!("{stem}.meta.json");

    match path.parent() {
        Some(parent) => parent.join(metadata_name),
        None => PathBuf::from(metadata_name),
    }
}

use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn::tensor::backend::Backend;

use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    record::{FullPrecisionSettings, Recorder},
};

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
            fc1: LinearConfig::new(13, 64).init(device),
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

/// Holds the z-score normalization stats (mean and std) saved alongside the model weights.
#[derive(Module, Debug)]
pub struct NormStats<B: Backend> {
    mean: Param<Tensor<B, 1>>,
    std: Param<Tensor<B, 1>>,
}

impl<B: Backend> NormStats<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            mean: Param::from_tensor(Tensor::zeros([13], device)),
            std: Param::from_tensor(Tensor::ones([13], device)),
        }
    }
}

/// Wraps EvictionMLP with the preprocessing that was applied during training:
///   1. sign(x) * log1p(|x|)  — log-scale compression
///   2. z-score normalisation using the training mean/std stored in the .pt file
pub struct EvictionMLPNormalized<B: Backend> {
    model: EvictionMLP<B>,
    norm: NormStats<B>,
}

impl<B: Backend> EvictionMLPNormalized<B> {
    pub fn load(path: &str, device: &B::Device) -> Self {
        let model_args = LoadArgs::new(path.into())
            .with_top_level_key("state_dict")
            .with_key_remap("net\\.0\\.(.*)", "fc1.$1")
            .with_key_remap("net\\.3\\.(.*)", "fc2.$1")
            .with_key_remap("net\\.6\\.(.*)", "fc3.$1");

        let model_record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(model_args, device)
            .expect("failed to load model weights");
        let model = EvictionMLP::<B>::init(device).load_record(model_record);

        // mean/std are at the top level of the .pt file, not under state_dict
        let norm_record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(LoadArgs::new(path.into()), device)
            .expect("failed to load norm stats");
        let norm = NormStats::<B>::init(device).load_record(norm_record);

        Self { model, norm }
    }

    /// Accepts a raw [batch, 13] feature tensor and returns logits.
    /// Positive logit → key0 should be evicted before key1.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        // log-scale compression: sign(x) * log(1 + |x|)
        let sign = x.clone().sign();
        let x = sign * (x.abs() + 1.0f32).log();

        // z-score normalisation — unsqueeze to [1, 13] for broadcasting
        let mean = self.norm.mean.val().unsqueeze::<2>();
        let std = self.norm.std.val().unsqueeze::<2>();
        let x = (x - mean) / std;

        self.model.forward(x)
    }
}

type MyBackend = burn_ndarray::NdArray<f32>;

fn main() {
    let device = Default::default();
    let model = EvictionMLPNormalized::<MyBackend>::load("eviction_mlp.pt", &device);

    // example: compare two candidates
    // let features: Vec<f32> = subtract_features(&phi0, &phi1); // length 13
    // let input = Tensor::<MyBackend, 1>::from_floats(features.as_slice(), &device).unsqueeze::<2>();
    // let logit = model.forward(input).into_scalar();
    // evict key0 if logit > 0.0
}

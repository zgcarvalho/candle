#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
mod model;

use anyhow::{anyhow, Error as E, Result};
use clap::Parser;
use candle::{Tensor, DType, Device};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;
use model::{Config, EsmModel};
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};

#[derive(Parser, Debug)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
    /// Run offline (you must have the files already cached)
    #[arg(long)]
    offline: bool,
}

impl Args {
    fn build_model(&self) -> Result<EsmModel> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "facebook/esm2_t6_8M_UR50D".to_string();
        let repo = Repo::model(default_model);
        let (config_filename, weights_filename) = if self.offline {
            let cache = Cache::default().repo(repo);
            (
                cache
                    .get("config.json")
                    .ok_or(anyhow!("Missing config file in cache"))?,
                cache
                    .get("model.safetensors")
                    .ok_or(anyhow!("Missing weights file in cache"))?,
            )
        } else {
            let api = Api::new()?;
            let api = api.repo(repo);
            (
                api.get("config.json")?,
                api.get("model.safetensors")?,
            )
        };
        let config = std::fs::read_to_string(config_filename)?;
        // println!("{}", config);
        // let config: Config = serde_json::from_str(&config)?; // TODO:
        // let config = Config::default();
        let config = Config::from_string(config)?;
        println!("{:?}", config);
        let weights = candle::safetensors::load(weights_filename, &device)?;
        println!("{:?}", weights);
        let vb = VarBuilder::from_tensors(weights, candle::DType::F32, &device);
        let model = EsmModel::load(vb.pp("esm"), &config)?; // TODO:
        Ok(model)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _model = args.build_model()?;
    let seq = Tensor::ones((2, 12), DType::I64, &Device::Cpu)?;
    let emb = _model.forward(&seq, &seq);
    println!("{:?}", emb);
    // let tokenizer = args.build_tokenizer()?;
    // let model = args.build_model()?;
    Ok(())
}

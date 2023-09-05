use candle::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, LayerNorm};
use serde::Deserialize;

// NOTE: I think this can be replaced by candle_nn::embedding
fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

// NOTE: Dropout is not implemented in candle_nn yet
struct Dropout {
    #[allow(dead_code)]
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

// fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
//     let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
//         (Ok(weight), Ok(bias)) => (weight, bias),
//         (Err(err), _) | (_, Err(err)) => {
//             if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
//                 (weight, bias)
//             } else {
//                 return Err(err);
//             }
//         }
//     };
//     Ok(LayerNorm::new(weight, bias, eps))
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    Absolute,
    RelativeKey,
    RelativeKeyQuery,
    #[default]
    Rotary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
        vocab_size: usize,
        mask_token_id: usize,
        pad_token_id: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        hidden_dropout_prob: f64,
        attention_probs_dropout_prob: f64,
        max_position_embeddings: usize,
        initializer_range: f64,
        layer_norm_eps: f64,
        hidden_act: HiddenAct,
        #[serde(default)]
        position_embedding_type: PositionEmbeddingType,
        use_cache: bool,
        token_dropout: bool,
        is_folding_model: bool,
        emb_layer_norm_before: bool,
        classifier_dropout: Option<f64>,
        esmfold_config: Option<String>,
        vocab_list: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 33,
            mask_token_id: 32,
            pad_token_id: 1,
            hidden_size: 320,
            num_hidden_layers: 6,
            num_attention_heads: 20,
            intermediate_size: 1280,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 1026,
            initializer_range: 0.02,
            layer_norm_eps: 1e-05,
            hidden_act: HiddenAct::Gelu,
            position_embedding_type: PositionEmbeddingType::Rotary,
            use_cache: true,
            token_dropout: true,
            is_folding_model: false,
            emb_layer_norm_before: false,
            classifier_dropout: None,
            esmfold_config: None,
            vocab_list: None,
        }
    }
}

impl Config {
    pub fn from_string(config: String) -> Result<Self> {
        Ok(serde_json::from_str(&config).unwrap())
    }
}


//
//https://github.dev/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/esm/modeling_esm.py#L165
struct EsmEmbeddings {
    word_embeddings: Embedding, 
    position_embeddings: Embedding,
    dropout: Dropout,
    // layer_norm: Option<LayerNorm>,
    // register_buffer // FIX: what is this?
    // padding_idx
    // token_dropout
    // mask_token_id
}

impl EsmEmbeddings {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // FIX: how to pass padding_idx?
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        // FIX: how to pass padding_idx?
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        // let layer_norm = layer_norm(
        //     config.hidden_size,
        //     config.layer_norm_eps,
        //     vb.pp("LayerNorm"),
        // )?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            dropout,
            // layer_norm,
            // padding_idx: config.pad_token_id,
            // token_dropout: config.token_dropout,
            // mask_token_id: config.mask_token_id,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let mut embeddings = input_embeddings; // TODO: replace this with positions etc
        // if let Some(layer_norm) = &self.layer_norm {
        //     embeddings = layer_norm.forward(&embeddings)?
        // }
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

pub struct EsmModel {
    // config // FIX: 
    embeddings: EsmEmbeddings,
    // encoder: EsmEncoder,
    // pooler: Option<EsmPooler>,
    // contact_head: EsmContactPredictionHead,
    // post_init // FIX:
}

impl EsmModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = EsmEmbeddings::load(vb.pp("embeddings"), config)?;
        Ok(Self {
            embeddings,
        })
    }
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut embeddings = self.embeddings.forward(input_ids, attention_mask)?;
        Ok(embeddings)
    }
}

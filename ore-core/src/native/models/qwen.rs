//! Qwen model implementation with quantization support.
//!
//! Qwen is a chat-optimized language model that supports 8-bit quantization
//! for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group Query Attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - [Model Card](https://huggingface.co/Qwen/Qwen2)
//!

use crate::native::engine::{ModelConfig, OreEngine};
use crate::native::models::qwen::ModelWeights as QwenModel;
use crate::swap::ContextMessage;
use candle_core::{
    DType, Device, IndexOp, Result, Tensor,
    quantized::{QMatMul, gguf_file},
};
use candle_nn::{Embedding, Module};
use candle_transformers::{quantized_nn::RmsNorm, utils::repeat_kv};
use std::collections::HashMap;
use std::io::{Read, Seek};
use tokenizers::Tokenizer;

pub fn load<R: Read + Seek>(
    _model_name: &str,
    model_content: gguf_file::Content,
    reader: &mut R,
    device: &Device,
    tokenizer: &Tokenizer,
) -> anyhow::Result<(OreEngine, ModelConfig)> {

    let arch_name = match model_content.metadata.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
        _ => "qwen2".to_string(), // Fallback
    };

    let m = QwenModel::from_gguf(model_content, reader, device)?;

    let mut stop_tokens = vec![151645, 151643];
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        stop_tokens.push(id);
    }
    if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
        stop_tokens.push(id);
    }

    let formatter: fn(&[ContextMessage], &str) -> String = {
        |history, prompt| {
            let mut out = String::new();

            let mut has_system = false;
            for msg in history {
                if msg.role == "system" {
                    has_system = true;
                }
                out.push_str(&format!(
                    "<|im_start|>{}\n{}<|im_end|>\n",
                    msg.role, msg.content
                ));
            }
            if !has_system {
                out.insert_str(
                    0,
                    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n",
                );
            }

            out.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            ));

            out
        }
    };

    let cfg = ModelConfig {
        architecture: arch_name,
        stop_tokens,
        formatter,
    };

    Ok((OreEngine::Qwen(m), cfg))
}

#[derive(Debug, Clone)]
pub struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
pub enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE { feed_forward_gate_inp, experts, n_expert_used } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs_flat = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs_flat)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        sum_routing_weights += rw[expert_idx];
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        selected_rws[expert_idx].push(rw[expert_idx] / sum_routing_weights)
                    }
                }

                let mut ys = xs_flat.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x_idx = &top_x[expert_idx];
                    if top_x_idx.is_empty() { continue; }
                    let top_x_t = Tensor::new(top_x_idx.as_slice(), xs_flat.device())?;
                    let selected_rws_t = Tensor::new(selected_rws[expert_idx].as_slice(), xs_flat.device())?.reshape(((), 1))?;
                    
                    let current_state = xs_flat.index_select(&top_x_t, 0)?.reshape(((), hidden_dim))?;
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states = current_hidden_states.broadcast_mul(&selected_rws_t)?;
                    ys = ys.index_add(&top_x_t, &current_hidden_states, 0)?;
                }
                ys.reshape((b_size, seq_len, hidden_dim))
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_bq: Tensor,
    pub attention_bk: Tensor,
    pub attention_bv: Tensor,
    pub attention_wo: QMatMul,
    pub attention_norm: RmsNorm,
    pub mlp_or_moe: MlpOrMoe,
    pub ffn_norm: RmsNorm,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub cos: Tensor,
    pub sin: Tensor,
    pub neg_inf: Tensor,
    pub kv_cache: Option<(Tensor, Tensor)>,
    pub span_attn: tracing::Span,
    pub span_rot: tracing::Span,
    pub span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.broadcast_add(&self.attention_bq)?;
        let k = k.broadcast_add(&self.attention_bk)?;
        let v = v.broadcast_add(&self.attention_bv)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // let (q, k) = self
        //     .rotary_embedding
        //     .apply_rotary_emb_qkv(&q, &k, index_pos)?;
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Support for MQA, useful for 70B models and mistral.
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    pub tok_embeddings: Embedding,
    pub layers: Vec<LayerWeights>,
    pub norm: RmsNorm,
    pub output: QMatMul,
    pub masks: HashMap<(usize, usize), Tensor>,
    pub span: tracing::Span,
    pub span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let arch = match ct.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(s)) => s.clone(),
            _ => "qwen2".to_string(), // Fallback
        };
        
        println!("-> [CANDLE] Parsing metadata for architecture family: '{}'", arch);

        let n_expert = md_get(&format!("{}.expert_count", arch))
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
            
        let n_expert_used = md_get(&format!("{}.expert_used_count", arch))
            .ok()
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;

        let head_count = md_get(&format!("{}.attention.head_count", arch))?.to_u32()? as usize;
        let head_count_kv = md_get(&format!("{}.attention.head_count_kv", arch))?.to_u32()? as usize;
        let embedding_length = md_get(&format!("{}.embedding_length", arch))?.to_u32()? as usize;
        let context_length = md_get(&format!("{}.context_length", arch))?.to_u32()? as usize;
        let block_count = md_get(&format!("{}.block_count", arch))?.to_u32()? as usize;
        let rms_norm_eps = md_get(&format!("{}.attention.layer_norm_rms_epsilon", arch))?.to_f32()? as f64;
        let rope_freq_base = md_get(&format!("{}.rope.freq_base", arch))
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let head_dim = embedding_length / head_count;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => {
                // use tie_word_embeddings
                QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?
            }
        };

        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;

        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;

            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp_or_moe = if n_expert <= 1 {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                })
            } else {
                let feed_forward_gate_inp = ct.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let feed_forward_w1 = ct.tensor(reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let feed_forward_w2 = ct.tensor(reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let feed_forward_w3 = ct.tensor(reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                    })
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(feed_forward_gate_inp)?,
                    experts,
                }
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: attention_bq.dequantize(device)?,
                attention_bk: attention_bk.dequantize(device)?,
                attention_bv: attention_bv.dequantize(device)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                cos: cos.clone(),
                sin: sin.clone(),
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    /// Clear the KV cache across all layers.
    ///
    /// Call this between independent conversations to free cached attention
    /// state without recreating the model.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.kv_cache = None;
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp_or_moe.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}

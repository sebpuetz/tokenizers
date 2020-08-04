extern crate tokenizers as tk;
use std::sync::Arc;

use neon::prelude::*;
use tk::normalizers::NormalizerWrapper;
use serde::{Serialize, Serializer};
use serde::ser::SerializeStruct;

use crate::extraction::*;

/// Normalizer
pub struct Normalizer {
    pub normalizer: Option<JsNormalizerWrapper>,
}

declare_types! {
    pub class JsNormalizer for Normalizer {
        init(_) {
            // This should not be called from JS
            Ok(Normalizer {
                normalizer: None
            })
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BertNormalizerOptions {
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: Option<bool>,
    lowercase: bool,
}
impl Default for BertNormalizerOptions {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    }
}

/// bert_normalizer(options?: {
///   cleanText?: bool = true,
///   handleChineseChars?: bool = true,
///   stripAccents?: bool = true,
///   lowercase?: bool = true
/// })
fn bert_normalizer(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let options = cx
        .extract_opt::<BertNormalizerOptions>(0)?
        .unwrap_or_else(BertNormalizerOptions::default);

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::bert::BertNormalizer::new(
            options.clean_text,
            options.handle_chinese_chars,
            options.strip_accents,
            options.lowercase,
        ).into());
    Ok(normalizer)
}

/// nfd()
fn nfd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::unicode::NFD.into());
    Ok(normalizer)
}

/// nfkd()
fn nfkd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::unicode::NFKD.into());
    Ok(normalizer)
}

/// nfc()
fn nfc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::unicode::NFC.into());
    Ok(normalizer)
}

/// nfkc()
fn nfkc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::unicode::NFKC.into());
    Ok(normalizer)
}

/// strip(left?: boolean, right?: boolean)
fn strip(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let left = cx.extract_opt::<bool>(0)?.unwrap_or(true);
    let right = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::strip::Strip::new(left, right).into());

    Ok(normalizer)
}

/// sequence(normalizers: Normalizer[])
fn sequence(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizers = Vec::new();
    for normalizer in cx.argument::<JsArray>(0)?.to_vec(&mut cx)? {
        let normalizer = normalizer.downcast::<JsNormalizer>().or_throw(&mut cx)?;
        let guard = cx.lock();
        let normalizer = normalizer.borrow(&guard);
        match normalizer.normalizer.as_ref().unwrap().clone() {
            JsNormalizerWrapper::Wrapped(norm) => normalizers.push(norm),
            JsNormalizerWrapper::Sequence(seq) => normalizers.extend(seq),
        }
    }

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(JsNormalizerWrapper::Sequence(normalizers));
    Ok(normalizer)
}

/// lowercase()
fn lowercase(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .replace(tk::normalizers::utils::Lowercase.into());
    Ok(normalizer)
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum JsNormalizerWrapper {
    Sequence(Vec<Arc<NormalizerWrapper>>),
    Wrapped(Arc<NormalizerWrapper>),
}

impl Serialize for JsNormalizerWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
        where
            S: Serializer,
    {
        match self {
            JsNormalizerWrapper::Sequence(seq) => {
                let mut ser = serializer.serialize_struct("Sequence", 2)?;
                ser.serialize_field("type", "Sequence")?;
                ser.serialize_field("normalizers", seq)?;
                ser.end()
            }
            JsNormalizerWrapper::Wrapped(inner) => inner.serialize(serializer),
        }
    }
}

impl<I> From<I> for JsNormalizerWrapper
    where
        I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        JsNormalizerWrapper::Wrapped(Arc::new(norm.into()))
    }
}

impl<I> From<I> for Normalizer
    where
        I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        Normalizer {
            normalizer: Some(norm.into().into()),
        }
    }
}

#[typetag::serde]
impl tk::Normalizer for JsNormalizerWrapper {
    fn normalize(&self, normalized: &mut tk::NormalizedString) -> tk::Result<()> {
        match self {
            JsNormalizerWrapper::Wrapped(inner) => inner.normalize(normalized),
            JsNormalizerWrapper::Sequence(inner) => {
                inner.iter().map(|n| n.normalize(normalized)).collect()
            }
        }
    }
}


/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BertNormalizer", prefix), bert_normalizer)?;
    m.export_function(&format!("{}_NFD", prefix), nfd)?;
    m.export_function(&format!("{}_NFKD", prefix), nfkd)?;
    m.export_function(&format!("{}_NFC", prefix), nfc)?;
    m.export_function(&format!("{}_NFKC", prefix), nfkc)?;
    m.export_function(&format!("{}_Sequence", prefix), sequence)?;
    m.export_function(&format!("{}_Lowercase", prefix), lowercase)?;
    m.export_function(&format!("{}_Strip", prefix), strip)?;
    Ok(())
}

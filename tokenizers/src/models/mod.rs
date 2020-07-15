//! Popular tokenizer models.

pub mod bpe;
pub mod wordlevel;
pub mod wordpiece;

use crate::{Model, Token};

use serde::{Serialize, Serializer, Deserialize};
use std::collections::HashMap;
use std::path::{PathBuf, Path};

/// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in order
/// of token ID, smallest to largest.
struct OrderedVocabIter<'a> {
    vocab_r: &'a HashMap<u32, String>,
}

impl<'a> OrderedVocabIter<'a> {
    fn new(vocab_r: &'a HashMap<u32, String>) -> Self {
        Self { vocab_r }
    }
}

impl<'a> Serialize for OrderedVocabIter<'a> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let iter = (0u32..(self.vocab_r.len() as u32)).map(|i| (&self.vocab_r[&i], i));
        serializer.collect_map(iter)
    }
}

#[derive(Deserialize, Serialize)]
pub enum ModelWrapper {
    WordPiece(wordpiece::WordPiece),
    BPE(bpe::BPE),
    WordLevel(wordlevel::WordLevel)
}

impl From<wordlevel::WordLevel> for ModelWrapper {
    fn from(wl: wordlevel::WordLevel) -> Self {
        ModelWrapper::WordLevel(wl)
    }
}
impl From<wordpiece::WordPiece> for ModelWrapper {
    fn from(wp: wordpiece::WordPiece) -> Self {
        ModelWrapper::WordPiece(wp)
    }
}
impl From<bpe::BPE> for ModelWrapper {
    fn from(bpe: bpe::BPE) -> Self {
        ModelWrapper::BPE(bpe)
    }
}

#[typetag::serde]
impl Model for ModelWrapper {
    fn tokenize(&self, tokens: Vec<(String, (usize, usize))>) -> crate::Result<Vec<Token>> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.tokenize(tokens),
            WordPiece(t) => t.tokenize(tokens),
            BPE(t) => t.tokenize(tokens)
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.token_to_id(token),
            WordPiece(t) => t.token_to_id(token),
            BPE(t) => t.token_to_id(token)
        }
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.id_to_token(id),
            WordPiece(t) => t.id_to_token(id),
            BPE(t) => t.id_to_token(id)
        }
    }

    fn get_vocab(&self) -> &HashMap<String, u32> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.get_vocab(),
            WordPiece(t) => t.get_vocab(),
            BPE(t) => t.get_vocab()
        }
    }

    fn get_vocab_size(&self) -> usize {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.get_vocab_size(),
            WordPiece(t) => t.get_vocab_size(),
            BPE(t) => t.get_vocab_size()
        }
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> crate::Result<Vec<PathBuf>> {
        use ModelWrapper::*;
        match self {
            WordLevel(t) => t.save(folder, name),
            WordPiece(t) => t.save(folder, name),
            BPE(t) => t.save(folder, name)
        }
    }
}
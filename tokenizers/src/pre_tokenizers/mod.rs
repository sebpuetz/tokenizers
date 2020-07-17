pub mod bert;
pub mod byte_level;
pub mod delimiter;
pub mod metaspace;
pub mod whitespace;

use serde::{Deserialize, Serialize};

use crate::pre_tokenizers::bert::BertPreTokenizer;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::delimiter::CharDelimiterSplit;
use crate::pre_tokenizers::metaspace::Metaspace;
use crate::pre_tokenizers::whitespace::Whitespace;
use crate::{PreTokenizer, NormalizedString};

#[derive(Serialize, Deserialize)]
pub enum PreTokenizerWrapper {
    BertPreTokenizer(BertPreTokenizer),
    ByteLevel(ByteLevel),
    Delimiter(CharDelimiterSplit),
    Metaspace(Metaspace),
    Whitespace(Whitespace),
}

#[typetag::serde]
impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, normalized: &mut NormalizedString) -> crate::Result<Vec<(String, (usize, usize))>> {
        match self {
            PreTokenizerWrapper::BertPreTokenizer(bpt) => bpt.pre_tokenize(normalized),
            PreTokenizerWrapper::ByteLevel(bpt) => bpt.pre_tokenize(normalized),
            PreTokenizerWrapper::Delimiter(dpt) => dpt.pre_tokenize(normalized),
            PreTokenizerWrapper::Metaspace(mspt) => mspt.pre_tokenize(normalized),
            PreTokenizerWrapper::Whitespace(wspt) => wspt.pre_tokenize(normalized),
        }
    }
}
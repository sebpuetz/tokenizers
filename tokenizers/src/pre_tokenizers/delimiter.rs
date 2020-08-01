use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use crate::tokenizer::{NormalizedString, Offsets, PreTokenizer, Result};

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct CharDelimiterSplit {
    delimiter: char,
}

impl CharDelimiterSplit {
    pub fn new(delimiter: char) -> Self {
        CharDelimiterSplit { delimiter }
    }
}

impl PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, normalized: &mut NormalizedString) -> Result<Vec<(String, Offsets)>> {
        let mut words = vec![];
        let mut word = Vec::with_capacity(1000);
        let mut offset = 0;

        normalized.get().chars().for_each(|c| {
            if c == self.delimiter {
                if !word.is_empty() {
                    let offsets = (offset - word.len(), offset);
                    words.push((word.drain(0..).collect::<String>(), offsets));
                }
            } else {
                word.push(c);
            }
            offset += 1;
        });
        if !word.is_empty() {
            let offsets = (offset - word.len(), offset);
            words.push((word.drain(0..).collect::<String>(), offsets));
        }

        Ok(words)
    }
}

impl Serialize for CharDelimiterSplit {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("CharDelimiterSplit", 2)?;
        m.serialize_field("type", "CharDelimiterSplit")?;
        m.serialize_field("delimiter", &self.delimiter)?;
        m.end()
    }
}

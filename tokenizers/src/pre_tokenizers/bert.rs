use crate::tokenizer::{NormalizedString, Offsets, PreTokenizer, Result};
use serde::{Deserialize, Serialize};
use std::mem;
use unicode_categories::UnicodeCategories;

fn is_bert_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

/// Split the given string as the `should_split` predicate dictates. Keep track of the offsets
fn split_on<F: Fn(char) -> (bool, bool)>(s: &str, should_split: F) -> Vec<(String, Offsets)> {
    let mut words: Vec<(String, Offsets)> = vec![];
    let mut offset = 0;
    let mut word = String::with_capacity(20);

    s.chars().for_each(|c| {
        let (is_delim, include) = should_split(c);
        if is_delim {
            if !word.is_empty() {
                let offsets = (offset - word.len(), offset);
                words.push((mem::replace(&mut word, String::with_capacity(20)), offsets));
            }
            if include {
                words.push((c.to_string(), (offset, offset + c.len_utf8())));
            }
        } else {
            word.push(c);
        }
        offset += c.len_utf8();
    });
    // Don't forget the potential last word
    if !word.is_empty() {
        let offsets = (offset - word.len(), offset);
        words.push((word, offsets));
    }

    words
}

#[derive(Serialize, Deserialize)]
pub struct BertPreTokenizer;

#[typetag::serde]
impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, normalized: &mut NormalizedString) -> Result<Vec<(String, Offsets)>> {
        Ok(split_on(normalized.get(), |c| {
            if char::is_whitespace(c) {
                (true, false)
            } else if is_bert_punc(c) {
                (true, true)
            } else {
                (false, false)
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let pretok = BertPreTokenizer;
        let mut input = NormalizedString::from("Hey friend!     How are you?!?");
        let res = pretok.pre_tokenize(&mut input).unwrap();
        assert_eq!(
            &res,
            &[
                ("Hey".into(), (0, 3)),
                ("friend".into(), (4, 10)),
                ("!".into(), (10, 11)),
                ("How".into(), (16, 19)),
                ("are".into(), (20, 23)),
                ("you".into(), (24, 27)),
                ("?".into(), (27, 28)),
                ("!".into(), (28, 29)),
                ("?".into(), (29, 30)),
            ]
        );
    }
}

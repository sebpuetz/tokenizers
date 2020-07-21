//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::tokenizer::{Model, Offsets, Result, Token};
use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

mod serialization;
mod trainer;
pub use trainer::*;
use std::borrow::Cow;

#[derive(Debug)]
pub enum Error {
    MissingUnkToken,
}
impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingUnkToken => write!(
                fmt,
                "WordPiece error: Missing [UNK] token from the vocabulary"
            ),
        }
    }
}

type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

/// A `WordPieceBuilder` can be used to create a `WordPiece` model with a custom configuration.
pub struct WordPieceBuilder {
    config: Config,
}

impl Default for WordPieceBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: HashMap::new(),
                unk_token: String::from("[UNK]"),
                continuing_subword_prefix: String::from("##"),
                max_input_chars_per_word: 100,
            },
        }
    }
}

impl WordPieceBuilder {
    /// Construct a new `WordPieceBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    pub fn vocab(mut self, vocab: Vocab) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Set the prefix for continuing subwords.
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    /// Contructs a `WordPiece` model that uses the `WordPieceBuilder`'s configuration.
    pub fn build(mut self) -> Result<WordPiece> {
        if let Some(vocab) = self.config.files {
            self.config.vocab = WordPiece::read_files(&vocab)?;
        }

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        Ok(WordPiece {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

/// A
/// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
#[derive(PartialEq)]
pub struct WordPiece {
    vocab: Vocab,
    vocab_r: VocabR,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl std::fmt::Debug for WordPiece {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("WordPiece")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            unk_token: String::from("[UNK]"),
            continuing_subword_prefix: String::from("##"),
            max_input_chars_per_word: 100,
        }
    }
}

impl WordPiece {
    /// Get a `WordPieceBuilder`.
    pub fn builder() -> WordPieceBuilder {
        WordPieceBuilder::new()
    }

    /// Read the given files to extract the vocab
    pub fn read_files(vocab: &str) -> Result<Vocab> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    /// Initialize a `WordPiece` model from a vocab mapping file.
    pub fn from_files(vocab: &str) -> WordPieceBuilder {
        WordPiece::builder().files(vocab.to_owned())
    }

    /// Create a `WordPiece` model from a `BPE` model.
    pub fn from_bpe(bpe: &BPE) -> Self {
        let mut wp = Self::builder()
            .vocab(bpe.get_vocab().clone())
            .build()
            .unwrap();
        if let Some(unk) = bpe.get_unk_token() {
            wp.unk_token = unk.to_owned();
        }
        if let Some(prefix) = bpe.get_continuing_subword_prefix() {
            wp.continuing_subword_prefix = prefix.to_owned();
        }
        wp
    }
}

#[typetag::serde]
impl Model for WordPiece {
    fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<(String, Offsets)>) -> Result<Vec<Token>> {
        let mut output_tokens = Vec::with_capacity(sentence.len());

        for (index, (token, initial_offsets)) in sentence.into_iter().enumerate() {
            let char_len = token.chars().count();
            if char_len > self.max_input_chars_per_word {
                output_tokens.push(Token {
                    value: self.unk_token.clone(),
                    id: *self
                        .vocab
                        .get(&self.unk_token)
                        .ok_or(Error::MissingUnkToken)?,
                    offsets: initial_offsets,
                    word: index as u32,
                });
                continue;
            }

            let mut start = 0;
            let chars = token.char_indices().map(|(i, c)| i).chain(Some(token.len())).collect::<Vec<_>>();

            'a: while start < token.len() {
                let substr = if start == 0 {
                    Cow::Borrowed(&token[start..])
                } else {
                    format!("{}{}", self.continuing_subword_prefix, &token[start..]).into()
                };
                for end in chars.iter().rev().copied() {
                    if let Some(idx) = self.vocab.get(&substr[..end-start]) {
                        output_tokens.push(Token {
                            id: *idx,
                            value: substr.into(),
                            offsets: (initial_offsets.0 + start, initial_offsets.0 + end),
                            word: index as u32,
                        });
                        start = end;
                        continue 'a;
                    }
                }
                output_tokens.push(Token {
                    value: self.unk_token.clone(),
                    id: *self
                        .vocab
                        .get(&self.unk_token)
                        .ok_or(Error::MissingUnkToken)?,
                    offsets: initial_offsets,
                    word: index as u32,
                });
                start += 1
            }
        }

        Ok(output_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(String::as_ref)
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{}-vocab.txt", name),
            None => "vocab.txt".to_string(),
        };

        // Write vocab.txt
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let mut vocab: Vec<(&String, &u32)> = self.vocab.iter().collect();
        vocab.sort_unstable_by_key(|k| *k.1);
        vocab_file.write_all(
            &vocab
                .into_iter()
                .flat_map(|(token, _)| format!("{}\n", token).as_bytes().to_owned())
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }
}

struct PieceIter<'a, 'b> {
    tokenizer: &'a WordPiece,
    sequence: &'b str,
    pos: usize,
    n_chars: usize,
    unk_id: u32,
}

impl<'a, 'b> PieceIter<'a, 'b> {
    fn new(sequence: &'b str, tokenizer: &'a WordPiece) -> Result<Self> {
        Ok(PieceIter {
            tokenizer,
            n_chars: sequence.chars().count(),
            sequence,
            pos: 0,
            unk_id: *tokenizer.
                vocab
                .get(&tokenizer.unk_token)
                .ok_or(Error::MissingUnkToken)?
        })
    }
}

impl<'a, 'b> Iterator for PieceIter<'a, 'b> {
    type Item = (u32, String, (usize, usize));

    fn next(&mut self) -> Option<Self::Item> {
        if self.sequence.is_empty() {
            return None;
        }
        let owned;
        let sequence = if self.pos > 0 {
            owned = true;
            Cow::Owned(format!("##{}", self.sequence))
        } else {
            owned = false;
            Cow::Borrowed(self.sequence)
        };
        for (i, end) in sequence
            .char_indices()
            .map(|(i, _)| i)
            .chain(Some(self.sequence.len()))
            .rev()
            .take_while(|end| (*end > 2 && owned) || (!owned && *end > 0))
            .enumerate()
        {
            if let Some(&id) = self.tokenizer.vocab.get(&sequence[..end]) {
                let ret_seq = sequence[..end].to_string();
                if owned {
                    self.sequence = &self.sequence[end - 2..];
                } else {
                    self.sequence = &self.sequence[end..];
                }

                let old_pos = self.pos;
                self.pos += self.n_chars - i;
                self.n_chars = i;
                return Some((id, ret_seq, (old_pos, self.pos)));
            }
        }
        let next_start = self
            .sequence
            .char_indices()
            .skip(1)
            .map(|(i, _)| i)
            .next()
            .unwrap_or(self.sequence.len());
        self.sequence = &self.sequence[next_start..];
        let offset = (self.pos, self.pos + 1);
        self.pos += 1;
        self.n_chars -= 1;
        Some((self.unk_id, self.tokenizer.unk_token.clone(), offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }
}

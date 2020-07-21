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
use std::borrow::Cow;
pub use trainer::*;

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
        let mut output_tokens = vec![];

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
            output_tokens.extend(
                PieceIter::new(&token, self)?.map(|(id, val, (start, end))| {
                    Token::new(
                        id,
                        val.into(),
                        (start + initial_offsets.0, end + initial_offsets.0),
                        index as u32,
                    )
                }),
            );
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

/// Iterator over WordPieces
struct PieceIter<'a, 'b> {
    tokenizer: &'a WordPiece,
    sequence: &'b str,
    n_bytes: usize,
    unk_id: u32,
}

impl<'a, 'b> PieceIter<'a, 'b> {
    /// Creates an Iterator over the WordPieces in `sequence`.
    fn new(sequence: &'b str, tokenizer: &'a WordPiece) -> Result<Self> {
        Ok(PieceIter {
            tokenizer,
            n_bytes: sequence.len(),
            sequence,
            unk_id: *tokenizer
                .vocab
                .get(&tokenizer.unk_token)
                .ok_or(Error::MissingUnkToken)?,
        })
    }
}

impl<'a, 'b> Iterator for PieceIter<'a, 'b> {
    type Item = (u32, Cow<'b, str>, (usize, usize));

    fn next(&mut self) -> Option<Self::Item> {
        if self.sequence.is_empty() {
            return None;
        }
        let pos = self.n_bytes - self.sequence.len();
        let cont = pos > 0;
        // only allocate a string when we need to prepend the cont-prefix
        let sequence = if cont {
            Cow::Owned(format!(
                "{}{}",
                self.tokenizer.continuing_subword_prefix, self.sequence
            ))
        } else {
            Cow::Borrowed(self.sequence)
        };
        let prefix_len = self.tokenizer.continuing_subword_prefix.len();
        for end in sequence
            .char_indices()
            .map(|(end, _)| end)
            .chain(Some(self.sequence.len())) // include sequence length as end
            .rev()
        // start iterating from back to match on longest sequence
        {
            // break if only the prefix is left to prevent generating invalid tokens
            if cont && end <= prefix_len {
                break;
            }
            if let Some(&id) = self.tokenizer.vocab.get(&sequence[..end]) {
                let ret = if cont {
                    let ret = Cow::Owned(sequence[..end].to_owned());
                    self.sequence = &self.sequence[end - prefix_len..];
                    ret
                } else {
                    let ret = Cow::Borrowed(&self.sequence[..end]);
                    self.sequence = &self.sequence[end..];
                    ret
                };
                return Some((id, ret, (pos, pos + end)));
            }
        }
        let unk_char_len = self.sequence.chars().next().unwrap().len_utf8();
        self.sequence = &self.sequence[unk_char_len..];
        let offset = (pos, pos + unk_char_len);
        Some((
            self.unk_id,
            Cow::Owned(self.tokenizer.unk_token.clone()),
            offset,
        ))
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

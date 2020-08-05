use std::borrow::Cow;

use crate::models::wordpiece::{Error, WordPiece};

/// Iterator over WordPieces
pub(crate) struct PieceIter<'a, 'b> {
    tokenizer: &'a WordPiece,
    sequence: &'b str,
    unk_id: u32,
    n_chars: usize,
    pos: usize,
}

impl<'a, 'b> PieceIter<'a, 'b> {
    /// Creates an Iterator over the WordPieces in `sequence`.
    pub(crate) fn new(sequence: &'b str, tokenizer: &'a WordPiece) -> crate::Result<Self> {
        Ok(PieceIter {
            tokenizer,
            n_chars: sequence.chars().count(),
            pos: 0,
            sequence,
            unk_id: *tokenizer
                .vocab
                .get(&tokenizer.unk_token)
                .ok_or(Error::MissingUnkToken)?,
        })
    }
}

impl<'a, 'b> Iterator for PieceIter<'a, 'b> {
    type Item = Result<(u32, Cow<'b, str>, (usize, usize)), u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.sequence.is_empty() {
            return None;
        }
        let cont = self.pos > 0;
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
        for (i, end) in sequence
            .char_indices()
            .map(|(end, _)| end)
            .chain(Some(sequence.len())) // include sequence length as end
            .rev()
            .enumerate()
        // start iterating from back to match on longest sequence
        {
            // short-circuit if any part of the sequence is OOV
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
                let old_pos = self.pos;
                self.pos = self.n_chars - i;
                return Some(Ok((id, ret, (old_pos, self.pos))));
            }
        }
        self.sequence = "";
        Some(Err(self.unk_id))
    }
}

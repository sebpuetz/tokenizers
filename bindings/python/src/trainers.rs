extern crate tokenizers as tk;

use crate::models::PyModelWrapper;
use crate::tokenizer::AddedToken;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass]
pub struct Trainer {
    pub trainer: Box<dyn tk::tokenizer::Trainer<Model = PyModelWrapper>>,
}

pub enum TrainWrapper {
    BpeTrainer(tk::models::bpe::BpeTrainer),
    WordPieceTrainer(tk::models::wordpiece::WordPieceTrainer),
}

impl tk::Trainer for TrainWrapper {
    type Model = PyModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            TrainWrapper::BpeTrainer(bpe) => bpe.should_show_progress(),
            TrainWrapper::WordPieceTrainer(wp) => wp.should_show_progress(),
        }
    }

    fn train(&self, words: HashMap<String, u32>) -> tk::Result<(Self::Model, Vec<tk::AddedToken>)> {
        match self {
            TrainWrapper::BpeTrainer(bpe) => bpe.train(words).map(|(model, added)| {
                let model = PyModelWrapper {
                    inner: Arc::new(model.into()),
                };
                (model, added)
            }),
            TrainWrapper::WordPieceTrainer(wp) => wp.train(words).map(|(model, added)| {
                let model = PyModelWrapper {
                    inner: Arc::new(model.into()),
                };
                (model, added)
            }),
        }
    }

    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        match self {
            TrainWrapper::BpeTrainer(bpe) => bpe.process_tokens(words, tokens),
            TrainWrapper::WordPieceTrainer(wp) => wp.process_tokens(words, tokens),
        }
    }
}

#[pyclass(extends=Trainer)]
pub struct BpeTrainer {}
#[pymethods]
impl BpeTrainer {
    /// new(/ vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[new]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, Trainer)> {
        let mut builder = tk::models::bpe::BpeTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.cast_as::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(AddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<AddedToken>>()
                                    {
                                        token.is_special_token = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::Exception::py_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().nth(0))
                                .filter(|c| c.is_some())
                                .map(|c| c.unwrap())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                };
            }
        }
        Ok((
            BpeTrainer {},
            Trainer {
                trainer: Box::new(TrainWrapper::BpeTrainer(builder.build())),
            },
        ))
    }
}

#[pyclass(extends=Trainer)]
pub struct WordPieceTrainer {}
#[pymethods]
impl WordPieceTrainer {
    /// new(/ vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[new]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, Trainer)> {
        let mut builder = tk::models::wordpiece::WordPieceTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.cast_as::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(AddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<AddedToken>>()
                                    {
                                        token.is_special_token = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::Exception::py_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().nth(0))
                                .filter(|c| c.is_some())
                                .map(|c| c.unwrap())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                };
            }
        }

        Ok((
            WordPieceTrainer {},
            Trainer {
                trainer: Box::new(TrainWrapper::WordPieceTrainer(builder.build())),
            },
        ))
    }
}

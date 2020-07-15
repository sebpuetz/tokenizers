use super::{added_vocabulary::AddedTokenWithId, Tokenizer};
use crate::Model;
use serde::{
    self,
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::marker::PhantomData;

static SERIALIZATION_VERSION: &str = "1.0";

impl<T> Serialize for Tokenizer<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 9)?;

        // Start by adding the current version
        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;

        // Params
        tokenizer.serialize_field("truncation", &self.truncation)?;
        tokenizer.serialize_field("padding", &self.padding)?;

        // Added tokens
        tokenizer.serialize_field("added_tokens", &self.added_vocabulary)?;

        // Then add our parts
        tokenizer.serialize_field("normalizer", &self.normalizer)?;
        tokenizer.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        tokenizer.serialize_field("post_processor", &self.post_processor)?;
        tokenizer.serialize_field("decoder", &self.decoder)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de, T> Deserialize<'de> for Tokenizer<T>
where
    T: Deserialize<'de> + Default + Model,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "Tokenizer",
            &[
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            ],
            TokenizerVisitor(PhantomData),
        )
    }
}

struct TokenizerVisitor<T>(PhantomData<T>);

impl<'de, T> Visitor<'de> for TokenizerVisitor<T>
where
    T: Deserialize<'de> + Default + Model,
{
    type Value = Tokenizer<T>;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Tokenizer")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut tokenizer = Tokenizer::new(T::default());
        let mut tokens: Vec<AddedTokenWithId> = vec![];
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "version" => {
                    let v: String = map.next_value()?;
                    if &v != "1.0" {
                        return Err(Error::custom(format!("Unknown tokenizer version '{}'", v)));
                    }
                }
                "truncation" => {
                    tokenizer.with_truncation(map.next_value()?);
                }
                "padding" => {
                    tokenizer.with_padding(map.next_value()?);
                }
                "added_tokens" => {
                    tokens = map.next_value()?;
                }
                "normalizer" => {
                    if let Some(normalizer) = map.next_value()? {
                        tokenizer.with_normalizer(normalizer);
                    }
                }
                "pre_tokenizer" => {
                    if let Some(pre_tok) = map.next_value()? {
                        tokenizer.with_pre_tokenizer(pre_tok);
                    }
                }
                "model" => {
                    tokenizer.with_model(map.next_value()?);
                }
                "decoder" => {
                    if let Some(decoder) = map.next_value()? {
                        tokenizer.with_decoder(decoder);
                    }
                }
                "post_processor" => {
                    if let Some(processor) = map.next_value()? {
                        tokenizer.with_post_processor(processor);
                    }
                }
                _ => {}
            };
        }

        // We take care of deserializing the added_tokens (instead of `AddedVocabulary` directly
        // because it let us check that associated IDs are still good, and warn the user otherwise
        for token in tokens {
            let tk = token.token.content.clone();
            if token.special {
                tokenizer.add_special_tokens(&[token.token]);
            } else {
                tokenizer.add_tokens(&[token.token]);
            }
            // Warn the user if the id is different than expected
            let received_id = tokenizer.token_to_id(&tk);
            if received_id != Some(token.id) {
                println!(
                    "Warning: Token '{}' was expected to have ID '{}' but was given ID '{}'",
                    tk,
                    token.id,
                    if let Some(rid) = received_id {
                        rid.to_string()
                    } else {
                        "None".to_string()
                    }
                );
            }
        }

        Ok(tokenizer)
    }
}

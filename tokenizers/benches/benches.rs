#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::{AddedToken, EncodeInput, Tokenizer, Trainer};
use tokenizers::{decoders, Model};

static BATCH_SIZE: usize = 1_000;

fn create_gpt2_tokenizer(bpe: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(bpe));
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel::default()));
    tokenizer.with_decoder(Box::new(ByteLevel::default()));
    tokenizer.add_tokens(&[AddedToken::from("ing", false).single_word(false)]);
    tokenizer.add_special_tokens(&[AddedToken::from("[ENT]", true).single_word(true)]);
    tokenizer
}

fn iter_bench_encode(iters: u64, tokenizer: &Tokenizer, lines: &[EncodeInput]) -> Duration {
    let mut duration = Duration::new(0, 0);
    let mut line_index: usize = 0;
    for _i in 0..iters {
        if line_index >= lines.len() {
            line_index = 0;
        }
        let input = lines[line_index].clone();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode(input, false));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn iter_bench_encode_batch(
    iters: u64,
    tokenizer: &Tokenizer,
    batches: &[Vec<EncodeInput>],
) -> Duration {
    let mut duration = Duration::new(0, 0);
    let mut batch_index: usize = 0;
    for _i in 0..iters {
        if batch_index >= batches.len() {
            batch_index = 0;
        }
        let batch = batches[batch_index].clone();
        let start = Instant::now();
        let _ = black_box(tokenizer.encode_batch(batch, false));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_gpt2(c: &mut Criterion) {
    let bpe = BPE::from_files("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let line: EncodeInput = line.unwrap().into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    c.bench_function("BPE GPT2 encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });

    let bpe = BPE::from_files("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .cache_capacity(0)
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);

    c.bench_function("BPE GPT2 encode, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("BPE GPT2 encode batch, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

#[allow(clippy::borrowed_box)]
fn iter_bench_train(
    iters: u64,
    tokenizer: &mut Tokenizer,
    trainer: &Box<dyn Trainer>,
    files: Vec<String>,
) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let _ = black_box(tokenizer.train(&trainer, files.clone()));
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_train(c: &mut Criterion) {
    let mut tokenizer = Tokenizer::new(Box::new(BPE::default()));
    tokenizer.with_pre_tokenizer(Box::new(Whitespace));

    let trainer: Box<dyn Trainer> =
        Box::new(BpeTrainerBuilder::default().show_progress(false).build());
    c.bench_function("BPE Train vocabulary (small)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &trainer,
                vec!["data/small.txt".to_string()],
            )
        })
    });

    c.bench_function("BPE Train vocabulary (big)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &trainer,
                vec!["data/big.txt".to_string()],
            )
        })
    });
}

fn create_bert_tokenizer(wp: WordPiece) -> Tokenizer {
    let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
    let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
    let mut tokenizer = Tokenizer::new(Box::new(wp));
    tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
    tokenizer.with_normalizer(Box::new(BertNormalizer::default()));
    tokenizer.with_decoder(Box::new(decoders::wordpiece::WordPiece::default()));
    tokenizer.with_post_processor(Box::new(BertProcessing::new(
        ("[SEP]".to_string(), sep_id),
        ("[CLS]".to_string(), cls_id),
    )));
    tokenizer
}

pub fn bench_bert(c: &mut Criterion) {
    let wp = WordPiece::from_files("data/bert-base-uncased-vocab.txt")
        .build()
        .unwrap();
    let tokenizer = create_bert_tokenizer(wp);
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let line: EncodeInput = line.unwrap().into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    c.bench_function("WordPiece BERT encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("WordPiece BERT encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_gpt2
}
criterion_group! {
    name = benches_train;
    config = Criterion::default().sample_size(10);
    targets = bench_train
}

criterion_group! {
    name = bert_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_bert
}

criterion_main!(benches, benches_train, bert_benches);

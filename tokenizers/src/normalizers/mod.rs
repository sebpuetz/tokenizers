pub mod bert;
pub mod strip;
pub mod unicode;
pub mod utils;

use serde::{Deserialize, Serialize};

use crate::{Normalizer, NormalizedString};
use crate::normalizers::bert::BertNormalizer;
use crate::normalizers::strip::Strip;
use crate::normalizers::unicode::{NFC, NFD, NFKC, NFKD};

/// Wrapper for known Normalizers.
#[derive(Deserialize, Serialize)]
pub enum NormalizerWrapper {
    BertNormalizer(BertNormalizer),
    StripNormalizer(Strip),
    NFC(NFC),
    NFD(NFD),
    NFKC(NFKC),
    NFKD(NFKD)
}

#[typetag::serde]
impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> crate::Result<()> {
        match self {
            NormalizerWrapper::BertNormalizer(bn) => bn.normalize(normalized),
            NormalizerWrapper::StripNormalizer(sn) => sn.normalize(normalized),
            NormalizerWrapper::NFC(nfc) => nfc.normalize(normalized),
            NormalizerWrapper::NFD(nfd) => nfd.normalize(normalized),
            NormalizerWrapper::NFKC(nfkc) => nfkc.normalize(normalized),
            NormalizerWrapper::NFKD(nfkd) => nfkd.normalize(normalized),
        }
    }
}

impl_enum_from!(BertNormalizer, NormalizerWrapper, BertNormalizer);
impl_enum_from!(NFKD, NormalizerWrapper, NFKD);
impl_enum_from!(NFKC, NormalizerWrapper, NFKC);
impl_enum_from!(NFC, NormalizerWrapper, NFC);
impl_enum_from!(NFD, NormalizerWrapper, NFD);
impl_enum_from!(Strip, NormalizerWrapper, StripNormalizer);
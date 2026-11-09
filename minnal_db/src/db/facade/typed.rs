//! The typed methods' rkyv layer: what a key or value type must support
//! ([`RkyvEncode`], [`RkyvDecode`]) and the conversions to and from bytes.

use crate::db::error::{KVError, Result};

// ── rkyv trait alias ───────────────────────────────────────────────────
//
// Bounds required for a type to be used with the `_typed` methods.
// Users derive these via `#[derive(Archive, RkyvSerialize, RkyvDeserialize)]`.

use rkyv::api::high::{HighSerializer, HighValidator};
use rkyv::rancor::Error as RkyvError;
use rkyv::rancor::Strategy;
use rkyv::ser::allocator::ArenaHandle;

/// A type the typed methods (`put_typed` and the rest) can write as a key or a
/// value: rkyv can serialize it. Every type deriving rkyv's `Serialize` is one.
pub trait RkyvEncode: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>> {}

impl<T> RkyvEncode for T where T: for<'a> rkyv::Serialize<HighSerializer<rkyv::util::AlignedVec, ArenaHandle<'a>, RkyvError>> {}

/// A type the typed methods can read back: rkyv can check its archived bytes and
/// deserialize them. Every type deriving rkyv's `Archive` and `Deserialize`, with
/// `bytecheck`, is one.
pub trait RkyvDecode:
    Sized
    + rkyv::Archive<
        Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<Self, Strategy<rkyv::de::Pool, RkyvError>>,
    >
{
}

impl<T> RkyvDecode for T
where
    T: rkyv::Archive,
    T::Archived: for<'a> bytecheck::CheckBytes<HighValidator<'a, RkyvError>> + rkyv::Deserialize<T, Strategy<rkyv::de::Pool, RkyvError>>,
{
}

/// Serialize a value to bytes via rkyv.
pub fn rkyv_serialize<T>(value: &T) -> Result<Vec<u8>>
where
    T: RkyvEncode,
{
    rkyv::to_bytes::<RkyvError>(value)
        .map(|buf| buf.to_vec())
        .map_err(|e| KVError::Serialization(format!("rkyv serialize: {}", e)))
}

/// Deserialize bytes into a value via rkyv, **validating them first**.
///
/// The bytes are whatever is stored under the key, and nothing ties them to `T`:
/// a caller can `put` raw bytes and then `get_typed` the key, `put_typed` one type
/// and `get_typed` another, or run a typed range scan over keys written any way at
/// all. This used `access_unchecked` on the assumption that the bytes always came
/// from `rkyv_serialize` of the same `T`, which made a safe public function able
/// to follow archived relative pointers out of bounds — undefined behaviour. Every
/// typed entry point already requires `T::Archived: CheckBytes`, so validating
/// costs no API change; mismatched bytes are now a `Serialization` error.
pub fn rkyv_deserialize<T>(bytes: &[u8]) -> Result<T>
where
    T: RkyvDecode,
{
    rkyv::from_bytes::<T, RkyvError>(bytes).map_err(|e| KVError::Serialization(format!("rkyv deserialize: {:?}", e)))
}

// ── Shared helpers ─────────────────────────────────────────────────────

/// Deserialize a list of raw key-value byte pairs into typed pairs.
pub fn deserialize_pairs<K, V>(pairs: Vec<(Vec<u8>, Vec<u8>)>) -> Result<Vec<(K, V)>>
where
    K: RkyvDecode,
    V: RkyvDecode,
{
    pairs
        .into_iter()
        .map(|(kb, vb)| {
            let key = rkyv_deserialize::<K>(&kb)?;
            let val = rkyv_deserialize::<V>(&vb)?;
            Ok((key, val))
        })
        .collect()
}

/// Deserialize a list of raw key bytes into typed keys.
pub fn deserialize_keys<K>(keys: Vec<Vec<u8>>) -> Result<Vec<K>>
where
    K: RkyvDecode,
{
    keys.into_iter().map(|kb| rkyv_deserialize::<K>(&kb)).collect()
}

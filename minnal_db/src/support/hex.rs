//! Hex encoding/decoding helpers shared across every layer.
//!
//! Lives in `support` rather than `doc_store` because the base engine needs it
//! too: the field-index gap record stores raw keys, which are arbitrary bytes
//! and not necessarily UTF-8, so they are hex-encoded to keep the record
//! human-readable.

/// Encode a byte slice as a lowercase hex string.
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a lowercase (or uppercase) hex string to bytes.
///
/// Returns `None` if the string has an odd length or contains anything but
/// `0-9`, `a-f`, `A-F`.
///
/// Works on the bytes, never on `str` slices: callers pass request input, and
/// slicing a `&str` at a byte offset inside a multi-byte character panics — with
/// the release profile's `panic = "abort"`, one request (`?cursor=aéb`) took the
/// whole server down. It also rejects the sign `u8::from_str_radix` accepts, so
/// `+f+f` is an error rather than `[0x0f, 0x0f]`.
pub fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    fn nibble(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    }
    let bytes = hex.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    bytes
        .chunks_exact(2)
        .map(|pair| Some((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Request input reaches this; it must reject, never panic. Byte-slicing a
    /// `str` inside a multi-byte character panicked (and the server aborts on panic).
    #[test]
    fn non_ascii_and_signed_input_is_rejected_not_a_panic() {
        assert_eq!(hex_to_bytes("aéb"), None); // 4 bytes: even length, but 'é' is two
        assert_eq!(hex_to_bytes("éé"), None);
        assert_eq!(hex_to_bytes("+f+f"), None);
        assert_eq!(hex_to_bytes("-1"), None);
        assert_eq!(hex_to_bytes("0aFf"), Some(vec![0x0a, 0xff]));
        assert_eq!(hex_to_bytes(""), Some(vec![]));
    }

    #[test]
    fn roundtrip() {
        let bytes = [0x00u8, 0xde, 0xad, 0xbe, 0xef, 0xff];
        assert_eq!(hex_to_bytes(&bytes_to_hex(&bytes)).unwrap(), bytes);
    }

    #[test]
    fn odd_length_returns_none() {
        assert!(hex_to_bytes("abc").is_none());
    }

    #[test]
    fn invalid_char_returns_none() {
        assert!(hex_to_bytes("zz").is_none());
    }

    #[test]
    fn empty_roundtrip() {
        assert_eq!(hex_to_bytes(&bytes_to_hex(&[])).unwrap(), Vec::<u8>::new());
    }
}

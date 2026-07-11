//! Hex encoding/decoding helpers shared across the doc-store and API layers.

/// Encode a byte slice as a lowercase hex string.
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a lowercase (or uppercase) hex string to bytes.
///
/// Returns `None` if the string has an odd length or contains a non-hex
/// character.
pub fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

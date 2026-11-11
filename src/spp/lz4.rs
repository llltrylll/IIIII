// Adapted from SubCraft (https://github.com/Aalzard/subcraft), `crates/mesh/src/spp/lz4.rs`.
// Copyright (c) Aalzard and the SubCraft contributors. Licensed under the MIT License
// (see src/spp/LICENSE-SUBCRAFT); used here, with thanks, as part of Material Painter
// (GPL-3.0-or-later).

//! LZ4 block decompression (the raw block format, no frame), used by Painter's texture cache.

#[derive(Debug, thiserror::Error)]
#[error("corrupt LZ4 block")]
pub struct Lz4Error;

/// Decompress `src` into exactly `size` bytes.
pub fn decompress(src: &[u8], size: usize) -> Result<Vec<u8>, Lz4Error> {
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(size).map_err(|_| Lz4Error)?;
    let mut i = 0;
    while i < src.len() {
        let token = src[i];
        i += 1;
        let mut literals = (token >> 4) as usize;
        if literals == 15 {
            loop {
                let b = *src.get(i).ok_or(Lz4Error)?;
                i += 1;
                literals = literals.checked_add(b as usize).ok_or(Lz4Error)?;
                if b != 255 {
                    break;
                }
            }
        }
        let literal_end = i.checked_add(literals).ok_or(Lz4Error)?;
        let output_len = out.len().checked_add(literals).ok_or(Lz4Error)?;
        if output_len > size {
            return Err(Lz4Error);
        }
        out.extend_from_slice(src.get(i..literal_end).ok_or(Lz4Error)?);
        i = literal_end;
        if i >= src.len() {
            break;
        }
        let offset_end = i.checked_add(2).ok_or(Lz4Error)?;
        let offset_bytes = src.get(i..offset_end).ok_or(Lz4Error)?;
        let offset = offset_bytes[0] as usize | (offset_bytes[1] as usize) << 8;
        i = offset_end;
        let mut len = (token & 15) as usize;
        if len == 15 {
            loop {
                let b = *src.get(i).ok_or(Lz4Error)?;
                i += 1;
                len = len.checked_add(b as usize).ok_or(Lz4Error)?;
                if b != 255 {
                    break;
                }
            }
        }
        len = len.checked_add(4).ok_or(Lz4Error)?;
        let output_len = out.len().checked_add(len).ok_or(Lz4Error)?;
        if offset == 0 || offset > out.len() || output_len > size {
            return Err(Lz4Error);
        }
        let start = out.len() - offset;
        for k in 0..len {
            let index = start.checked_add(k).ok_or(Lz4Error)?;
            let b = *out.get(index).ok_or(Lz4Error)?;
            out.push(b);
        }
    }
    if out.len() != size {
        return Err(Lz4Error);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_and_overlapping_match() {
        // "abc" then a match of length 7 at offset 3 -> "abcabcabca" (10 bytes).
        let block = [0x33, b'a', b'b', b'c', 0x03, 0x00];
        assert_eq!(decompress(&block, 10).unwrap(), b"abcabcabca");
    }

    #[test]
    fn long_literal_run() {
        let mut block = vec![0xf0, 5];
        block.extend((0..20).map(|i| i as u8));
        assert_eq!(decompress(&block, 20).unwrap(), (0..20).map(|i| i as u8).collect::<Vec<_>>());
    }

    #[test]
    fn rejects_wrong_size_and_bad_offset() {
        let block = [0x33, b'a', b'b', b'c', 0x03, 0x00];
        assert!(decompress(&block, 11).is_err());
        let bad = [0x10, b'a', 0x05, 0x00];
        assert!(decompress(&bad, 5).is_err());
    }
}

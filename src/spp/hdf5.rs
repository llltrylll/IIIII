// Adapted from SubCraft (https://github.com/Aalzard/subcraft), `crates/mesh/src/spp/hdf5.rs`.
// Copyright (c) Aalzard and the SubCraft contributors. Licensed under the MIT License
// (see src/spp/LICENSE-SUBCRAFT); used here, with thanks, as part of Material Painter
// (GPL-3.0-or-later).

//! Minimal read-only HDF5 reader, just enough for Substance 3D Painter `.spp` projects.
//!
//! Supported: superblock version 0, version-1 object headers, old-style groups (symbol tables,
//! local heaps, version-1 B-trees) and byte datasets stored as compact, contiguous or chunked
//! (unfiltered) data. Anything else is reported as an error instead of being guessed at.

use std::io::{Read, Seek, SeekFrom};

const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];
const UNDEFINED: u64 = u64::MAX;
/// The import only needs ordinary texture tiles and model data; reject
/// unreasonable metadata sizes before allocating from an untrusted HDF5 file.
const MAX_DATASET_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_HDF5_METADATA_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum Hdf5Error {
    #[error("not an HDF5 file")]
    NotHdf5,
    #[error("unsupported HDF5 feature: {0}")]
    Unsupported(String),
    #[error("corrupt HDF5 file: {0}")]
    Corrupt(String),
    #[error("no dataset named '{0}'")]
    NotFound(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

type Result<T> = std::result::Result<T, Hdf5Error>;

fn corrupt<T>(what: &str) -> Result<T> {
    Err(Hdf5Error::Corrupt(what.into()))
}

fn u16_at(b: &[u8], o: usize) -> Result<u16> {
    let end = o.checked_add(2).ok_or_else(|| Hdf5Error::Corrupt("short read".into()))?;
    b.get(o..end).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| Hdf5Error::Corrupt("short read".into()))
}

fn u32_at(b: &[u8], o: usize) -> Result<u32> {
    let end = o.checked_add(4).ok_or_else(|| Hdf5Error::Corrupt("short read".into()))?;
    b.get(o..end).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| Hdf5Error::Corrupt("short read".into()))
}

fn u64_at(b: &[u8], o: usize) -> Result<u64> {
    let end = o.checked_add(8).ok_or_else(|| Hdf5Error::Corrupt("short read".into()))?;
    b.get(o..end)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or_else(|| Hdf5Error::Corrupt("short read".into()))
}

fn dataset_byte_len(dims: &[u64], element_size: u64) -> Result<u64> {
    if element_size == 0 {
        return corrupt("zero-sized datatype");
    }
    let elements = dims.iter().try_fold(1u64, |total, &dim| {
        total.checked_mul(dim).ok_or_else(|| Hdf5Error::Corrupt("dataset dimensions overflow".into()))
    })?;
    elements.checked_mul(element_size).ok_or_else(|| Hdf5Error::Corrupt("dataset byte size overflow".into()))
}

/// One dataset's location, found from its object header.
#[derive(Clone, Debug)]
pub struct DatasetInfo {
    /// Total size in bytes.
    pub len: u64,
    layout: Layout,
}

#[derive(Clone, Debug)]
enum Layout {
    Compact(Vec<u8>),
    Contiguous { address: u64 },
    Chunked { btree: u64, dims: Vec<u32> },
}

pub struct Hdf5File<R: Read + Seek> {
    r: R,
    root_btree: u64,
    root_heap: u64,
}

impl<R: Read + Seek> Hdf5File<R> {
    pub fn open(mut r: R) -> Result<Self> {
        let sb = read_at(&mut r, 0, 96)?;
        if sb[..8] != SIGNATURE {
            return Err(Hdf5Error::NotHdf5);
        }
        if sb[8] != 0 {
            return Err(Hdf5Error::Unsupported(format!("superblock version {}", sb[8])));
        }
        if sb[13] != 8 || sb[14] != 8 {
            return Err(Hdf5Error::Unsupported("offset/length sizes other than 8 bytes".into()));
        }
        // Root group symbol table entry follows the four 8-byte addresses at offset 24.
        let entry = 56;
        let cache_type = u32_at(&sb, entry + 16)?;
        if cache_type != 1 {
            return Err(Hdf5Error::Unsupported("root group without a symbol table cache".into()));
        }
        Ok(Self { root_btree: u64_at(&sb, entry + 24)?, root_heap: u64_at(&sb, entry + 32)?, r })
    }

    /// Find a dataset by its slash-separated path (for example `paint/scene.assbin`).
    pub fn dataset(&mut self, path: &str) -> Result<DatasetInfo> {
        let (mut btree, mut heap) = (self.root_btree, self.root_heap);
        let parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
        for (i, part) in parts.iter().enumerate() {
            let addr = self.lookup(btree, heap, part)?.ok_or_else(|| Hdf5Error::NotFound(path.into()))?;
            if i + 1 == parts.len() {
                return self.read_dataset_header(addr);
            }
            let msgs = self.object_messages(addr)?;
            let st = msgs.iter().find(|m| m.0 == 0x11).ok_or_else(|| Hdf5Error::NotFound(path.into()))?;
            btree = u64_at(&st.1, 0)?;
            heap = u64_at(&st.1, 8)?;
        }
        Err(Hdf5Error::NotFound(path.into()))
    }

    /// Names of the direct children of the group at `path` (empty path = the root group).
    pub fn list(&mut self, path: &str) -> Result<Vec<String>> {
        let (mut btree, mut heap) = (self.root_btree, self.root_heap);
        for part in path.split('/').filter(|p| !p.is_empty()) {
            let addr = self.lookup(btree, heap, part)?.ok_or_else(|| Hdf5Error::NotFound(path.into()))?;
            let msgs = self.object_messages(addr)?;
            let st = msgs.iter().find(|m| m.0 == 0x11).ok_or_else(|| Hdf5Error::NotFound(path.into()))?;
            btree = u64_at(&st.1, 0)?;
            heap = u64_at(&st.1, 8)?;
        }
        let mut names = Vec::new();
        self.walk_group(btree, heap, &mut |name, _| {
            names.push(name.to_string());
            false
        })?;
        Ok(names)
    }

    /// Read the whole dataset.
    pub fn read(&mut self, info: &DatasetInfo) -> Result<Vec<u8>> {
        if info.len > MAX_DATASET_BYTES {
            return Err(Hdf5Error::Unsupported("dataset exceeds the safe import size limit".into()));
        }
        let len = usize::try_from(info.len).map_err(|_| Hdf5Error::Corrupt("dataset size".into()))?;
        match &info.layout {
            Layout::Compact(data) => {
                if data.len() != len {
                    return corrupt("compact dataset size");
                }
                Ok(data.clone())
            }
            Layout::Contiguous { address } => {
                if *address == UNDEFINED {
                    return Ok(vec![0; len]);
                }
                read_at(&mut self.r, *address, len)
            }
            Layout::Chunked { btree, dims } => {
                let mut out = vec![0u8; len];
                if *btree != UNDEFINED {
                    let rank = dims.len();
                    self.read_chunks(*btree, rank, &mut out)?;
                }
                Ok(out)
            }
        }
    }

    fn read_chunks(&mut self, node: u64, rank: usize, out: &mut [u8]) -> Result<()> {
        let head = read_at(&mut self.r, node, 24)?;
        if &head[..4] != b"TREE" || head[4] != 1 {
            return corrupt("chunk B-tree node");
        }
        let level = head[5];
        let used = u16_at(&head, 6)? as usize;
        if rank != 1 {
            return Err(Hdf5Error::Unsupported("chunked datasets must be one-dimensional byte arrays".into()));
        }
        let key_size = 8 + 8 * rank;
        let body_len = used.checked_mul(key_size + 8).and_then(|n| n.checked_add(key_size))
            .ok_or_else(|| Hdf5Error::Corrupt("chunk B-tree size overflow".into()))?;
        let body_at = node.checked_add(24).ok_or_else(|| Hdf5Error::Corrupt("chunk B-tree address overflow".into()))?;
        let body = read_at(&mut self.r, body_at, body_len)?;
        for i in 0..used {
            let key = i * (key_size + 8);
            let size = u32_at(&body, key)? as usize;
            let mask = u32_at(&body, key + 4)?;
            let child = u64_at(&body, key + key_size)?;
            if level > 0 {
                self.read_chunks(child, rank, out)?;
                continue;
            }
            if mask != 0 {
                return Err(Hdf5Error::Unsupported("filtered chunks".into()));
            }
            // Offsets are per dimension; datasets here are one-dimensional bytes.
            let offset = u64_at(&body, key + 8)? as usize;
            if offset >= out.len() {
                continue;
            }
            let take = size.min(out.len() - offset);
            let data = read_at(&mut self.r, child, take)?;
            out[offset..offset + take].copy_from_slice(&data);
        }
        Ok(())
    }

    fn lookup(&mut self, btree: u64, heap: u64, name: &str) -> Result<Option<u64>> {
        let mut found = None;
        self.walk_group(btree, heap, &mut |n, addr| {
            if n == name {
                found = Some(addr);
                true
            } else {
                false
            }
        })?;
        Ok(found)
    }

    /// Visit every (name, object header address) of a group; the callback returns true to stop.
    fn walk_group(&mut self, btree: u64, heap: u64, f: &mut dyn FnMut(&str, u64) -> bool) -> Result<()> {
        let h = read_at(&mut self.r, heap, 32)?;
        if &h[..4] != b"HEAP" {
            return corrupt("local heap");
        }
        let heap_data = u64_at(&h, 24)?;
        let heap_size = usize::try_from(u64_at(&h, 8)?).map_err(|_| Hdf5Error::Corrupt("local heap size".into()))?;
        if heap_size as u64 > MAX_HDF5_METADATA_BYTES {
            return Err(Hdf5Error::Unsupported("local heap exceeds the metadata size limit".into()));
        }
        let heap_bytes = read_at(&mut self.r, heap_data, heap_size)?;
        self.walk_btree(btree, &heap_bytes, f).map(|_| ())
    }

    fn walk_btree(&mut self, node: u64, heap: &[u8], f: &mut dyn FnMut(&str, u64) -> bool) -> Result<bool> {
        let head = read_at(&mut self.r, node, 24)?;
        if &head[..4] != b"TREE" || head[4] != 0 {
            return corrupt("group B-tree node");
        }
        let level = head[5];
        let used = u16_at(&head, 6)? as usize;
        let body_len = used.checked_mul(16).and_then(|n| n.checked_add(8))
            .ok_or_else(|| Hdf5Error::Corrupt("group B-tree size overflow".into()))?;
        let body_at = node.checked_add(24).ok_or_else(|| Hdf5Error::Corrupt("group B-tree address overflow".into()))?;
        let body = read_at(&mut self.r, body_at, body_len)?;
        for i in 0..used {
            let child = u64_at(&body, 8 + i * 16)?;
            if level > 0 {
                if self.walk_btree(child, heap, f)? {
                    return Ok(true);
                }
                continue;
            }
            let snod = read_at(&mut self.r, child, 8)?;
            if &snod[..4] != b"SNOD" {
                return corrupt("symbol table node");
            }
            let count = u16_at(&snod, 6)? as usize;
            let entries_len = count.checked_mul(40).ok_or_else(|| Hdf5Error::Corrupt("symbol table size overflow".into()))?;
            let entries_at = child.checked_add(8).ok_or_else(|| Hdf5Error::Corrupt("symbol table address overflow".into()))?;
            let entries = read_at(&mut self.r, entries_at, entries_len)?;
            for e in 0..count {
                let name_off = u64_at(&entries, e * 40)? as usize;
                let addr = u64_at(&entries, e * 40 + 8)?;
                let end = heap.get(name_off..).and_then(|s| s.iter().position(|&c| c == 0)).ok_or_else(|| Hdf5Error::Corrupt("link name".into()))?;
                let name = String::from_utf8_lossy(&heap[name_off..name_off + end]);
                if f(&name, addr) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// All header messages of a version-1 object header as (type, data).
    fn object_messages(&mut self, addr: u64) -> Result<Vec<(u16, Vec<u8>)>> {
        let head = read_at(&mut self.r, addr, 16)?;
        if head[0] != 1 {
            return Err(Hdf5Error::Unsupported(format!("object header version {}", head[0])));
        }
        let mut remaining = u16_at(&head, 2)? as usize;
        let first_block = addr.checked_add(16).ok_or_else(|| Hdf5Error::Corrupt("object header address overflow".into()))?;
        let first_len = u32_at(&head, 8)? as u64;
        if first_len > MAX_HDF5_METADATA_BYTES {
            return Err(Hdf5Error::Unsupported("object header exceeds the metadata size limit".into()));
        }
        let mut blocks = vec![(first_block, first_len)];
        let mut out = Vec::new();
        while let Some((start, len)) = blocks.pop() {
            if len > MAX_HDF5_METADATA_BYTES {
                return Err(Hdf5Error::Unsupported("object header continuation exceeds the metadata size limit".into()));
            }
            let len = usize::try_from(len).map_err(|_| Hdf5Error::Corrupt("object header size".into()))?;
            let data = read_at(&mut self.r, start, len)?;
            let mut p = 0;
            while p + 8 <= data.len() && remaining > 0 {
                let kind = u16_at(&data, p)?;
                let size = u16_at(&data, p + 2)? as usize;
                let body = data.get(p + 8..p + 8 + size).ok_or_else(|| Hdf5Error::Corrupt("header message".into()))?.to_vec();
                p += 8 + size;
                remaining -= 1;
                if kind == 0x10 {
                    let next_addr = u64_at(&body, 0)?;
                    let next_len = u64_at(&body, 8)?;
                    if next_len > MAX_HDF5_METADATA_BYTES {
                        return Err(Hdf5Error::Unsupported("object header continuation exceeds the metadata size limit".into()));
                    }
                    blocks.push((next_addr, next_len));
                } else {
                    out.push((kind, body));
                }
            }
        }
        Ok(out)
    }

    fn read_dataset_header(&mut self, addr: u64) -> Result<DatasetInfo> {
        let msgs = self.object_messages(addr)?;
        let find = |t: u16| msgs.iter().find(|m| m.0 == t).map(|m| m.1.as_slice());
        if find(0x0b).is_some() {
            return Err(Hdf5Error::Unsupported("filtered (compressed) dataset".into()));
        }
        let space = find(0x01).ok_or_else(|| Hdf5Error::Corrupt("dataset without dataspace".into()))?;
        let dims = dataspace_dims(space)?;
        let ty = find(0x03).ok_or_else(|| Hdf5Error::Corrupt("dataset without datatype".into()))?;
        let element = u32_at(ty, 4)? as u64;
        let len = dataset_byte_len(&dims, element)?;
        let lay = find(0x08).ok_or_else(|| Hdf5Error::Corrupt("dataset without layout".into()))?;
        let layout_version = *lay.first().ok_or_else(|| Hdf5Error::Corrupt("empty layout message".into()))?;
        if layout_version != 3 {
            return Err(Hdf5Error::Unsupported(format!("data layout version {layout_version}")));
        }
        let layout_class = *lay.get(1).ok_or_else(|| Hdf5Error::Corrupt("short layout message".into()))?;
        let layout = match layout_class {
            0 => {
                let size = u16_at(lay, 2)? as usize;
                let end = 4usize.checked_add(size).ok_or_else(|| Hdf5Error::Corrupt("compact dataset size overflow".into()))?;
                Layout::Compact(lay.get(4..end).ok_or_else(|| Hdf5Error::Corrupt("compact data".into()))?.to_vec())
            }
            1 => Layout::Contiguous { address: u64_at(lay, 2)? },
            2 => {
                let rank = *lay.get(2).ok_or_else(|| Hdf5Error::Corrupt("short chunked layout message".into()))? as usize;
                if rank != 1 {
                    return Err(Hdf5Error::Unsupported("chunked datasets must be one-dimensional byte arrays".into()));
                }
                let btree = u64_at(lay, 3)?;
                let dimensions_end = 11usize.checked_add(rank.checked_mul(4).ok_or_else(|| Hdf5Error::Corrupt("chunk dimensions overflow".into()))?)
                    .ok_or_else(|| Hdf5Error::Corrupt("chunk layout size overflow".into()))?;
                if dimensions_end > lay.len() {
                    return corrupt("short chunked layout dimensions");
                }
                let mut d = Vec::with_capacity(rank);
                for i in 0..rank {
                    let offset = 11usize.checked_add(i.checked_mul(4).ok_or_else(|| Hdf5Error::Corrupt("chunk dimension offset overflow".into()))?)
                        .ok_or_else(|| Hdf5Error::Corrupt("chunk dimension offset overflow".into()))?;
                    d.push(u32_at(lay, offset)?);
                }
                Layout::Chunked { btree, dims: d }
            }
            class => return Err(Hdf5Error::Unsupported(format!("layout class {class}"))),
        };
        Ok(DatasetInfo { len, layout })
    }
}

fn dataspace_dims(m: &[u8]) -> Result<Vec<u64>> {
    let version = *m.first().ok_or_else(|| Hdf5Error::Corrupt("empty dataspace message".into()))?;
    let rank = *m.get(1).ok_or_else(|| Hdf5Error::Corrupt("short dataspace message".into()))? as usize;
    let start = match version {
        1 => 8usize,
        2 => 4usize,
        v => return Err(Hdf5Error::Unsupported(format!("dataspace version {v}"))),
    };
    if rank > 8 {
        return Err(Hdf5Error::Unsupported("dataspace rank exceeds the supported limit".into()));
    }
    let end = start.checked_add(rank.checked_mul(8).ok_or_else(|| Hdf5Error::Corrupt("dataspace dimensions overflow".into()))?)
        .ok_or_else(|| Hdf5Error::Corrupt("dataspace dimensions overflow".into()))?;
    if end > m.len() {
        return corrupt("short dataspace dimensions");
    }
    (0..rank).map(|i| {
        let offset = start + i * 8;
        u64_at(m, offset)
    }).collect()
}

fn read_at<R: Read + Seek>(r: &mut R, offset: u64, len: usize) -> Result<Vec<u8>> {
    r.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}


#[cfg(test)]
mod parser_tests {
    use super::*;

    #[test]
    fn dataset_length_rejects_dimension_and_element_overflow() {
        assert!(matches!(dataset_byte_len(&[u64::MAX, 2], 1), Err(Hdf5Error::Corrupt(_))));
        assert!(matches!(dataset_byte_len(&[u64::MAX], 2), Err(Hdf5Error::Corrupt(_))));
        assert!(matches!(dataset_byte_len(&[1], 0), Err(Hdf5Error::Corrupt(_))));
    }

    #[test]
    fn dataspace_parser_rejects_empty_and_truncated_messages() {
        assert!(matches!(dataspace_dims(&[]), Err(Hdf5Error::Corrupt(_))));
        assert!(matches!(dataspace_dims(&[1]), Err(Hdf5Error::Corrupt(_))));
        assert!(matches!(dataspace_dims(&[1, 1, 0, 0]), Err(Hdf5Error::Corrupt(_))));
    }

    #[test]
    fn dataspace_parser_rejects_unreasonable_rank() {
        assert!(matches!(dataspace_dims(&[1, 9]), Err(Hdf5Error::Unsupported(_))));
    }

    #[test]
    fn dataspace_parser_reads_one_dimensional_byte_extent() {
        let mut message = vec![0u8; 16];
        message[0] = 1;
        message[1] = 1;
        message[8..16].copy_from_slice(&512u64.to_le_bytes());
        assert_eq!(dataspace_dims(&message).unwrap(), vec![512]);
        assert_eq!(dataset_byte_len(&[512], 1).unwrap(), 512);
    }
}

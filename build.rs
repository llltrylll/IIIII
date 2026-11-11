use std::env;
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/ui/icon.png");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("build output directory"));
    let png = fs::read(root.join("assets/ui/icon.png")).expect("read application icon");
    let source = image::load_from_memory(&png).expect("decode application icon PNG");
    // One PNG per size so Explorer, the taskbar and Alt+Tab each get a sharp icon.
    let pngs: Vec<Vec<u8>> = ICON_SIZES
        .iter()
        .map(|&size| {
            let mut encoded = Cursor::new(Vec::new());
            source
                .resize_exact(size, size, image::imageops::FilterType::Lanczos3)
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("encode Windows icon size");
            encoded.into_inner()
        })
        .collect();
    let ico = out.join("material-painter.ico");
    fs::write(&ico, ico_from_pngs(&pngs)).expect("write Windows icon");

    let rc = out.join("material-painter.rc");
    let res = out.join("material-painter.res");
    let icon_path = ico.to_string_lossy().replace('\\', "\\\\");
    fs::write(&rc, format!("1 ICON \"{icon_path}\"\n")).expect("write resource script");

    let compiler = find_resource_compiler().unwrap_or_else(|| {
        panic!("Windows SDK rc.exe was not found; install the Windows SDK to embed the executable icon")
    });
    let status = Command::new(&compiler)
        .arg("/nologo")
        .arg("/fo")
        .arg(&res)
        .arg(&rc)
        .status()
        .unwrap_or_else(|e| panic!("could not run {}: {e}", compiler.display()));
    assert!(status.success(), "Windows resource compiler failed");

    println!("cargo:rustc-link-arg-bins={}", res.display());
}

/// Sizes embedded in the .exe's icon, smallest first.
pub const ICON_SIZES: [u32; 7] = [16, 24, 32, 48, 64, 128, 256];

/// An .ico file holding each square PNG as is (PNG-in-ICO, Windows Vista and later).
pub fn ico_from_pngs(pngs: &[Vec<u8>]) -> Vec<u8> {
    let header = 6 + 16 * pngs.len();
    let mut ico = Vec::with_capacity(header + pngs.iter().map(Vec::len).sum::<usize>());
    ico.extend_from_slice(&0u16.to_le_bytes()); // reserved
    ico.extend_from_slice(&1u16.to_le_bytes()); // type: icon
    ico.extend_from_slice(&(pngs.len() as u16).to_le_bytes());
    let mut offset = header as u32;
    for png in pngs {
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"), "icon entry is not a PNG");
        assert_eq!(&png[12..16], b"IHDR", "icon entry has no IHDR header");
        let width = u32::from_be_bytes(png[16..20].try_into().expect("PNG width"));
        let height = u32::from_be_bytes(png[20..24].try_into().expect("PNG height"));
        assert!(width == height && (1..=256).contains(&width), "icon sizes are square, up to 256");
        // A size of 256 is stored as 0 in the directory.
        let dim = if width == 256 { 0 } else { width as u8 };
        ico.extend_from_slice(&[dim, dim, 0, 0]); // width, height, palette size, reserved
        ico.extend_from_slice(&1u16.to_le_bytes()); // color planes
        ico.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
        ico.extend_from_slice(&(png.len() as u32).to_le_bytes());
        ico.extend_from_slice(&offset.to_le_bytes());
        offset += png.len() as u32;
    }
    for png in pngs {
        ico.extend_from_slice(png);
    }
    ico
}

fn find_resource_compiler() -> Option<PathBuf> {
    for name in ["rc.exe", "llvm-rc.exe"] {
        if let Ok(output) = Command::new("where.exe").arg(name).output()
            && output.status.success()
                && let Some(path) = String::from_utf8_lossy(&output.stdout).lines().next() {
                    let path = PathBuf::from(path.trim());
                    if path.is_file() {
                        return Some(path);
                    }
                }
    }

    let mut roots = Vec::new();
    if let Some(sdk) = env::var_os("WindowsSdkDir") {
        roots.push(PathBuf::from(sdk));
    }
    roots.push(PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10"));

    for root in roots {
        let Ok(entries) = fs::read_dir(root.join("bin")) else { continue };
        let mut versions: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
        versions.sort();
        for version in versions.into_iter().rev() {
            for arch in ["x64", "arm64", "x86"] {
                let candidate = version.join(arch).join("rc.exe");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

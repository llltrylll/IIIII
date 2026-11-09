use std::env;
use std::fs;
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
    let ico = out.join("material-painter.ico");
    fs::write(&ico, png_ico(&png)).expect("write Windows icon");

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

    // Link the compiled ICON resource into the actual Windows executable.
    println!("cargo:rustc-link-arg-bins={}", res.display());
}

fn png_ico(png: &[u8]) -> Vec<u8> {
    // Embed the original PNG as an ICO image. The ICO size byte is zero for
    // dimensions >= 256 px, as required by the Windows icon directory format.
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"), "application icon is not a PNG");
    assert!(png.len() >= 24, "application icon PNG header is truncated");
    assert_eq!(&png[12..16], b"IHDR", "application icon has no IHDR header");
    let width = u32::from_be_bytes(png[16..20].try_into().expect("PNG width"));
    let height = u32::from_be_bytes(png[20..24].try_into().expect("PNG height"));
    assert!(width > 0 && height > 0, "application icon has invalid dimensions");
    assert!(width <= 256 && height <= 256, "Windows ICO PNG entries cannot exceed 256 px");
    let icon_width = if width == 256 { 0 } else { width as u8 };
    let icon_height = if height == 256 { 0 } else { height as u8 };

    let mut ico = Vec::with_capacity(22 + png.len());
    ico.extend_from_slice(&0u16.to_le_bytes()); // reserved
    ico.extend_from_slice(&1u16.to_le_bytes()); // icon type
    ico.extend_from_slice(&1u16.to_le_bytes()); // image count
    ico.extend_from_slice(&[icon_width, icon_height, 0, 0]);
    ico.extend_from_slice(&1u16.to_le_bytes()); // color planes
    ico.extend_from_slice(&32u16.to_le_bytes()); // bits per pixel
    ico.extend_from_slice(&(png.len() as u32).to_le_bytes());
    ico.extend_from_slice(&22u32.to_le_bytes()); // image offset
    ico.extend_from_slice(png);
    ico
}

fn find_resource_compiler() -> Option<PathBuf> {
    for name in ["rc.exe", "llvm-rc.exe"] {
        if let Ok(output) = Command::new("where.exe").arg(name).output() {
            if output.status.success() {
                if let Some(path) = String::from_utf8_lossy(&output.stdout).lines().next() {
                    let path = PathBuf::from(path.trim());
                    if path.is_file() {
                        return Some(path);
                    }
                }
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

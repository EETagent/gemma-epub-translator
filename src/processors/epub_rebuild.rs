use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

#[derive(Debug, Error)]
pub enum RebuildError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
}

pub fn rebuild_epub_with<I: AsRef<Path>, O: AsRef<Path>>(
    input_path: I,
    output_path: O,
    mut replace_html: impl FnMut(&str, &[u8]) -> Option<Vec<u8>>,
) -> Result<(), RebuildError> {
    let input_path = input_path.as_ref();
    let output_path = output_path.as_ref();

    let input_file = File::open(input_path)?;
    let mut archive = ZipArchive::new(input_file)?;

    let output_file = File::create(output_path)?;
    let mut writer = ZipWriter::new(output_file);

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();
        let mut options = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .last_modified_time(file.last_modified());

        if name == "mimetype" {
            options = options.compression_method(CompressionMethod::Stored);
        }

        if file.is_dir() {
            writer.add_directory(&name, options)?;
            continue;
        }

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let replacement = replace_html(&name, &buffer);
        let payload = replacement.as_deref().unwrap_or(&buffer);

        writer.start_file(&name, options)?;
        writer.write_all(payload)?;
    }

    writer.finish()?;
    Ok(())
}

pub fn output_path_with_locale(input_path: &Path, target_locale: &str) -> PathBuf {
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("book");
    let file_name = format!("{stem}-{target_locale}.epub");
    parent.join(file_name)
}

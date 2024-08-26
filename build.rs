fn main() {
    #[cfg(feature = "c-bindgen")]
    generate_c_bindings();
}

#[cfg(feature = "c-bindgen")]
fn generate_c_bindings() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_root_or_default(crate_dir))
        .generate()
        .expect("failed to generate C bindings")
        .write_to_file("ferroc.h");
}

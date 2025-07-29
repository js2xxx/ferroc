use std::{
    collections::HashMap,
    io::{BufWriter, Write},
};

fn main() {
    generate_configs();
    #[cfg(feature = "c-bindgen")]
    generate_c_bindings();
}

fn generate_configs() {
    #[derive(Debug, Clone, Copy)]
    struct ConfigValue {
        value: usize,
        comment: &'static str,
    }

    const CONFIGS: &[(&str, ConfigValue)] = &[
        ("SHARD_SHIFT", ConfigValue {
            value: 6 + 10,
            comment: "The minimal allocation unit of slabs, in bits.",
        }),
        ("SLAB_SHIFT", ConfigValue {
            value: 2 + 10 + 10,
            comment: "The minimal allocation unit of arenas, in bits.",
        }),
    ];

    let mut configs = CONFIGS.iter().copied().collect::<HashMap<_, _>>();

    for (env, value) in std::env::vars() {
        if let Some(name) = env.strip_prefix("FE_")
            && let Some(slot) = configs.get_mut(name)
            && let Ok(value) = value.parse::<usize>()
        {
            slot.value = value;
        }
    }

    let output_dir = std::env::var("OUT_DIR").unwrap();
    let file = std::fs::File::create(format!("{output_dir}/config.rs")).unwrap();
    let mut file = BufWriter::new(file);

    for (name, config) in configs {
        writeln!(
            &mut file,
            "#[doc = \"{}\"] pub const {name}: usize = {};",
            config.comment, config.value
        )
        .unwrap();
    }
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

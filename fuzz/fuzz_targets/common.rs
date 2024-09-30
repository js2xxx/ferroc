use std::sync::atomic::AtomicUsize;

use libfuzzer_sys::arbitrary::Arbitrary;

pub const THREADS: usize = 12;
pub const TRANSFER_COUNT: usize = 1000;

#[derive(Debug, Clone, Copy, Arbitrary)]
pub enum AllocIface {
    Ferroc,
    Native,
    Global,
}

#[derive(Debug, Arbitrary)]
pub enum Action {
    Allocate {
        size: u32,
        align_shift: u8,
        zeroed: bool,
        iface: AllocIface,
    },
    Deallocate {
        index: u8,
    },
    LayoutOf {
        index: u8,
    },
    Collect {
        force: bool,
    },
    Transfer {
        from: u8,
        to: u8,
    },
    Manage {
        size: u8,
    },
}

pub static MANAGED: AtomicUsize = AtomicUsize::new(0);
pub const MAX_MANAGED: usize = 20;

#[doc(hidden)]
pub mod thread;

#[cfg(feature = "stat")]
#[macro_export]
#[doc(hidden)]
macro_rules! config_stat {
    () => {
        $vis fn stat(&self) -> $crate::stat::Stat {
            thread::with(|heap| heap.stat())
        }
    };
}

#[cfg(not(feature = "stat"))]
#[macro_export]
#[doc(hidden)]
macro_rules! config_stat {
    () => {};
}

#[macro_export]
#[doc(hidden)]
macro_rules! config_inner {
    (@TYPES $vis:vis, $bt:ty) => {
        $vis type Chunk = $crate::base::Chunk<$bt>;
        $vis type Heap<'arena, 'cx> = $crate::heap::Heap<'arena, 'cx, $bt>;
        $vis type Context<'arena> = $crate::heap::Context<'arena, $bt>;
        $vis type Arenas = $crate::arena::Arenas<$bt>;
        $vis type Error = $crate::arena::Error<$bt>;
    };
    (@ARENA $vis:vis, $bs:expr) => {
        static ARENAS: Arenas = Arenas::new($bs);
    };
    (@DEF $vis:vis, $name:ident, $bt:ty) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        $vis struct $name;

        impl $name {
            $vis fn base(&self) -> &$bt {
                ARENAS.base()
            }

            $vis fn manage(&self, chunk: Chunk) -> Result<(), Error> {
                ARENAS.manage(chunk)
            }

            $vis fn collect(&self, force: bool) {
                thread::with(|heap| heap.collect(force))
            }

            $crate::config_stat!();

            $vis fn allocate(&self, layout: core::alloc::Layout) -> Result<core::ptr::NonNull<[u8]>, Error> {
                thread::with(|heap| heap.allocate(layout))
            }

            /// # Safety
            ///
            /// `ptr` must point to an owned, valid memory block of `layout`, previously
            /// allocated by a certain instance of `Heap` alive in the scope, created
            /// from the same arena.
            $vis unsafe fn layout_of(&self, ptr: core::ptr::NonNull<u8>) -> Option<core::alloc::Layout> {
                thread::with(|heap| heap.layout_of(ptr))
            }

            /// # Safety
            ///
            /// See [`core::alloc::Allocator::deallocate`] for more information.
            $vis unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
                thread::with(|heap| heap.deallocate(ptr, layout))
            }

            /// # Safety
            ///
            /// `ptr` must point to an owned, valid memory block, previously allocated
            /// by a certain instance of `Heap` alive in the scope.
            #[cfg(feature = "c")]
            #[inline]
            $vis unsafe fn free(&self, ptr: core::ptr::NonNull<u8>) {
                thread::with(|heap| heap.free(ptr))
            }
        }

        unsafe impl core::alloc::Allocator for $name {
            fn allocate(&self, layout: core::alloc::Layout) -> Result<core::ptr::NonNull<[u8]>, core::alloc::AllocError> {
                thread::with(|heap| core::alloc::Allocator::allocate(&heap, layout))
            }

            unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
                thread::with(|heap| core::alloc::Allocator::deallocate(&heap, ptr, layout))
            }
        }

        unsafe impl core::alloc::GlobalAlloc for $name {
            unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
                self.allocate(layout)
                    .map_or(core::ptr::null_mut(), |ptr| ptr.as_ptr().cast())
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: core::alloc::Layout) {
                if let Some(ptr) = core::ptr::NonNull::new(ptr) {
                    self.deallocate(ptr, layout)
                }
            }
        }
    };
}

#[macro_export]
#[allow_internal_unstable(allocator_api)]
macro_rules! config {
    ($vis:vis $name:ident($bs:expr) => $bt:ty: $($options:tt)*) => {
        $crate::thread_mod!($($options)*);
        $crate::config_inner!(@TYPES $vis, $bt);
        $crate::config_inner!(@ARENA $vis, $bs);
        $crate::config_inner!(@DEF $vis, $name, $bt);
    };
    ($vis:vis $name:ident($bs:expr) => $bt:ty) => {
        $crate::thread_mod!();
        $crate::config_inner!(@TYPES $vis, $bt);
        $crate::config_inner!(@ARENA $vis, $bs);
        $crate::config_inner!(@DEF $vis, $name, $bt);
    };
    ($vis:vis $name:ident => $bt:ty: $($options:tt)*) => {
        $crate::thread_mod!($($options)*);
        $crate::config_inner!(@TYPES $vis, $bt);
        $crate::config_inner!(@ARENA $vis, <$bt>::new());
        $crate::config_inner!(@DEF $vis, $name, $bt);
    };
    ($vis:vis $name:ident => $bt:ty) => {
        $crate::thread_mod!();
        $crate::config_inner!(@TYPES $vis, $bt);
        $crate::config_inner!(@ARENA $vis, <$bt>::new());
        $crate::config_inner!(@DEF $vis, $name, $bt);
    };
    ($vis:vis $bt:ty: $($options:tt)*) => {
        $crate::thread_mod!($($options)*);
        $crate::config_inner!(@TYPES $vis, $bt);
        $crate::config_inner!(@ARENA $vis, <$bt>::new());
        $crate::config_inner!(@DEF $vis, Ferroc, $bt);
    };
    ($vis:vis $bt:ty) => {
        $crate::thread_mod!();
        $crate::config_inner!(@TYPES $vis, $bt);
        $crate::config_inner!(@ARENA $vis, <$bt>::new());
        $crate::config_inner!(@DEF $vis, Ferroc, $bt);
    };
}

#[macro_export]
macro_rules! config_mod {
    ($vis:vis $name:ident: $($rest:tt)*) => {
        $vis mod $name {
            $crate::config!($($rest)*);
        }
    };
}

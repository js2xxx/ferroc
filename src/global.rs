#[doc(hidden)]
pub mod thread;

#[cfg(feature = "stat")]
#[macro_export]
#[doc(hidden)]
macro_rules! config_stat {
    ($vis:vis) => {
        /// Retrieves the statistics of this allocator.
        ///
        /// This function forwards the call to [`Heap::stat`].
        $vis fn stat(&self) -> $crate::stat::Stat {
            thread::with(|heap| heap.stat())
        }
    };
}

#[cfg(not(feature = "stat"))]
#[macro_export]
#[doc(hidden)]
macro_rules! config_stat {
    ($vis:vis) => {};
}

#[macro_export]
#[doc(hidden)]
#[allow_internal_unstable(allocator_api)]
macro_rules! config_inner {
    (@TYPES $vis:vis, $bt:ty) => {
        #[doc = concat!("The chunk type of the `", stringify!($bt), "` backend.")]
        #[doc = concat!("\n\nSee [`Chunk`](", stringify!($crate), "::base::Chunk) for more information.")]
        $vis type Chunk = $crate::base::Chunk<$bt>;
        #[doc = concat!("The heap type of the `", stringify!($bt), "` backend.")]
        #[doc = concat!("\n\nSee [`Heap`](", stringify!($crate), "::heap::Heap) for more information.")]
        $vis type Heap<'arena, 'cx> = $crate::heap::Heap<'arena, 'cx, $bt>;
        #[doc = concat!("The context type of the `", stringify!($bt), "` backend.")]
        #[doc = concat!("\n\nSee [`Context`](", stringify!($crate), "::heap::Context) for more information.")]
        $vis type Context<'arena> = $crate::heap::Context<'arena, $bt>;
        #[doc = concat!("The arena collection type of the `", stringify!($bt), "` backend.")]
        #[doc = concat!("\n\nSee [`Arenas`](", stringify!($crate), "::arena::Arenas) for more information.")]
        $vis type Arenas = $crate::arena::Arenas<$bt>;
        #[doc = concat!("The error type of the `", stringify!($bt), "` backend.")]
        #[doc = concat!("\n\nSee [`Error`](", stringify!($crate), "::arena::Error) for more information.")]
        $vis type Error = $crate::arena::Error<$bt>;
    };
    (@ARENA $vis:vis, $bs:expr) => {
        static ARENAS: Arenas = Arenas::new($bs);
    };
    (@DEF $vis:vis, $name:ident, $bt:ty) => {
        #[doc = concat!("The configured allocator backed by `", stringify!($bt), "`.\n\n")]
        /// This allocator is the interface of a global instance of arenas
        /// and thread-local contexts and heaps. It forwards most of the
        /// function calls to the actual implementation of them.
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        $vis struct $name;

        impl $name {
            /// Retrieves the base allocator of this configured memory allocator.
            ///
            /// This function forwards the call to [`Arenas::base`].
            $vis fn base(&self) -> &$bt {
                ARENAS.base()
            }

            /// Manages another chunk previously allocated by an instance of its base
            /// allocator.
            ///
            /// This function creates a new arena from the chunk and push it to the
            /// allocator for further allocation, extending the heap's overall
            /// capacity.
            ///
            /// This function forwards the call to [`Arenas::manage`].
            $vis fn manage(&self, chunk: Chunk) -> Result<(), Error> {
                ARENAS.manage(chunk)
            }

            /// Clean up some garbage data of the current heap immediately.
            ///
            /// In short, due to implementation details, the free list (i.e. the popper
            /// of allocation) and the deallocation list (i.e. the pusher of
            /// deallocation) of a [`Heap`] are 2 distinct lists.
            ///
            /// Those 2 lists will be swapped if the former becomes empty during the
            /// allocation process, which is precisely what this function does.
            ///
            /// This function forwards the call to [`Heap::collect`].
            $vis fn collect(&self, force: bool) {
                thread::with(|heap| heap.collect(force))
            }

            $crate::config_stat!($vis);

            /// Allocate a memory block of `layout` from the current heap.
            ///
            /// The allocation can be deallocated by any instance of this configured
            /// allocator.
            ///
            /// This function forwards the call to [`Heap::allocate`].
            ///
            /// # Errors
            ///
            /// Errors are returned when allocation fails, see [`Error`] for more
            /// information.
            $vis fn allocate(&self, layout: core::alloc::Layout)
                -> Result<core::ptr::NonNull<[u8]>, Error>
            {
                thread::with(|heap| heap.allocate(layout))
            }

            /// Retrieves the layout information of a specific allocation.
            ///
            /// The layout returned may not be the same of the layout passed to
            /// [`allocate`](Heap::allocate), but is the most fit layout of it, and can
            /// be passed to [`deallocate`](Heap::deallocate).
            ///
            /// This function forwards the call to [`Heap::layout_of`].
            ///
            /// # Safety
            ///
            /// `ptr` must point to an owned, valid memory block of `layout`, previously
            /// allocated by a certain instance of `Heap` alive in the scope, created
            /// from the same arena.
            $vis unsafe fn layout_of(&self, ptr: core::ptr::NonNull<u8>) -> Option<core::alloc::Layout> {
                thread::with(|heap| heap.layout_of(ptr))
            }

            /// Deallocates an allocation previously allocated by an instance of this
            /// type.
            ///
            /// This function forwards the call to [`Heap::deallocate`].
            ///
            /// # Safety
            ///
            /// See [`core::alloc::Allocator::deallocate`] for more information.
            $vis unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
                thread::with(|heap| heap.deallocate(ptr, layout))
            }

            /// Like `deallocate`, but without specifying its layout information.
            ///
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

/// Configures and instantiates a global instance of a `ferroc`ator in place.
///
/// See [`the crate-level documentation`](crate) for its usage.
///
/// # Arguments
///
/// - `vis` - the visibility of the allocator type.
/// - `name`(optional) - the ident of the allocator type.
/// - `bs`(optional) - the initialization expresstion of the base allocator,
///   must be able to be called at compile time. Defaults to `<$bt>::new()`.
/// - `bt` - the type path of the wrapped base allocator.
/// - `options`(optional):
///   - `pthread` - use the `pthread` thread-local key destructor when a thread
///     exits. Requires `libc` feature.
#[macro_export]
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

/// Configures and instantiates a global instance of a `ferroc`ator in an inline
/// module.
///
/// See [`config`] for more information about its arguments.
///
/// # Arguments
///
/// - `vis` - the visibility of the module (not the allocator!).
/// - `name` - the name of the module (not the allocator!).
/// - `rest` - forwards to `config!`.
#[macro_export]
macro_rules! config_mod {
    ($vis:vis $name:ident: $($rest:tt)*) => {
        $vis mod $name {
            $crate::config!($($rest)*);
        }
    };
}

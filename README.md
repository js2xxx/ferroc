# Ferroc: A Multithread Lock-free Memory Allocator

Ferroc (combined from "ferrous" and "malloc") is a lock-free concurrent memory allocator written in Rust, primarily inspired by [`mimalloc`](https://github.com/microsoft/mimalloc).

This memory allocator is designed to work as fast as other mainstream memory allocators while providing flexible configurations such as embedded/bare-metal environment integrations.
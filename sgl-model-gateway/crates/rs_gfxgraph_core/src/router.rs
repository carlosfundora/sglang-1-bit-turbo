//! Pure-Rust bucket routing for CUDA graph shape bucketing.
//!
//! Extracted from the PyO3 `BucketRouter` — this module contains the
//! algorithmic core without any Python dependency, allowing it to be
//! used from pure-Rust consumers, WASM sandboxes, or thin FFI shells.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::GfxGraphError;

/// Routing state for a resolved bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BucketState {
    /// Bucket has been warmed up and is ready for replay.
    Ready,
    /// Bucket exists but has not been warmed up yet.
    NeedsWarmup,
    /// Bucket capture/replay previously failed; use eager fallback.
    Failed,
}

/// Core bucket routing engine for CUDA graph shape bucketing.
///
/// Maps variable input sizes to the nearest pre-defined bucket size via
/// binary search, then tracks warmup and failure state per bucket.
#[derive(Debug)]
pub struct BucketRouterCore {
    /// Sorted list of supported bucket sizes.
    buckets: Vec<usize>,
    /// Buckets that have been successfully warmed up.
    warmed_up: HashSet<usize>,
    /// Buckets that failed during capture/replay.
    failed_buckets: HashSet<usize>,
}

impl BucketRouterCore {
    /// Create a new router with the given bucket sizes.
    ///
    /// The `buckets` list is sorted internally; callers do not need to
    /// pre-sort.
    pub fn new(mut buckets: Vec<usize>) -> Self {
        buckets.sort_unstable();
        Self {
            buckets,
            warmed_up: HashSet::new(),
            failed_buckets: HashSet::new(),
        }
    }

    /// Resolve `input_size` to a bucket and its current state.
    ///
    /// Returns `(bucket_size, state)` or an error if the input exceeds
    /// the largest bucket.
    pub fn route(&self, input_size: usize) -> Result<(usize, BucketState), GfxGraphError> {
        let bucket = match self.buckets.binary_search(&input_size) {
            Ok(idx) => self.buckets[idx],
            Err(idx) => {
                if idx < self.buckets.len() {
                    self.buckets[idx]
                } else {
                    return Err(GfxGraphError::ExecutionError {
                        branch: format!("bucket_{}", input_size),
                        reason: format!(
                            "Input size {} exceeds largest bucket {}. Add a larger bucket.",
                            input_size,
                            self.buckets.last().copied().unwrap_or(0)
                        ),
                    });
                }
            }
        };

        let state = if self.warmed_up.contains(&bucket) {
            BucketState::Ready
        } else if self.failed_buckets.contains(&bucket) {
            BucketState::Failed
        } else {
            BucketState::NeedsWarmup
        };

        Ok((bucket, state))
    }

    /// Mark a bucket as successfully warmed up.
    pub fn mark_warmed_up(&mut self, bucket_size: usize) {
        self.warmed_up.insert(bucket_size);
    }

    /// Mark a bucket as failed (will route to eager fallback).
    pub fn mark_failed(&mut self, bucket_size: usize) {
        self.failed_buckets.insert(bucket_size);
    }

    /// Return the sorted list of warmed-up bucket sizes.
    pub fn warmed_up_list(&self) -> Vec<usize> {
        let mut list: Vec<usize> = self.warmed_up.iter().copied().collect();
        list.sort_unstable();
        list
    }

    /// Return the sorted list of failed bucket sizes.
    pub fn failed_list(&self) -> Vec<usize> {
        let mut list: Vec<usize> = self.failed_buckets.iter().copied().collect();
        list.sort_unstable();
        list
    }

    /// Return the raw bucket sizes.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_routes_correctly() {
        let router = BucketRouterCore::new(vec![64, 128, 256, 512]);
        let (bucket, state) = router.route(128).unwrap();
        assert_eq!(bucket, 128);
        assert_eq!(state, BucketState::NeedsWarmup);
    }

    #[test]
    fn ceil_match_routes_to_next_bucket() {
        let router = BucketRouterCore::new(vec![64, 128, 256, 512]);
        let (bucket, state) = router.route(100).unwrap();
        assert_eq!(bucket, 128);
        assert_eq!(state, BucketState::NeedsWarmup);
    }

    #[test]
    fn exceeds_largest_bucket_is_error() {
        let router = BucketRouterCore::new(vec![64, 128, 256]);
        let result = router.route(512);
        assert!(result.is_err());
    }

    #[test]
    fn warmup_state_transition() {
        let mut router = BucketRouterCore::new(vec![64, 128]);
        assert_eq!(router.route(64).unwrap().1, BucketState::NeedsWarmup);

        router.mark_warmed_up(64);
        assert_eq!(router.route(64).unwrap().1, BucketState::Ready);
    }

    #[test]
    fn failed_state_transition() {
        let mut router = BucketRouterCore::new(vec![64, 128]);
        router.mark_failed(128);
        assert_eq!(router.route(128).unwrap().1, BucketState::Failed);
    }

    #[test]
    fn sorted_lists() {
        let mut router = BucketRouterCore::new(vec![512, 64, 256, 128]);
        router.mark_warmed_up(256);
        router.mark_warmed_up(64);
        router.mark_failed(512);

        assert_eq!(router.warmed_up_list(), vec![64, 256]);
        assert_eq!(router.failed_list(), vec![512]);
    }

    #[test]
    fn unsorted_input_is_sorted_internally() {
        let router = BucketRouterCore::new(vec![256, 64, 512, 128]);
        assert_eq!(router.buckets(), &[64, 128, 256, 512]);
    }
}

//! Pure-Rust graph runner state management.
//!
//! Extracted from the PyO3 `ConditionalGraphRunner` — this module owns
//! the branch registry and failure tracking logic. The actual execution
//! (torch CUDAGraph replay, no_grad context, eager fallback calls) stays
//! in the PyO3 binding crate since those operations require GIL access.

use std::collections::HashSet;
use std::sync::RwLock;

/// Thread-safe state tracker for the conditional graph runner.
///
/// Tracks which branches exist and which have failed (requiring eager
/// fallback). This struct is designed to be embedded inside a PyO3
/// `#[pyclass]` wrapper that adds the `Py<PyAny>` fields for graphs,
/// static outputs, and callbacks.
#[derive(Debug)]
pub struct GraphRunnerState {
    /// Known branch names.
    branches: Vec<String>,
    /// Branches that have failed during replay — will be routed to
    /// eager fallback on subsequent calls.
    failed_branches: RwLock<HashSet<String>>,
}

impl GraphRunnerState {
    /// Create a new runner state with the given branch names and
    /// an initial set of known-failed branches.
    pub fn new(branches: Vec<String>, initial_failed: Vec<String>) -> Self {
        let mut failed = HashSet::new();
        for b in initial_failed {
            failed.insert(b);
        }
        Self {
            branches,
            failed_branches: RwLock::new(failed),
        }
    }

    /// Check whether a branch name is registered.
    pub fn is_known_branch(&self, branch: &str) -> bool {
        self.branches.iter().any(|b| b == branch)
    }

    /// Check whether a branch is marked as failed.
    pub fn is_failed(&self, branch: &str) -> bool {
        self.failed_branches
            .read()
            .map(|lock| lock.contains(branch))
            .unwrap_or(false)
    }

    /// Mark a branch as failed (future calls will use eager fallback).
    pub fn mark_failed(&self, branch: &str) {
        if let Ok(mut lock) = self.failed_branches.write() {
            lock.insert(branch.to_string());
        }
    }

    /// Return the list of registered branch names.
    pub fn branches(&self) -> &[String] {
        &self.branches
    }

    /// Return the sorted list of failed branch names.
    pub fn failed_list(&self) -> Vec<String> {
        self.failed_branches
            .read()
            .map(|lock| {
                let mut list: Vec<String> = lock.iter().cloned().collect();
                list.sort_unstable();
                list
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_branch_lookup() {
        let state = GraphRunnerState::new(vec!["decode".into(), "prefill".into()], vec![]);
        assert!(state.is_known_branch("decode"));
        assert!(state.is_known_branch("prefill"));
        assert!(!state.is_known_branch("nonexistent"));
    }

    #[test]
    fn initial_failed_branches() {
        let state =
            GraphRunnerState::new(vec!["a".into(), "b".into(), "c".into()], vec!["b".into()]);
        assert!(!state.is_failed("a"));
        assert!(state.is_failed("b"));
        assert!(!state.is_failed("c"));
    }

    #[test]
    fn mark_failed_transitions() {
        let state = GraphRunnerState::new(vec!["x".into()], vec![]);
        assert!(!state.is_failed("x"));

        state.mark_failed("x");
        assert!(state.is_failed("x"));
    }

    #[test]
    fn failed_list_sorted() {
        let state = GraphRunnerState::new(
            vec!["c".into(), "a".into(), "b".into()],
            vec!["c".into(), "a".into()],
        );
        assert_eq!(state.failed_list(), vec!["a", "c"]);
    }
}

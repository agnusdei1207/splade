use std::collections::HashSet;

use anyhow::{Result, bail, ensure};

#[derive(Clone, Debug, PartialEq)]
pub struct SparseVector {
    term_ids: Vec<u16>,
    weights: Vec<f32>,
}

impl SparseVector {
    pub fn try_from_pairs(
        pairs: impl IntoIterator<Item = (usize, f32)>,
        limit: usize,
    ) -> Result<Self> {
        let mut seen = HashSet::new();
        let mut checked = Vec::new();
        for (term_id, weight) in pairs {
            let term_id = u16::try_from(term_id)?;
            ensure!(seen.insert(term_id), "duplicate sparse term id {term_id}");
            ensure!(weight.is_finite() && weight >= 0.0, "invalid sparse weight");
            if weight > 0.0 {
                checked.push((term_id, weight));
            }
        }
        checked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        checked.truncate(limit);
        checked.sort_unstable_by_key(|pair| pair.0);
        let (term_ids, weights) = checked.into_iter().unzip();
        Ok(Self { term_ids, weights })
    }

    #[must_use]
    pub fn term_ids(&self) -> &[u16] {
        &self.term_ids
    }

    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    #[must_use]
    pub fn active_terms(&self) -> usize {
        self.term_ids.len()
    }

    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut left = 0;
        let mut right = 0;
        let mut score = 0.0;
        while left < self.term_ids.len() && right < other.term_ids.len() {
            match self.term_ids[left].cmp(&other.term_ids[right]) {
                std::cmp::Ordering::Equal => {
                    score += self.weights[left] * other.weights[right];
                    left += 1;
                    right += 1;
                }
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
            }
        }
        score
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.term_ids.len() == self.weights.len(),
            "sparse term/weight length mismatch"
        );
        if self.term_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            bail!("sparse term ids are not strictly increasing");
        }
        ensure!(
            self.weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0),
            "sparse vector contains an invalid weight"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_keeps_strongest_terms_and_scores_intersection() {
        let vector = SparseVector::try_from_pairs([(9, 0.5), (2, 3.0), (7, 2.0)], 2).unwrap();
        assert_eq!(vector.term_ids(), &[2, 7]);
        assert_eq!(vector.weights(), &[3.0, 2.0]);
        let other = SparseVector::try_from_pairs([(2, 2.0), (9, 8.0)], 2).unwrap();
        assert_eq!(vector.dot(&other), 6.0);
    }

    #[test]
    fn vector_rejects_duplicate_terms_and_non_finite_weights() {
        assert!(SparseVector::try_from_pairs([(1, 1.0), (1, 2.0)], 2).is_err());
        assert!(SparseVector::try_from_pairs([(1, f32::NAN)], 2).is_err());
    }
}

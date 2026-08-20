use std::collections::{HashMap, HashSet};

use anyhow::{Result, ensure};

use crate::SparseVector;

#[derive(Debug, Default)]
pub struct SparseIndex {
    document_ids: Vec<String>,
    postings: HashMap<u16, Vec<(u32, f32)>>,
}

impl SparseIndex {
    pub fn build(documents: impl IntoIterator<Item = (String, SparseVector)>) -> Result<Self> {
        let mut document_ids = Vec::new();
        let mut seen_ids = HashSet::new();
        let mut postings: HashMap<u16, Vec<(u32, f32)>> = HashMap::new();
        for (document_id, vector) in documents {
            ensure!(
                seen_ids.insert(document_id.clone()),
                "duplicate document id {document_id}"
            );
            vector.validate()?;
            let ordinal = u32::try_from(document_ids.len())?;
            for (&term_id, &weight) in vector.term_ids().iter().zip(vector.weights()) {
                postings.entry(term_id).or_default().push((ordinal, weight));
            }
            document_ids.push(document_id);
        }
        Ok(Self {
            document_ids,
            postings,
        })
    }

    #[must_use]
    pub fn posting_count(&self) -> usize {
        self.postings.values().map(Vec::len).sum()
    }

    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = b"SPLADE01".to_vec();
        push_u32(&mut bytes, self.document_ids.len());
        for document_id in &self.document_ids {
            push_u32(&mut bytes, document_id.len());
            bytes.extend_from_slice(document_id.as_bytes());
        }
        let mut terms = self.postings.iter().collect::<Vec<_>>();
        terms.sort_unstable_by_key(|(term_id, _)| **term_id);
        push_u32(&mut bytes, terms.len());
        for (term_id, postings) in terms {
            bytes.extend_from_slice(&term_id.to_le_bytes());
            push_u32(&mut bytes, postings.len());
            for (document_ordinal, weight) in postings {
                bytes.extend_from_slice(&document_ordinal.to_le_bytes());
                bytes.extend_from_slice(&weight.to_le_bytes());
            }
        }
        bytes
    }

    #[must_use]
    pub fn search(&self, query: &SparseVector, limit: usize) -> Vec<(String, f32)> {
        let mut scores = vec![0.0f32; self.document_ids.len()];
        let mut touched = Vec::new();
        let mut seen = vec![false; self.document_ids.len()];
        for (&term_id, &query_weight) in query.term_ids().iter().zip(query.weights()) {
            for &(document_ordinal, document_weight) in
                self.postings.get(&term_id).into_iter().flatten()
            {
                let ordinal = usize::try_from(document_ordinal).expect("u32 fits usize");
                scores[ordinal] += query_weight * document_weight;
                if !seen[ordinal] {
                    seen[ordinal] = true;
                    touched.push(ordinal);
                }
            }
        }
        let mut ranked = touched
            .into_iter()
            .map(|ordinal| (self.document_ids[ordinal].clone(), scores[ordinal]))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit);
        ranked
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(
        &u32::try_from(value)
            .expect("SPLADE index exceeds u32 format limit")
            .to_le_bytes(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverted_index_ranks_by_sparse_dot_product() {
        let index = SparseIndex::build([
            (
                "weak".to_owned(),
                SparseVector::try_from_pairs([(1, 1.0)], 8).unwrap(),
            ),
            (
                "strong".to_owned(),
                SparseVector::try_from_pairs([(1, 2.0), (2, 1.0)], 8).unwrap(),
            ),
        ])
        .unwrap();
        let query = SparseVector::try_from_pairs([(1, 1.0), (2, 3.0)], 8).unwrap();
        assert_eq!(
            index.search(&query, 2),
            vec![("strong".to_owned(), 5.0), ("weak".to_owned(), 1.0)]
        );
        assert_eq!(index.posting_count(), 3);
        assert_eq!(index.to_bytes(), index.to_bytes());
    }

    #[test]
    fn index_rejects_duplicate_document_ids() {
        let vector = SparseVector::try_from_pairs([(1, 1.0)], 8).unwrap();
        assert!(
            SparseIndex::build([
                ("same".to_owned(), vector.clone()),
                ("same".to_owned(), vector),
            ])
            .is_err()
        );
    }
}

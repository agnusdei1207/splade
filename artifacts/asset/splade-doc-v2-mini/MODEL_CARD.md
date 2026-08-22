---
language: en
license: apache-2.0
tags:
- learned sparse
- opensearch
- transformers
- retrieval
- passage-retrieval
- document-expansion
- bag-of-words
- sentence-transformers
- sparse-encoder
- sparse
- asymmetric
- inference-free
- splade
pipeline_tag: feature-extraction
library_name: sentence-transformers
---

# opensearch-neural-sparse-encoding-doc-v2-mini

## Select the model
The model should be selected considering search relevance, model inference and retrieval efficiency(FLOPS). We benchmark models' **zero-shot performance** on a subset of BEIR benchmark: TrecCovid,NFCorpus,NQ,HotpotQA,FiQA,ArguAna,Touche,DBPedia,SCIDOCS,FEVER,Climate FEVER,SciFact,Quora.

Overall, the v2 series of models have better search relevance, efficiency and inference speed than the v1 series. The specific advantages and disadvantages may vary across different datasets.

| Model | Inference-free for Retrieval | Model Parameters | AVG NDCG@10 | AVG FLOPS |
|-------|------------------------------|------------------|-------------|-----------|
| [opensearch-neural-sparse-encoding-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1) |  | 133M | 0.524 | 11.4 |
| [opensearch-neural-sparse-encoding-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill) |  | 67M | 0.528 | 8.3 |
| [opensearch-neural-sparse-encoding-doc-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v1) | ✔️ | 133M | 0.490 | 2.3 |
| [opensearch-neural-sparse-encoding-doc-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill) | ✔️ | 67M | 0.504 | 1.8 |
| [opensearch-neural-sparse-encoding-doc-v2-mini](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini) | ✔️ | 23M | 0.497 | 1.7 |
| [opensearch-neural-sparse-encoding-doc-v3-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill) | ✔️ | 67M | 0.517 | 1.8 |
| [opensearch-neural-sparse-encoding-doc-v3-gte](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) | ✔️ | 133M | 0.546 | 1.7 |

## Overview
- **Paper**: [Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers](https://arxiv.org/abs/2411.04403)
- **Fine-tuning sample**: [opensearch-sparse-model-tuning-sample](https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample)

This is a learned sparse retrieval model. It encodes the documents to 30522 dimensional **sparse vectors**. For queries, it just use a tokenizer and a weight look-up table to generate sparse vectors. The non-zero dimension index means the corresponding token in the vocabulary, and the weight means the importance of the token. And the similarity score is the inner product of query/document sparse vectors.

The training datasets includes MS MARCO, eli5_question_answer, squad_pairs, WikiAnswers, yahoo_answers_title_question, gooaq_pairs, stackexchange_duplicate_questions_body_body, wikihow, S2ORC_title_abstract, stackexchange_duplicate_questions_title-body_title-body, yahoo_answers_question_answer, searchQA_top5_snippets, stackexchange_duplicate_questions_title_title, yahoo_answers_title_answer.

OpenSearch neural sparse feature supports learned sparse retrieval with lucene inverted index. Link: https://opensearch.org/docs/latest/query-dsl/specialized/neural-sparse/. The indexing and search can be performed with OpenSearch high-level API.

## Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.

```python

from sentence_transformers.sparse_encoder import SparseEncoder

# Download from the 🤗 Hub
model = SparseEncoder("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini")

query = "What's the weather in ny now?"
document = "Currently New York is rainy."

query_embed = model.encode_query(query)
document_embed = model.encode_document(document)

sim = model.similarity(query_embed, document_embed)
print(f"Similarity: {sim}")
# Similarity: tensor([[13.8444]])

decoded_query = model.decode(query_embed)
decoded_document = model.decode(document_embed)

for i in range(len(decoded_query)):
    query_token, query_score = decoded_query[i]
    doc_score = next((score for token, score in decoded_document if token == query_token), 0)
    if doc_score != 0:
        print(f"Token: {query_token}, Query score: {query_score:.4f}, Document score: {doc_score:.4f}")

# Token: ny, Query score: 5.7729, Document score: 1.0251
# Token: weather, Query score: 4.5684, Document score: 1.1145
# Token: now, Query score: 3.5895, Document score: 0.5356
# Token: ?, Query score: 3.3313, Document score: 0.2710
```

## Usage (HuggingFace)
This model is supposed to run inside OpenSearch cluster. But you can also use it outside the cluster, with HuggingFace models API. 

```python
import json
import itertools
import torch

from transformers import AutoModelForMaskedLM, AutoTokenizer


# get sparse vector from dense vectors with shape batch_size * seq_len * vocab_size
def get_sparse_vector(feature, output):
    values, _ = torch.max(output*feature["attention_mask"].unsqueeze(-1), dim=1)
    values = torch.log(1 + torch.relu(values))
    values[:,special_token_ids] = 0
    return values
    
# transform the sparse vector to a dict of (token, weight)
def transform_sparse_vector_to_dict(sparse_vector):
    sample_indices,token_indices=torch.nonzero(sparse_vector,as_tuple=True)
    non_zero_values = sparse_vector[(sample_indices,token_indices)].tolist()
    number_of_tokens_for_each_sample = torch.bincount(sample_indices).cpu().tolist()
    tokens = [transform_sparse_vector_to_dict.id_to_token[_id] for _id in token_indices.tolist()]

    output = []
    end_idxs = list(itertools.accumulate([0]+number_of_tokens_for_each_sample))
    for i in range(len(end_idxs)-1):
        token_strings = tokens[end_idxs[i]:end_idxs[i+1]]
        weights = non_zero_values[end_idxs[i]:end_idxs[i+1]]
        output.append(dict(zip(token_strings, weights)))
    return output
    
# download the idf file from model hub. idf is used to give weights for query tokens
def get_tokenizer_idf(tokenizer):
    from huggingface_hub import hf_hub_download
    local_cached_path = hf_hub_download(repo_id="opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini", filename="idf.json")
    with open(local_cached_path) as f:
        idf = json.load(f)
    idf_vector = [0]*tokenizer.vocab_size
    for token,weight in idf.items():
        _id = tokenizer._convert_token_to_id_with_added_voc(token)
        idf_vector[_id]=weight
    return torch.tensor(idf_vector)

# load the model
model = AutoModelForMaskedLM.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini")
tokenizer = AutoTokenizer.from_pretrained("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini")
idf = get_tokenizer_idf(tokenizer)

# set the special tokens and id_to_token transform for post-process
special_token_ids = [tokenizer.vocab[token] for token in tokenizer.special_tokens_map.values()]
get_sparse_vector.special_token_ids = special_token_ids
id_to_token = ["" for i in range(tokenizer.vocab_size)]
for token, _id in tokenizer.vocab.items():
    id_to_token[_id] = token
transform_sparse_vector_to_dict.id_to_token = id_to_token



query = "What's the weather in ny now?"
document = "Currently New York is rainy."

# encode the query
feature_query = tokenizer([query], padding=True, truncation=True, return_tensors='pt', return_token_type_ids=False)
input_ids = feature_query["input_ids"]
batch_size = input_ids.shape[0]
query_vector = torch.zeros(batch_size, tokenizer.vocab_size)
query_vector[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
query_sparse_vector = query_vector*idf

# encode the document
feature_document = tokenizer([document], padding=True, truncation=True, return_tensors='pt', return_token_type_ids=False)
output = model(**feature_document)[0]
document_sparse_vector = get_sparse_vector(feature_document, output)


# get similarity score
sim_score = torch.matmul(query_sparse_vector[0],document_sparse_vector[0])
print(sim_score)   # tensor(13.8344, grad_fn=<DotBackward0>)


query_token_weight = transform_sparse_vector_to_dict(query_sparse_vector)[0]
document_query_token_weight = transform_sparse_vector_to_dict(document_sparse_vector)[0]
for token in sorted(query_token_weight, key=lambda x:query_token_weight[x], reverse=True):
    if token in document_query_token_weight:
        print("score in query: %.4f, score in document: %.4f, token: %s"%(query_token_weight[token],document_query_token_weight[token],token))
        

        
# result:
# score in query: 5.7729, score in document: 1.0251, token: ny
# score in query: 4.5684, score in document: 1.1145, token: weather
# score in query: 3.5895, score in document: 0.5356, token: now
# score in query: 3.3313, score in document: 0.2710, token: ?
```

The above code sample shows an example of neural sparse search. Although there is no overlap token in original query and document, but this model performs a good match. 

## Detailed Search Relevance

<div style="overflow-x: auto;">
    
| Model | Average | Trec Covid | NFCorpus | NQ | HotpotQA | FiQA | ArguAna | Touche | DBPedia | SCIDOCS | FEVER | Climate FEVER | SciFact | Quora |
|-------|---------|------------|----------|----|----------|------|---------|--------|---------|---------|-------|---------------|---------|-------|
| [opensearch-neural-sparse-encoding-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1) | 0.524 | 0.771 | 0.360 | 0.553 | 0.697 | 0.376 | 0.508 | 0.278 | 0.447 | 0.164 | 0.821 | 0.263 | 0.723 | 0.856 |
| [opensearch-neural-sparse-encoding-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v2-distill) | 0.528 | 0.775 | 0.347 | 0.561 | 0.685 | 0.374 | 0.551 | 0.278 | 0.435 | 0.173 | 0.849 | 0.249 | 0.722 | 0.863 |
| [opensearch-neural-sparse-encoding-doc-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v1) | 0.490 | 0.707 | 0.352 | 0.521 | 0.677 | 0.344 | 0.461 | 0.294 | 0.412 | 0.154 | 0.743 | 0.202 | 0.716 | 0.788 |
| [opensearch-neural-sparse-encoding-doc-v2-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill) | 0.504 | 0.690 | 0.343 | 0.528 | 0.675 | 0.357 | 0.496 | 0.287 | 0.418 | 0.166 | 0.818 | 0.224 | 0.715 | 0.841 |
| [opensearch-neural-sparse-encoding-doc-v2-mini](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini) | 0.497 | 0.709 | 0.336 | 0.510 | 0.666 | 0.338 | 0.480 | 0.285 | 0.407 | 0.164 | 0.812 | 0.216 | 0.699 | 0.837 |
| [opensearch-neural-sparse-encoding-doc-v3-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill) | 0.517 | 0.724 | 0.345 | 0.544 | 0.694 | 0.356 | 0.520 | 0.294 | 0.424 | 0.163 | 0.845 | 0.239 | 0.708 | 0.863 |
| [opensearch-neural-sparse-encoding-doc-v3-gte](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte) | 0.546 | 0.734 | 0.360 | 0.582 | 0.716 | 0.407 | 0.520 | 0.389 | 0.455 | 0.167 | 0.860 | 0.312 | 0.725 | 0.873 |

</div>

## License

This project is licensed under the [Apache v2.0 License](https://github.com/opensearch-project/neural-search/blob/main/LICENSE).

## Copyright

Copyright OpenSearch Contributors. See [NOTICE](https://github.com/opensearch-project/neural-search/blob/main/NOTICE) for details.
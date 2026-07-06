# Data directory

Large data files are **not** committed to this repository due to size.

## Required files

Place the following files inside `data/kg_exports/` before running the pipeline:

| File | Description |
|------|-------------|
| `patent_full_table.parquet` | Main patent feature table (91 K U.S. EV patents, 2014–2023). Includes KG node2vec embeddings (`kg_emb_0` … `kg_emb_63`), Sentence-BERT text embeddings (`txt_emb`), metadata fields, and outcome labels. |
| `rels_cites.txt` | Citation edges (src\_id, dst\_id) |
| `rels_invented_by.txt` | Inventor edges (patent → inventor) |
| `rels_assigned_to.txt` | Assignee edges (patent → assignee) |

## Data sources

The patent data was collected from:
- **USPTO Patent Grant Full-Text Data** (bulk downloads via [PatentsView](https://patentsview.org))
- **USPTO Maintenance Fee Events** (bulk download)
- **PATSTAT** (European Patent Office) — triadic family membership
- **OECD Patent Quality Indicators**

CPC-subclass filtering (B60L, B60K, H01M, H02J, …) and keyword matching on titles and abstracts were used to identify electric-vehicle patents.

## KG construction

The heterogeneous knowledge graph was built in **Neo4j**. Node and edge export scripts are available on request. The node2vec embeddings were trained on the temporally masked heterogeneous graph (edges observable on or before each patent's grant date).

## Genderize API

First-name gender estimates were obtained via the [Genderize.io](https://genderize.io) API. The mapping files are included in `data/` as reference.

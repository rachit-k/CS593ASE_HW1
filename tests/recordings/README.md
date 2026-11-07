# Recorded API responses

Real responses, recorded once on 2026-11-06 and replayed by the tests, so the fakes match what these services actually send. The tests never go online.

| File | Request | Status |
|---|---|---|
| `arxiv_three_papers.xml` | `id_list=2404.01484,1706.03762,hep-th/9901001` (arXiv returned them in a different order) | 200 |
| `arxiv_keyword_search.xml` | `search_query=all:code AND all:review AND all:automation`, newest first, 2 results | 200 |
| `arxiv_no_results.xml` | a keyword search that matches nothing | 200 |
| `arxiv_unknown_id.xml` | `id_list=2401.99999`: a valid format, but no such paper | 200 |
| `arxiv_bad_id.xml` | `id_list=notanid`: an error entry inside a normal-looking feed | 400 |
| `semantic_scholar_batch.json` | batch lookup of `1706.03762`, `9999.99999` (unknown, so `null`), `hep-th/9901001` | 200 |
| `semantic_scholar_429.json` | the same lookup, refused by the no-key rate limit | 429 |

`real_papers.json` holds 29 real papers, recorded the same way. For each one: the opening text of the PDF up to just past its abstract, as `pdfs.extract` produced it, the PDF's embedded title and author fields, and the title, abstract, and authors from arXiv's own metadata. Email addresses and phone numbers are replaced with placeholders. In one author field, `someone@example.org` stands in for the address, because the app must handle an address there. The papers cover common layouts, including conference papers, IEEE, Springer's "Abstract We..." template, and papers with no abstract heading at all. `test_real_papers.py` uses arXiv's metadata as the right answer.

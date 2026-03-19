

In this directory we explain how to obtain the full SQUiC annotations. If you want to download **NeWMe** to run all WMN-related analyses, please visit [this repository](https://github.com/ainagari/WMNindicator_detection/tree/main/obtaining_data).


# Scare Quotes in Conversation - Reconstruction from standoff annotations

This directory contains the standoff annotation release. Because the underlying data are Reddit posts from the
[CMV WinningArguments corpus](https://convokit.cornell.edu/documentation/winning.html), we are not distributing the original text directly. Instead, we provide:

- **standoff annotations** (annotated labels)
- **positional information** (character offsets pointing to the original text)
- a **reconstruction script** that downloads the corpus via
  [convokit](https://convokit.cornell.edu/) and rebuilds the full dataset. 


The code and readme in this directory was mainly written with the help of Claude 4.6, with human supervision and verification.

---

## Files

| File | Description |
|---|---|
| `standoff_annotations.json` | Annotations for 2 522 instances. Includes 22 instances annotated as "Discarded". Labels only, no Reddit text |
| `quotations_winningargs_info_public.tsv` | Positional and lexical metadata for all 9 260 candidate quotes extracted from WinningArguments (of which we annotated a subset) |
| `reconstruct_from_standoff.py` | Script to rebuild the full dataset from the above files + convokit |


## How to reconstruct the full dataset

To reconstruct the full dataset with text and annotations and be able to run the analyses in the aprent directory, you should:

### 1. Install dependencies

```bash
pip install convokit pandas
```

> convokit will also install its own dependencies (spaCy, etc.) automatically.

### 2. Run the reconstruction script

```bash
python reconstruct_from_standoff.py
```

On the first run, convokit downloads the WinningArguments corpus (~400 MB). Subsequent runs use the cached copy.

The script creates `annotated_corpus.json` in the parent directory, so that you can readily run all analyses using the notebooks and scripts there. 

---

## File Formats

### `standoff_annotations.json`

A JSON array. Each element represents one annotated quotation instance:

```json
{
  "data": {
    "id":                            "t3_334q3e_t1_cqvwx9u_10",
    "quoted_passage":                "owned",
    "item_order":                    5328,
    "used_previously_in_subthread":  false
  },
  "annotations": [
    {
      "result": [
        {
          "from_name": "quote-type-broad",
          "to_name":   "dialogue",
          "type":      "choices",
          "value":     { "choices": ["Scare quotes"] }
        },
        {
          "from_name": "quote-type-fine1",
          "to_name":   "dialogue",
          "type":      "choices",
          "value":     { "choices": ["SQ: Usage"] }
        }
      ]
    }
  ]
}
```

#### `data` fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Item identifier: `{conv_id}_{utt_id}_{quote_index}` (e.g. `t3_334q3e_t1_cqvwx9u_10`). The first two `_`-delimited components are the convokit conversation ID; components 3–4 are the convokit utterance ID; the trailing number distinguishes multiple quotes from the same utterance. |
| `quoted_passage` | string | The short quoted string (1–3 words) as it appears in the utterance. This is the only Reddit-derived text retained in the standoff release. |
| `item_order` | int | Unique integer key linking this item to a row in the TSV. |
| `used_previously_in_subthread` | bool | Whether the same string appeared verbatim (case-insensitive) in an ancestor utterance in the same reply chain. Computed automatically during corpus extraction. |

#### `annotations[].result` entries

Each result entry has a `from_name` field identifying the annotation dimension:

| `from_name` | Description |
|---|---|
| `quote-type-broad` | High-level quotation type (see *Annotation Schema* below) |
| `quote-type-fine1` | Fine-grained quotation subtype |

---

### `quotations_winningargs_info_public.tsv`

Tab-separated file covering **all 9 260 candidate quoted phrases** extracted from WinningArguments (before any filtering or annotation). The annotated subset is identified by `kept_for_annotation == True`.

| Column | Description |
|---|---|
| `quoted_passage` | The quoted string |
| `ngram` | Number of words in the quoted string (`1`, `2`, `3`) |
| `utt_id` | convokit utterance ID (e.g. `t1_cqvwx9u`) |
| `conv_id` | convokit conversation ID (e.g. `t3_334q3e`) |
| `author` | Speaker identifier in annotation format: `username_##_utt_id_##_rt-parent_id` |
| `start` | Character offset of the quote's **opening** position within the utterance text (after citation markup; see *Citation Markup* below). This is the position of the first character of `quoted_passage` itself, i.e. the character **after** the opening quote mark. |
| `end` | Character offset one past the quote's **closing** position (exclusive). `utterance_text[start:end]` yields `quoted_passage`. |
| `conv_in_newme` | Whether this conversation also appears in the NeWMe corpus |
| `sign` | Quote mark used: `"` (double), `'` (single), or `-` for *so-called* |
| `item_order` | Integer linking to `standoff_annotations.json` |
| `used_previously_in_subthread` | Same as in the standoff JSON |
| `id` | Full item identifier (same as `data.id` in the standoff JSON) |
| `kept_for_annotation` | Whether this candidate was included in the annotation task |

---

### `annotated_corpus.json` (generated)

A JSON array with one entry per annotated instance. Each entry contains:

```json
{
  "data": {
    "id":                            "t3_334q3e_t1_cqvwx9u_10",
    "quoted_passage":                "owned",
    "item_order":                    5328,
    "used_previously_in_subthread":  false,
    "sign":                          "'",
    "ngram":                         "1",

    "conversation": [
      { "utt_id": "t3_334q3e_title", "speaker": "TITLE",
        "text": "CMV: Gun rights are actually a liberal concept",
        "reply_to": null },
      { "utt_id": "t1_cqhpv4a", "speaker": "CalicoZack",
        "text": "A couple of points: ...",
        "reply_to": "t3_334q3e" },
      { "utt_id": "t1_cqvwx9u", "speaker": "DrTTmenxion",
        "text": "... 'owned' ...",
        "reply_to": "t1_cqhpv4a",
        "is_annotated_utterance": true,
        "quote_start": 123,
        "quote_end":   128 }
    ],
    "annotated_utt_index": 2,

    "html_conversation": "<div …>…</div>",
    "html_quote_start":  456,
    "html_quote_end":    461
  },
  "annotations": [ ... ]
}
```

---

#### About position information

There are two complementary sets of position fields pointing to the samequoted string from different angles.

* **Plain-text positions (recommended in general)**

| Field | Description |
|---|---|
| `conversation[annotated_utt_index]["quote_start"]` | Character offset of the first character of `quoted_passage` within that utterance's `text` (citation markers already stripped) |
| `conversation[annotated_utt_index]["quote_end"]` | Corresponding end offset (exclusive) |
| `data.annotated_utt_index` | Index into `data.conversation` of the utterance that contains the quote |


* **HTML positions (only for Label Studio use)**

| Field | Description |
|---|---|
| `data.html_quote_start` | Character offset of the first character of the highlighted quoted passage within `data.html_conversation` |
| `data.html_quote_end` | Corresponding end offset (exclusive) |

The highlighted text in the HTML is wrapped in:
```html
<span style='color: #1924F7; font-weight: bold;'>quoted_passage</span>
```

Reddit citation lines (lines beginning with `&gt;`, representing quoted text from a previous post) are wrapped with `[STA-CITE]`/`[END-CITE]` to facilitate their identification when annotating in LabelStudio:

```
[STA-CITE]&gt; original citation text

[END-CITE]
```

These markers appear only in `data.html_conversation`. (the HTML representation). They are stripped from all utterance texts in `data.conversation`.


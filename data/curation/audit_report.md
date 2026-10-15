# Curation record

Every number below is recomputed from the parquet files shipped in `../original`
and `../curated` by `build_summary.py`, not copied from the paper.

Items are aligned between the two releases by `extra_info.orig_idx`, which gives
the positional index of the item in the upstream release. Alignment by question
text would be wrong: a rewritten item has a modified option and therefore a
modified question string.

## Training split

| Domain | Original | Unchanged | Relabelled | Rewritten | Removed | Curated |
| --- | --- | --- | --- | --- | --- | --- |
| Biology | 450 | 447 | 0 | 0 | 3 | 447 |
| Chemistry | 1890 | 1856 | 1 | 4 | 29 | 1861 |
| Materials Science | 841 | 801 | 1 | 0 | 39 | 802 |
| Physics | 720 | 524 | 26 | 64 | 106 | 614 |
| **Total** | **3901** | **3628** | **28** | **68** | **177** | **3724** |

## Test split

| Domain | Original | Unchanged | Relabelled | Rewritten | Removed | Curated |
| --- | --- | --- | --- | --- | --- | --- |
| Biology | 50 | 50 | 0 | 0 | 0 | 50 |
| Chemistry | 210 | 208 | 1 | 0 | 1 | 209 |
| Materials Science | 94 | 93 | 0 | 0 | 1 | 93 |
| Physics | 80 | 50 | 4 | 7 | 19 | 61 |
| **Total** | **434** | **401** | **5** | **7** | **21** | **413** |

## Categories

* **Unchanged** - question, options and ground-truth label all identical to the
  upstream release.
* **Relabelled** - only the ground-truth letter differs; the question and all
  four options are byte-identical to upstream.
* **Rewritten** - the text of the labelled option was replaced; the question
  stem, the option positions and the ground-truth letter are unchanged. The
  replaced text is recorded in `changed_items.csv` under
  `replaced_option_text`.
* **Removed** - present upstream, absent from the curated release.

## Files

* `summary.csv` - the two tables above in machine-readable form.
* `changed_items.csv` - one row per relabelled or rewritten item
  (108 rows), with the domain, split, action, the original and curated
  answer letters, the replaced option text where applicable, and the full
  question text.
* `removed_items.csv` - one row per removed item (198 rows), with the
  domain, split, original answer letter and full question text.

## Fields retained in the curated parquet files

Each curated item keeps the audit decision as structured fields:
`audit_verdict` (`ok`, `wrong_key`, `key_not_in_options`), `audit_confidence`,
`audit_orig_ground_truth` and `audit_orig_option_text`. Free-text notes taken
during the audit are not included in this release. Fields such as `hint_l1`,
`hint_l3`, `hint_family`, `elo` and `achievement_prior` are inherited from the
dataset construction pipeline and are not read by the training code; they are
kept so that the shipped files are the files the runs actually read.

# Data

The scientific-reasoning experiments use the reasoning subset (L3) of
SciKnowEval, covering undergraduate-level Biology, Chemistry, Materials Science
and Physics. Both the release as distributed upstream and the curated release
used in the paper are included, so that every change can be inspected.

```
original/sciknoweval-<domain>/{train,test}.parquet   upstream release, unmodified
curated/sciknoweval-<domain>/{train,test}.parquet    the splits used in the paper
curation/summary.csv                                 per-domain, per-split counts
curation/changed_items.csv                           every relabelled or rewritten item
curation/removed_items.csv                           every removed item
curation/audit_report.md                             the counts as tables, with definitions
curation/build_summary.py                            recomputes all of the above from the parquets
```

`python3 curation/build_summary.py` recomputes the counts directly from the two
parquet trees; the numbers in `audit_report.md` and in the appendix table of the
paper are the output of that script, not transcribed by hand.

## Row format

Both releases use the same schema: `prompt` (a system message and the question),
`data_source`, `ability`, `reward_model` (with `ground_truth`, the answer letter)
and `extra_info`. A curated row additionally carries the audit decision as
structured fields: `audit_verdict` (`ok`, `wrong_key`, `key_not_in_options`),
`audit_confidence`, `audit_orig_ground_truth`, `audit_orig_option_text`, and
`orig_idx`, the positional index of the item in the upstream release. It also
carries `hint_l1`, `hint_l3`, `hint_family`, `elo` and `achievement_prior`,
inherited from the dataset construction pipeline. None of these is read by the
training or evaluation code, which matters because `hint_l3` states the correct
answer; they are kept only so that the shipped files are the files the runs
actually read.

`orig_idx` is the correct key for comparing the two releases. Comparing by
question text is not: a rewritten item has one option replaced and therefore a
different question string, which would make it look like a removal plus an
unrelated insertion.

## Non-ASCII characters

The upstream questions contain scientific notation that is not representable in
ASCII: the angstrom sign, Greek letters, superscripts and subscripts, the degree
sign, the minus sign and two middle-dot characters used as separators in
formulas. These characters are preserved exactly, since removing them would
corrupt the questions.

## Interactive-agent data

The ALFWorld, WebShop and Search task data are third-party distributions with
their own licences and are not redistributed here. The Search training corpus is
drawn from NQ and HotpotQA; the Search score reported in the paper averages the
exact-match score over all seven QA datasets of the evaluation set.

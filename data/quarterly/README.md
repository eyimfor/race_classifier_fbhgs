# Quarterly venture funding to Black-founded U.S. startups, 2015Q1 to 2024Q4

This workbook tracks how much venture capital goes to Black-founded startups in the
United States, quarter by quarter from 2015 through 2024. An annual sheet and a notes
sheet sit alongside the quarterly one. Each row answers the same question two ways.
The first count includes every startup my classifier identifies as Black-founded. The
second includes only the startups that carry a Black Leadership tag in Crunchbase's
Diversity Spotlight, which is what you would see if you relied on self-reported tags
alone. Both come as a share of all venture dollars and as a share of all venture
rounds. The distance between them is the funding to Black founders that a tag-only
view misses.

Source: Yimfor, Emmanuel. "The Invisible Majority: Selection Bias in Self-Reported
Data." Working paper. Data derived from the Crunchbase bulk export dated
2025-01-26 (https://www.crunchbase.com) and the author's founder race
classification. This workbook contains aggregates only, no company-level Crunchbase data.

## Sample

The startups are U.S. companies founded between 2000 and 2022 with at least one
founder, owner, or CEO in Crunchbase's job records whose race I classified. The rounds
are seed, Series A through H, venture, and series-unknown rounds announced between
January 2015 and December 2024 with a positive dollar amount. Grants, debt,
crowdfunding, private equity, and post-IPO rounds are out. That leaves 69,875
rounds by 36,591 startups. 1,413 of those startups are Black-founded by
my classification, and 479 of them also carry the Diversity Spotlight tag.
Quarters follow the round announcement date. 2024Q4 is the last full quarter in the
export.

## The two measures

Classification. A startup counts as Black-founded if at least one founder, owner, or
CEO is classified as Black by the image classifier described in the paper, with
clerical review. This is the benchmark.

Diversity Spotlight. A startup counts only if it carries the Black Leadership tag in
Crunchbase's Diversity Spotlight and is also Black-founded by the classification. Tag
status comes from the Spotlight lists I collected in 2020 and 2023. This is what you
get if you rely on self-reported tags.

## Columns

| Column | What it is |
|---|---|
| Quarter or Year | Calendar quarter (YYYYQn) or year of the round announcement |
| All venture rounds | Rounds in the sample, all startups |
| All venture dollars (USD) | Dollars raised in those rounds |
| Rounds to Black-founded startups (classification) | Rounds by startups that are Black-founded by the classification |
| Dollars to Black-founded startups (classification) | Dollars in those rounds |
| Rounds to Black-founded startups (Diversity Spotlight) | Rounds by startups that are Black-founded and tagged in Spotlight |
| Dollars to Black-founded startups (Diversity Spotlight) | Dollars in those rounds |
| Black-founded share of dollars (classification) | Black-founded dollars by the classification, divided by all venture dollars |
| Black-founded share of dollars (Diversity Spotlight) | The same share, counting only tagged startups |
| Black-founded share of rounds (classification) | Black-founded rounds by the classification, divided by all venture rounds |
| Black-founded share of rounds (Diversity Spotlight) | The same share, counting only tagged startups |

A note on the dollars. They are raw sums of what Crunchbase reports for each round,
with no trimming, so one very large round can swing a quarter. The headline numbers
in the paper use founder-level totals winsorized at the 1st and 99th percentiles,
which is why they do not match ratios of these sums exactly.

## How to cite

```
@unpublished{yimfor2026invisible,
  author = {Yimfor, Emmanuel},
  title  = {The Invisible Majority: Selection Bias in Self-Reported Data},
  year   = {2026},
  note   = {Working paper. Quarterly data: https://github.com/eyimfor/race_classifier_fbhgs/tree/main/data/quarterly}
}
```

The data are released under CC BY-NC 4.0. The underlying data come from Crunchbase
(https://www.crunchbase.com). The code that builds this workbook is in the paper's
replication package.

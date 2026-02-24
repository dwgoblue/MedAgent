# Evidence Contract Templates

Each run defines required verification categories and stop criteria before synthesis.

## Common categories

- `variant`: genotype interpretation claims
- `gene_disease`: mechanistic or association claims
- `drug`: pharmacogenomic or contraindication claims
- `guideline`: recommendation-policy claims
- `trial`: active-trial relevance claims
- `imaging`: image/report-derived claims
- `lab`: lab/vital interpretation claims
- `epidemiology`: prevalence/risk claims

## Contract rules

- Every `must_verify=true` claim must include retrieval evidence or explicit caveat.
- Weak evidence forces uncertainty language and risk-aware de-prioritization.
- Failed evidence removes or rewrites the claim.

## Output requirement

Final output must include a compact evidence table with source IDs (for example PMIDs, trial IDs, or BioMCP identifiers).

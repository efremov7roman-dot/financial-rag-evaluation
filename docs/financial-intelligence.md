# Financial intelligence

PHASE 4 adds finance-specific routing and exact calculations without using an LLM to infer intent
or arithmetic.

## Query routes

- Narrative questions use hybrid Dense + BM25 RRF, followed by company, filing type, and reporting
  period filters. Results are evidence passages rather than generated prose.
- Structured fact questions resolve a supported metric to explicit US-GAAP concepts and search the
  local inline-XBRL index.
- Calculation questions require every operand to be present as a compatible XBRL fact. Percentage
  change, absolute difference, and gross margin use `Decimal` arithmetic.

The router recognizes companies already present in the corpus, explicit CIKs/tickers, calendar
years, `10-K`/annual wording, and `10-Q`/quarter wording. When the corpus contains exactly one
company, that company is used as the unambiguous default.

## Supported metric aliases

The initial transparent mapping covers revenue/sales, net income, gross profit, operating income,
assets, liabilities, operating cash flow, and research and development expense. Gross margin uses
gross profit divided by revenue. Unsupported metrics remain narrative queries; they are never
silently mapped to a guessed XBRL concept.

## Evidence and refusal behavior

Every numeric claim cites the source URL, accession number, XBRL concept, reporting period, value,
and unit. Calculations cite every operand. Narrative results cite section and chunk ID. If a period,
fact, denominator, or filtered passage is missing, the response status is
`insufficient_evidence` and no unsupported answer is produced.

## Reproduction

```powershell
financial-rag build-fact-index
financial-rag ask-financial "What was Apple revenue in 2024 annual filing?"
financial-rag ask-financial "What was Apple's gross margin in 2024 annual filing?"
financial-rag ask-financial "What supply chain risks did Apple discuss in 2024?"
```

The current nine-filing Apple corpus produces 6,169 normalized numeric facts. Observed FY2024
checks are $391.035 billion revenue, 2.02% revenue growth from FY2023, and 46.21% gross margin.
These figures are direct consequences of the cited filing facts, not benchmark claims.

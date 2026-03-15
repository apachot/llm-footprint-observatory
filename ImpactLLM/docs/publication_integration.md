# Publication Integration

## Positioning in Publication 1

The first publication can claim three concrete research outputs:

- a quantified literature review on environmental impacts of LLMs;
- an open dataset of extracted values with source-level traceability;
- a machine-accessible interface through both an HTTP API and an MCP server.

## Suggested Contribution Statement

Suggested wording:

> We complement the literature review with an open dataset of quantified environmental indicators extracted from the reviewed publications. Each value is stored with its unit, analytical scope, system boundary, and precise source locator. To support reproducibility and reuse, the dataset is exposed through a local HTTP API and a Model Context Protocol server.

## Suggested Methods Paragraph

Suggested wording:

> For each publication retained in the review, we extract one record per quantified indicator. Each record stores the reported value, its unit, the analytical phase (training, inference, infrastructure, or lifecycle), the type of evidence (measured, calculated, modeled, projected, or estimated), the geographical scope, the system boundary, and a source locator pointing to the exact page, table, section, or paragraph used for extraction. The canonical dataset is distributed as CSV and JSON, and is made machine-queryable through an HTTP API and an MCP server.

## Suggested Reproducibility Paragraph

Suggested wording:

> Beyond the manuscript itself, we publish the extracted evidence base as structured open data. This design separates the narrative layer of the paper from the reusable evidence layer, making it easier to update, audit, and compare future studies on LLM environmental impacts.

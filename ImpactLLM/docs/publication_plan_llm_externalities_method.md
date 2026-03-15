# Publication Plan: LLM Externalities Estimation for Software Systems

## Working Title

Estimating the Environmental Externalities of Software Systems Integrating Large Language Models: An Open Data and Factor-Based Method

## Research Problem

Organizations increasingly integrate LLMs into software products, but they lack a reliable method for estimating the associated environmental externalities at feature, service, and annual product levels.

## Main Contribution

The paper would contribute:

- an open dataset of quantified environmental factors extracted from the literature;
- a method for selecting and contextualizing these factors;
- an estimation framework for feature-level and software-level environmental accounting;
- a machine-actionable implementation through an API and an MCP server.

## Suggested Paper Structure

### 1. Introduction

- Why prompt-level intuition is insufficient
- Why software carbon accounting needs an LLM-specific method
- Research questions

### 2. Related Work

- LLM environmental impact literature
- Green software and software carbon accounting
- Limits of current factor availability

### 3. Open Dataset of Environmental Factors

- Dataset scope
- Extraction rules
- Source traceability
- Coverage and limitations

### 4. Estimation Method

- Activity model
- Factor selection hierarchy
- Contextual adjustment rules
- Software-system boundary
- Annualization

### 5. Uncertainty and Comparability

- Confidence levels
- Comparison families
- Derived versus observed values
- Exclusions

### 6. Implementation

- API
- MCP tools
- Example estimation workflow

### 7. Evaluation on Representative Scenarios

- Simple chat feature
- RAG assistant
- Internal coding assistant
- Batch generation workflow

### 8. Discussion

- Practical usefulness
- Limits for audited reporting
- Required future data disclosure by providers

### 9. Conclusion

- Summary
- Reproducibility and open data angle

## Suggested Research Questions

- How can heterogeneous LLM environmental factors be made operational for software impact estimation?
- Which metadata are necessary to avoid invalid comparisons across studies?
- What level of uncertainty remains when estimating feature-level impacts from published literature?
- How can API and MCP interfaces improve reuse of environmental evidence for responsible AI practices?

## Evaluation Strategy

Use 3 to 5 scenarios with increasing complexity:

- one direct chat feature;
- one feature with retrieval;
- one feature with multi-step orchestration;
- one self-hosted scenario if data are sufficient.

For each scenario, report:

- selected factors;
- assumptions;
- low, central, high estimates;
- uncertainty level;
- reduction levers.

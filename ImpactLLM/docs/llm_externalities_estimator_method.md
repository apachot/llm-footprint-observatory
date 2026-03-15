# LLM Externalities Estimator Method

## Objective

Build a reusable estimation method for the following practical question:

> Given a prompt, a target LLM, and a software usage scenario, estimate the environmental externalities associated with this use.

The method is designed for two complementary uses:

- operational estimation for one prompt, one feature, or one product flow;
- environmental accounting for software systems integrating one or more LLM services.

## Core Principle

The estimator must not output a single decontextualized figure. It must output:

- a central estimate;
- a low and high range when possible;
- the impact factors used;
- the assumptions used;
- the applicability domain of the estimate.

The method is based on this general logic:

`impact = activity data x impact factor x contextual adjustment`

## Conceptual Model

The estimator is organized in four layers.

### Layer 1. LLM Service Activity

This layer describes the direct LLM usage.

Core variables:

- `provider`
- `model_id`
- `deployment_mode`
- `request_type`
- `input_tokens`
- `output_tokens`
- `requests_count`
- `streaming`
- `tool_calls_count`
- `multimodal_flags`

Typical deployment modes:

- remote API
- managed cloud endpoint
- self-hosted model
- edge or on-premise

### Layer 2. Execution Context

This layer describes where and how the LLM runs.

Core variables:

- `country`
- `grid_carbon_intensity_gco2_per_kwh`
- `water_intensity_l_per_kwh`
- `datacenter_pue`
- `hardware_profile`
- `utilization_rate`
- `time_scope`

This layer is needed because the same model can have very different environmental footprints depending on the electricity mix, datacenter overhead, and hardware allocation.

### Layer 3. Impact Factors

This layer links the activity to empirical factors from the dataset.

Examples:

- `Wh/prompt`
- `gCO2e/prompt`
- `mL/prompt`
- `kWh/page`
- `gCO2e/page`
- `L/page`
- `TWh/year`

Each factor must carry:

- a source;
- a source locator;
- a system boundary;
- a data type;
- an uncertainty level;
- an applicability domain.

### Layer 4. Software System Boundary

This layer extends the estimate from the LLM call to the full software service.

Additional components:

- orchestration or agent runtime;
- embeddings generation;
- vector database queries;
- storage and logs;
- application servers;
- monitoring and observability;
- client-side overhead when relevant.

This layer is required for software carbon accounting. A software feature using an LLM cannot be assessed only through the direct prompt impact.

## Data Schema for the Estimator

### Input Payload

```json
{
  "scenario_id": "feature_support_chat_q1",
  "provider": "openai",
  "model_id": "gpt-4-class",
  "deployment_mode": "remote_api",
  "request_type": "chat_generation",
  "input_tokens": 1200,
  "output_tokens": 350,
  "requests_count": 1,
  "country": "FR",
  "grid_carbon_intensity_gco2_per_kwh": 40,
  "water_intensity_l_per_kwh": 0.4,
  "datacenter_pue": 1.2,
  "time_scope": "2026-03",
  "software_components": [
    {
      "component_type": "vector_search",
      "quantity": 1,
      "unit": "query"
    },
    {
      "component_type": "application_server",
      "quantity": 2,
      "unit": "seconds_cpu"
    }
  ]
}
```

### Factor Record

Each factor selected from the literature should follow a structure like:

```json
{
  "factor_id": "elsworth2025_prompt_energy",
  "metric_name": "prompt_energy",
  "metric_value": 0.24,
  "metric_unit": "Wh/prompt",
  "phase": "inference",
  "impact_category": "energy",
  "system_boundary": "comprehensive method",
  "data_type": "measured",
  "applicability_domain": {
    "provider_type": "hyperscaler production",
    "request_type": "chat prompt",
    "scope_limitations": "median prompt at Google scale"
  },
  "uncertainty_level": "medium",
  "source_key": "elsworth2025",
  "source_locator": "Table 2, p. 7"
}
```

### Output Payload

```json
{
  "scenario_id": "feature_support_chat_q1",
  "estimate_level": "request",
  "results": {
    "energy_wh": {
      "low": 0.24,
      "central": 0.45,
      "high": 2.9
    },
    "carbon_gco2e": {
      "low": 0.01,
      "central": 0.02,
      "high": 0.12
    },
    "water_ml": {
      "low": 0.26,
      "central": 0.50,
      "high": 5.0
    }
  },
  "selected_factors": [
    "elsworth2025_prompt_energy",
    "elsworth2025_prompt_carbon",
    "elsworth2025_prompt_water",
    "epri2024_chatgpt_query"
  ],
  "assumptions": [
    "No direct provider telemetry available",
    "Prompt classified as text-only chat generation",
    "French electricity mix used for contextual carbon adjustment"
  ],
  "uncertainty_level": "high",
  "applicability_note": "Cross-provider transfer from published literature, suitable for screening, not for audited product declarations"
}
```

## Estimation Logic

### Step 1. Select an Impact Family

The first step is to identify the appropriate comparison family.

Possible families:

- prompt-level inference
- query-level inference
- page-generation inference
- training event
- annual infrastructure
- lifecycle creation

For software practices, the relevant default family is usually prompt-level or query-level inference.

### Step 2. Select Candidate Factors

Candidate factors are selected by:

- matching the phase;
- matching the unit of analysis;
- matching the deployment mode when possible;
- filtering on source quality and system boundary.

Priority order:

1. measured production factors;
2. measured benchmark factors;
3. calculated comparative factors;
4. estimated macro factors.

### Step 3. Convert to a Common Activity Unit

A common activity unit must be chosen before estimation.

Recommended units:

- per request
- per 1,000 requests
- per user session
- per feature usage
- per month
- per year

If the factor is published in `Wh/prompt`, then:

`energy_total_Wh = requests_count x factor_Wh_per_prompt`

If the factor is published in `kWh/page`, then:

`energy_total_kWh = pages_generated x factor_kWh_per_page`

No conversion should be performed when the publication does not justify equivalence between units.

### Step 4. Contextual Adjustments

Contextual adjustment can be applied only when the literature and the metadata support it.

Typical adjustments:

- electricity-mix adjustment for carbon:
  `carbon = energy_kWh x grid_carbon_intensity`
- water-intensity adjustment for electricity-related water use:
  `water = energy_kWh x water_intensity`
- datacenter overhead adjustment:
  `energy_total = IT_energy x PUE`

These adjustments must be tagged as derived values, not raw literature values.

### Step 5. Add Software-System Overheads

A responsible accounting method must add software overheads that are not part of the LLM factor itself.

Examples:

- retrieval query energy;
- embeddings calls;
- orchestration loops;
- logging and storage;
- CPU service time;
- fallback or retry logic.

At this stage, total feature impact becomes:

`feature_impact = llm_direct_impact + software_overheads + optional infrastructure_allocation`

### Step 6. Aggregate Over Time

For product accounting, the estimate must be annualized:

`annual_impact = unit_impact x number_of_feature_uses_per_year`

This is the level relevant for:

- software carbon footprints;
- responsible AI reporting;
- eco-design prioritization.

## Uncertainty Framework

The estimator should not hide uncertainty. It should classify every estimate.

### Low Uncertainty

- same unit of analysis;
- same deployment mode;
- measured factor;
- clear system boundary;
- direct activity data available.

### Medium Uncertainty

- same phase and same family of use;
- measured or calculated factor;
- partial context transfer;
- some assumptions on infrastructure.

### High Uncertainty

- cross-provider transfer;
- estimated or projected factors;
- unclear unit equivalence;
- unknown hardware or unknown datacenter context.

## Responsible Use Rules

The tool should enforce these rules:

- never compare training and inference factors directly without re-scoping;
- never convert prompt factors into token factors unless the source explicitly supports it;
- always display the source and source locator;
- always distinguish observed values from derived values;
- never output a single precise number when the evidence is weak.

## Use Cases

### Use Case 1. Prompt-Level Estimate

Question:

> Here is my prompt with this LLM. Estimate its environmental externalities.

Expected output:

- direct impact estimate per request;
- low, central, high range;
- source-backed factor list;
- recommendation to reduce impact.

### Use Case 2. Feature-Level Estimate

Question:

> Estimate the annual impact of my support chatbot feature.

Expected output:

- unit impact per interaction;
- annualized impact;
- contribution of LLM calls versus software overheads;
- main reduction levers.

### Use Case 3. Software Carbon Accounting

Question:

> Integrate the LLM feature into the software carbon footprint.

Expected output:

- system boundary definition;
- activity model;
- factor selection rationale;
- annual totals by component;
- uncertainty and exclusions.

## Research Contribution

This method supports a new scientific contribution beyond the literature review:

- an open dataset of environmental factors for LLMs;
- a factor-selection and contextualization framework;
- a practical estimation method for software carbon accounting with LLM integration;
- a machine-actionable implementation path through API and MCP.

## Next Implementation Steps

1. Extend the dataset schema with estimation-specific metadata:
   - `unit_of_analysis`
   - `source_type`
   - `uncertainty_level`
   - `applicability_domain`
   - `raw_or_derived`
2. Add a factor-selection module on top of the dataset.
3. Add an estimation endpoint to the API.
4. Add an MCP tool such as `estimate_externalities`.
5. Validate the method on 3 to 5 representative software scenarios.

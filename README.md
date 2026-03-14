# ImpactLLM

ImpactLLM is an open tool for exploring and estimating the environmental footprint of large language models.

The project is designed as a transparent screening tool rather than a black-box score. It builds on indicators published in the scientific and technical literature, links each estimate back to its sources, keeps assumptions visible, and favors a traceable, auditable calculation logic.

## What This Repository Contains

This repository brings together two main components:

- `llm-environment-opendata/`: the open dataset, local HTTP API, MCP server, web calculation interface, and estimation logic
- `llm-environment-opendata-paper/`: the scientific paper, bibliography, and related materials

ImpactLLM is useful for:

- exploring source-backed data on the environmental impacts of LLMs
- producing comparable estimates for uses, features, or inference scenarios
- documenting the assumptions, system boundaries, and country factors used in the calculation
- supporting software design, research, and decision-making discussions

## Positioning

ImpactLLM estimates are research-oriented and intended for decision support. They do not mechanically copy figures from one paper to another: energy is retained from documented anchors, then carbon is recalculated according to the electricity mix associated with the selected country context. The value of the tool comes precisely from the fact that assumptions remain inspectable, debatable, and tied to sources.

Our work on responsible AI emphasizes methodological rigor, traceability, and decision support in real-world contexts. It combines scientific research, product design, and operational deployment to make AI systems more transparent, more accountable, and more useful in practice.

## Quick Links

- Demo: `https://dev.emotia.com/impact-llm/`
- Paper PDF: `https://dev.emotia.com/impact-llm/downloads/llm_environment_opendata_paper.pdf`
- GitHub repository: `https://github.com/apachot/ImpactLLM`

## Get Started

To use the software stack:

- [Software README](/Users/apachot/Documents/GitHub/ImpactLLM/llm-environment-opendata/README.md)

To browse the publication materials:

- [Paper README](/Users/apachot/Documents/GitHub/ImpactLLM/llm-environment-opendata-paper/README.md)

## Technical Overview

The project combines:

- a local corpus of literature-derived data
- a local HTTP API for querying records and estimators
- an MCP server for integration with compatible agents and tools
- a web interface for describing a use case and obtaining a structured estimate

Local execution mainly relies on Python 3. An OpenAI key is only required for natural-language parsing in the web interface.

## License

ImpactLLM is distributed under the GNU GPL.

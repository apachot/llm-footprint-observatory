# ImpactLLM

ImpactLLM is an open tool for estimating the environmental footprint of LLMs.

The project is designed as a transparent screening tool rather than a black-box score. It builds on indicators published in the scientific and technical literature, links each estimate back to its sources, keeps assumptions visible, and favors a traceable, auditable calculation logic.

## What This Repository Contains

This repository brings together two main components:

- `ImpactLLM/`: the open dataset, local HTTP API, MCP server, web calculation interface, and estimation logic
- `ImpactLLM-paper/`: the scientific paper, bibliography, and related materials

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
- Paper PDF: `https://dev.emotia.com/impact-llm/downloads/ImpactLLM_paper.pdf` (`Transparent Screening for LLM Inference and Training Impacts`)
- GitHub repository: `https://github.com/apachot/ImpactLLM`

## Get Started

To use the software stack:

- [Software README](/Users/apachot/Documents/GitHub/ImpactLLM/ImpactLLM/README.md)

To browse the publication materials:

- [Paper README](/Users/apachot/Documents/GitHub/ImpactLLM/ImpactLLM-paper/README.md)

## Technical Overview

The project combines:

- a local corpus of literature-derived data
- a local HTTP API for querying records and estimators
- an MCP server for integration with compatible agents and tools
- a web interface for describing a use case and obtaining a structured estimate

Local execution mainly relies on Python 3. An OpenAI key is only required for natural-language parsing in the web interface.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

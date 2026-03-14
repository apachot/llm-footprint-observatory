# ImpactLLM

ImpactLLM est un outil ouvert pour explorer et estimer l'empreinte environnementale des grands modeles de langage.

Le projet est concu comme un outil de screening transparent, et non comme un score boite noire. Il s'appuie sur des indicateurs publies dans la litterature scientifique et technique, relie chaque estimation a ses sources, conserve les hypotheses visibles, et privilegie une logique de calcul tracable et auditable.

## Ce que contient ce depot

Ce depot rassemble deux briques principales :

- `llm-environment-opendata/` : le dataset ouvert, l'API HTTP locale, le serveur MCP, l'interface web de calcul et la logique d'estimation
- `llm-environment-opendata-paper/` : l'article scientifique, la bibliographie et les materiaux associes

ImpactLLM est utile pour :

- explorer des donnees sourcees sur les impacts environnementaux des LLM
- produire des estimations comparables pour des usages, fonctionnalites ou scenarios d'inference
- documenter les hypotheses, perimetres et facteurs pays utilises dans le calcul
- alimenter des discussions de conception logicielle, de recherche et de decision

## Positionnement

Les estimations produites par ImpactLLM sont orientees recherche et aide a la decision. Elles ne recopient pas mecaniquement des chiffres de papier a papier : l'energie est retenue a partir d'ancrages documentes, puis le carbone est recontextualise selon le mix electrique du pays selectionne. L'interet de l'outil tient precisement au fait que les hypotheses restent inspectables, discutees et rattachees aux sources.

Notre travail sur l'IA responsable met l'accent sur la rigueur methodologique, la tracabilite et l'aide a la decision dans des contextes reels. Il combine recherche scientifique, conception produit et deploiement operationnel pour rendre les systemes d'IA plus transparents, plus responsables et plus utiles en pratique.

## Acces rapides

- Demo : `https://dev.emotia.com/impact-llm/`
- Article PDF : `https://dev.emotia.com/impact-llm/downloads/llm_environment_opendata_paper.pdf`
- Depot GitHub : `https://github.com/apachot/ImpactLLM`

## Commencer

Pour utiliser la stack logicielle :

- [README logiciel](/Users/apachot/Documents/GitHub/ImpactLLM/llm-environment-opendata/README.md)

Pour consulter les materiaux de publication :

- [README article](/Users/apachot/Documents/GitHub/ImpactLLM/llm-environment-opendata-paper/README.md)

## Vue d'ensemble technique

Le projet combine :

- un corpus local de donnees issues de la litterature
- une API HTTP locale pour interroger les enregistrements et les estimateurs
- un serveur MCP pour l'integration avec des agents et outils compatibles
- une interface web permettant de decrire un cas d'usage et d'obtenir une estimation structuree

L'execution locale repose principalement sur Python 3. Une cle OpenAI n'est necessaire que pour l'analyse en langage naturel dans l'interface web.

## Licence

ImpactLLM est distribue sous licence GNU GPL.

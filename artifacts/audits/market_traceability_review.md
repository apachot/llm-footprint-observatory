# Mega revue de tracabilite du catalogue marche

- Lignes quantifiees auditees : 83
- `strict_benchmark` : 56
- `partial_data_benchmark` : 27
- `exact_in_cited_source` : 31
- `derivable_from_cited_source` : 1
- `third_party_estimate` : 24
- `partial_data_prior` : 27

## Lecture generale

- Le catalogue public ne contient plus de `project_proxy_no_url`, `internal_proxy_no_url`, `needs_cleanup` ni `manual_review` dans les bases de parametres exposees par l'application.
- Les chiffres du marche se repartissent maintenant entre valeurs exactes en source, valeurs derivables, estimations tierces explicites et priors `partial-data` derives de donneurs sourcés.
- Les champs d'entraînement restent majoritairement du screening : `training_tokens_status=screening_prior` pour 81/83 modeles, `training_regime_status=screening_prior` pour 81/83, et proxy materiel projet pour 83/83.

## Priors partial-data a surveiller

- total des priors `partial_data_prior` : 27
- `cross_provider_fallback` : 4
- `same_provider_fallback` : 6
- `same_provider_family_prior` : 4
- `same_provider_size_tier_prior` : 9
- `single_donor_family_prior` : 4

### Lignes les plus fragiles

- `gemini-3.1-flash-lite` : `cross_provider_fallback` ; donneurs `claude-3.5-haiku` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same documented size tier across the strict catalog. Donors: Claude 3.5 Haiku (22B). Retained central active-parameter basis: 22B (low/high prior: 15.4B - 28.6B).
- `grok-4.3` : `cross_provider_fallback` ; donneurs `grok-4|claude-opus-4.6|claude-opus-4.7` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider when available, then same documented size tier across the strict catalog. Donors: Grok 4 (600B), Claude Opus 4.6 (5000B), Claude Opus 4.7 (4000B). Retained central active-parameter basis: 4000B (low/high prior: 510B - 5750B).
- `gemini-3.1-flash-tts` : `cross_provider_fallback` ; donneurs `claude-3.5-haiku` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same documented size tier across the strict catalog. Donors: Claude 3.5 Haiku (22B). Retained central active-parameter basis: 22B (low/high prior: 15.4B - 28.6B).
- `claude-3-sonnet` : `cross_provider_fallback` ; donneurs `claude-2|grok-1|gpt-3.5-turbo` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider when available, then same documented size tier across the strict catalog. Donors: Claude 2 (100B), Grok 1 (78.5B), GPT-3.5 Turbo (175B). Retained central active-parameter basis: 100B (low/high prior: 66.7B - 201.2B).

### Lignes moyennement fragiles

- `gpt-4o-mini` : `same_provider_fallback` ; donneurs `gpt-4o|gpt-4|gpt-3.5-turbo` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider in the strict catalog. Donors: GPT-4o (200B), GPT-4 (440B), GPT-3.5 Turbo (175B). Retained central active-parameter basis: 200B (low/high prior: 148.8B - 506B).
- `gemini-3-flash` : `single_donor_family_prior` ; donneurs `gemini-2.0-flash` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider and same documented family/tier. Donors: Gemini 2.0 Flash (30B). Retained central active-parameter basis: 30B (low/high prior: 21B - 39B).
- `grok-4.20-reasoning` : `single_donor_family_prior` ; donneurs `grok-4` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider and same documented family/tier. Donors: Grok 4 (600B). Retained central active-parameter basis: 600B (low/high prior: 420B - 780B). Central prior floored to the largest same-family frontier donor because the target is a later revision in the same provider family with no direct public parameter disclosure.
- `grok-4.20-non-reasoning` : `single_donor_family_prior` ; donneurs `grok-4` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider and same documented family/tier. Donors: Grok 4 (600B). Retained central active-parameter basis: 600B (low/high prior: 420B - 780B). Central prior floored to the largest same-family frontier donor because the target is a later revision in the same provider family with no direct public parameter disclosure.
- `gemini-3.1-flash-live` : `single_donor_family_prior` ; donneurs `gemini-2.0-flash` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider and same documented family/tier. Donors: Gemini 2.0 Flash (30B). Retained central active-parameter basis: 30B (low/high prior: 21B - 39B).
- `o1-mini` : `same_provider_fallback` ; donneurs `gpt-3.5-turbo|gpt-4o|gpt-4` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider in the strict catalog. Donors: GPT-3.5 Turbo (175B), GPT-4o (200B), GPT-4 (440B). Retained central active-parameter basis: 200B (low/high prior: 148.8B - 506B).
- `claude-3-opus` : `same_provider_fallback` ; donneurs `claude-2` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider in the strict catalog. Donors: Claude 2 (100B). Retained central active-parameter basis: 100B (low/high prior: 70B - 130B).
- `claude-3-haiku` : `same_provider_fallback` ; donneurs `claude-2` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider in the strict catalog. Donors: Claude 2 (100B). Retained central active-parameter basis: 100B (low/high prior: 70B - 130B).
- `gemini-1.5-pro` : `same_provider_fallback` ; donneurs `lamda-1|glam-130b` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider in the strict catalog. Donors: LaMDA 1 (137B), GLaM 130B (130B). Retained central active-parameter basis: 133.5B (low/high prior: 110.5B - 157.6B).
- `grok-1.5` : `same_provider_fallback` ; donneurs `grok-1` ; note : Partial-data prior derived from benchmark-ready donor models with source-linked parameter counts. Selection rule: same provider in the strict catalog. Donors: Grok 1 (78.5B). Retained central active-parameter basis: 78.5B (low/high prior: 55B - 102B).

## Ecarts entre la publication et le catalogue courant

- `GPT-5.5` (illustrative_outputs) : `inference_wh,inference_gco2e` ; papier = `3.0833 Wh`, `1.1871 gCO2e` ; catalogue = `4.5322 Wh`, `1.7449 gCO2e`
- `Claude Mythos Preview` (illustrative_outputs) : `inference_wh,inference_gco2e` ; papier = `3.8014 Wh`, `1.4635 gCO2e` ; catalogue = `7.6502 Wh`, `2.9453 gCO2e`
- `Claude Mythos Preview` (representative_models) : `inference_wh,inference_gco2e,training_gwh` ; papier = `3.8014 Wh`, `1.4635 gCO2e`, `139.48 GWh` ; catalogue = `7.6502 Wh`, `2.9453 gCO2e`, `50.9455 GWh`
- `GPT-5.5` (representative_models) : `inference_wh,inference_gco2e,training_gwh` ; papier = `3.0833 Wh`, `1.1871 gCO2e`, `74.50 GWh` ; catalogue = `4.5322 Wh`, `1.7449 gCO2e`, `123.7950 GWh`
- `Gemini 3.1 Pro` (representative_models) : `inference_wh,inference_gco2e,training_gwh` ; papier = `0.4282 Wh`, `0.1649 gCO2e`, `1.07 GWh` ; catalogue = `4.7180 Wh`, `1.8164 gCO2e`, `185.6926 GWh`

## Conclusion operationnelle

- Je ne vois plus de chiffre marche expose sans categorie de provenance ou sans methode de screening explicite.
- En revanche, plusieurs chiffres recents restent des priors `partial-data` faibles ou moyens, donc defendables seulement comme ordres de grandeur de screening, pas comme quasi-mesures.
- La publication locale n'est plus coherente avec le catalogue courant pour certains modeles proprietaires frontier; si les reseaux sociaux reprennent les chiffres du papier, il faut soit regeler le catalogue sur cette version, soit mettre a jour la publication et les posts.

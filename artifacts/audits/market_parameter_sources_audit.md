# Audit des sources de parametres des modeles du marche

- lignes auditees : 83
- `exact_in_cited_source` : 31
- `derivable_from_cited_source` : 1
- `third_party_estimate` : 15
- `project_proxy_no_url` : 35
- `internal_proxy_no_url` : 1
- `needs_cleanup` : 0
- `manual_review` : 0

## Lignes a revoir

- aucune

## Regles d'audit

- `exact_in_cited_source` : la valeur retenue devrait apparaitre explicitement dans la source citee.
- `derivable_from_cited_source` : la valeur retenue est derivee directement d'une information explicite de la source citee.
- `third_party_estimate` : la valeur retenue vient d'une estimation tierce, pas d'une divulgation du fournisseur.
- `project_proxy_no_url` : la valeur retenue est un proxy de screening interne et le champ `parameter_source_url` est volontairement vide.
- `internal_proxy_no_url` : la valeur retenue est une approximation interne conservee sans URL de source de parametres.

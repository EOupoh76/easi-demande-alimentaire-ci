# Demande alimentaire urbaine en Côte d'Ivoire : système EASI (ENV 2008 et EHCVM 2021-2022)

Programmes de l'article « La demande alimentaire urbaine en Côte d'Ivoire face à deux crises de prix :
estimations du système EASI sur l'ENV 2008 et l'EHCVM 2021-2022 », Oupoh Oupoh Emmanuel.

## Contenu

| Fichier | Rôle |
| --- | --- |
| `prep_ehcvm2021.py` | Construit le fichier d'estimation urbain à partir des CSV de l'EHCVM 2021-2022 |
| `easi_estimation.py` | Estime le système EASI (SUR, symétrie imposée), élasticités, bootstrap, variantes |
| `requirements.txt` | Bibliothèques Python nécessaires |

## Données (non incluses)

- **EHCVM 2021-2022** : Microdata Library de la Banque mondiale, référence `CIV_2021_EHCVM-2_v01_M`
  (inscription et acceptation des conditions d'utilisation), version CSV.
- **ENV 2008** : à demander à l'Institut National de la Statistique (ANStat).

Les microdonnées ne sont pas redistribuées dans ce dépôt.

## Utilisation

```bash
pip install -r requirements.txt

# EHCVM 2021-2022
python3 prep_ehcvm2021.py donnees/ehcvm_csv sorties/ehcvm2021_urbain.csv
python3 easi_estimation.py csv sorties/ehcvm2021_urbain.csv sorties/2021 main 199

# Variantes : drop_imputed, weighted, prix_grappe ; repas hors ménage :
python3 prep_ehcvm2021.py donnees/ehcvm_csv sorties/ehcvm2021_urbain_repas.csv avec_repas

# ENV 2008 (fichiers d'estimation « pauvre » et « riche »)
python3 easi_estimation.py env2008 donnees/env2008_pauvre.txt donnees/env2008_riche.txt sorties/2008 main 199
```

Sorties : `results_<variante>.json`, `tableau1/2/3_*.csv`, `figure1_parts_*.csv`, `engel_rows.csv`.
Graine aléatoire du bootstrap : 2014.

## Citation

Pour citer ce code :

Oupoh, O. E. (2026). *Programmes EASI : demande alimentaire urbaine en Côte d'Ivoire
(ENV 2008 et EHCVM 2021-2022)*, version 1.0 [logiciel]. Zenodo.
https://doi.org/10.5281/zenodo.23687723

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23687723.svg)](https://doi.org/10.5281/zenodo.23687723)

## Déclaration

Ces programmes ont été écrits avec l'aide d'un assistant d'IA générative (Claude, Anthropic),
puis relus, exécutés et vérifiés par l'auteur, qui en assume la responsabilité.

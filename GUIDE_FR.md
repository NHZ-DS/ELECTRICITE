# Guide pas à pas : faire tourner GridCast FR sur les vraies données

Ce guide est pour toi, il n'a pas besoin d'être publié sur GitHub (tu peux le supprimer avant).

## Étape 1 : installer (15 min)

```bash
cd gridcast-fr
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Mac / Linux
pip install -r requirements-dev.txt
$env:PYTHONPATH="src;."           # Windows PowerShell
# export PYTHONPATH=src:.         # Mac / Linux
pytest -q                         # doit afficher 9 passed
```

## Étape 2 : essayer hors ligne (5 min)

```bash
python -m gridcast.cli demo
python -m gridcast.cli backtest --weeks 12
python -m gridcast.cli replay --days 21
python -m gridcast.cli forecast
python -m gridcast.cli monitor
streamlit run app/dashboard.py
```
Ces chiffres viennent de données **simulées** : ils ne doivent jamais apparaître dans ton CV ou ton README.
Ensuite, supprime `data/gridcast.duckdb` et le dossier `models/` avant de passer aux vraies données.

## Étape 3 : les vraies données (30 min)

```bash
python -m gridcast.cli ingest --start 2021-01-01
```
Si le téléchargement RTE échoue, ouvre dans ton navigateur
`https://odre.opendatasoft.com/explore/dataset/eco2mix-national-tr/` et vérifie le nom exact des
colonnes (`date_heure`, `consommation`). Le code est dans `src/gridcast/data/rte.py`.

Vérifie ensuite les données avant tout modèle, c'est ce que ferait un vrai data scientist :
- pas de trous importants (heures manquantes) ;
- le changement d'heure (fin mars, fin octobre) ;
- une consommation entre environ 30 000 et 90 000 MW ;
- l'hiver plus haut que l'été.

## Étape 4 : évaluer et comprendre (1 à 2 jours)

```bash
python -m gridcast.cli backtest --weeks 52
```
Puis regarde et **note par écrit** :
1. Le modèle hybride bat-il LightGBM seul ? Sur les jours froids ?
2. La couverture des intervalles est-elle proche de 90 % ?
3. Quelle sensibilité au froid le modèle physique trouve-t-il ? (RTE annonce environ 2 400 MW par °C en hiver : compare.)
4. Quels jours sont les plus mal prévus ? (jours fériés, ponts, vagues de froid…) Pourquoi ?

C'est cette analyse qui te rend crédible en entretien. Mets tes vrais chiffres dans le tableau du README.

## Étape 5 : améliorer une chose, toi-même

Choisis UNE amélioration et mesure son effet avec le backtest. Exemples :
- ajouter la nébulosité (`cloud_cover` dans Open-Meteo) : moins de soleil, plus d'éclairage ;
- estimer le seuil de chauffage (15 °C) au lieu de le fixer ;
- ajouter les vacances scolaires.

Garde-la seulement si elle améliore le backtest. En entretien, raconter « j'ai essayé X, ça a gagné Y % » vaut plus que tout le reste.

## Étape 6 : publier (1 jour)

1. Crée un dépôt GitHub public `gridcast-fr`, pousse le code.
2. Lance une fois le backtest et le replay, puis pousse `data/gridcast.duckdb` et `models/` avec `git add -f`.
3. Onglet **Actions** : active le workflow « daily forecast ». Il publiera une prévision chaque nuit.
4. Déploie le tableau de bord gratuitement sur Streamlit Community Cloud (share.streamlit.io), fichier `app/dashboard.py`.
5. Fais une capture du tableau de bord (vraies données) dans `docs/dashboard.png`, et mets le lien en haut du README.

## Étape 7 : le CV (seulement quand tout tourne)

Exemple de ligne, à remplir avec **tes** chiffres :

> **GridCast FR – Prévision de la consommation électrique française** | Python, LightGBM, FastAPI, Streamlit, DuckDB | 2026
> – Pipeline automatisé (RTE, météo) et modèle hybride physique + LightGBM : MAPE de X % en backtest sur 52 semaines, contre Y % pour la référence.
> – Intervalles de prédiction conformes (couverture Z % pour 90 % visés), monitoring de dérive, API et tableau de bord en ligne.

## Ce que tu dois savoir expliquer en entretien
- Pourquoi un backtest glissant et pas un simple découpage train/test.
- Pourquoi les décalages (lags) sont d'au moins 24 h.
- Comment marche la conformal prediction, et pourquoi une marge par heure.
- Pourquoi un modèle hybride : la physique donne la structure, le ML corrige les écarts.
- Ce que tu ferais si l'alerte de dérive se déclenchait.

Lis chaque fichier du dossier `src/gridcast/` et assure-toi de comprendre chaque fonction : c'est ton projet, on te posera des questions dessus.

# POC FAIR – RAG

Ce projet est un Proof of Concept visant à explorer l’utilisation d’une approche
RAG (Retrieval-Augmented Generation) pour assister l’analyse de risque cyber
basée sur la méthodologie FAIR.

## Objectif
Aider à la construction de scénarios de risque et à la structuration
des estimations d’impact financier à partir de rapports de menace.

## Principe
- Ingestion d’un rapport ENISA
- Indexation via FAISS
- Recherche sémantique
- Génération d’une réponse structurée (scénario, impacts, hypothèses, limites)

## Limites
- POC volontairement simple
- Données majoritairement qualitatives
- Validation humaine indispensable

## Exécution
```bash   
pip install -r requirements.txt
python poc_rag_fair.py

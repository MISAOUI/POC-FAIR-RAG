from pathlib import Path

from rich.console import Console

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI


# =========================
# Configuration générale
# =========================

console = Console()

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index/faiss_index")


# =========================
# Règles système (LLM)
# =========================

SYSTEM_RULES = """Tu es un consultant en cybersécurité spécialisé en analyse de risques (méthodologie FAIR).

Règles :
- Utilise UNIQUEMENT le CONTEXTE fourni pour les affirmations factuelles.
- Si les informations sont insuffisantes, indique clairement :
  "Données insuffisantes dans les documents."
- Lorsque tu estimes des impacts financiers, fournis :
  un minimum, une estimation probable et un maximum.
- Sois clair, structuré et concis.
- Cite les sources sous la forme [1], [2], [3] en t'appuyant sur la liste SOURCES.
"""


# =========================
# Prompt FAIR simplifié
# =========================

FAIR_PROMPT = """CONTEXTE :
{context}

TÂCHE :
À partir du contexte ci-dessus, estime les impacts financiers possibles
d’une attaque par ransomware sur un hôpital de taille moyenne en France.

Fournis une réponse structurée avec :
1) Description du scénario
2) Impacts financiers possibles (min / probable / max) en euros
3) Hypothèses retenues
4) Incertitudes et limites de l’estimation

SOURCES :
{sources}
"""


# =========================
# Recherche du PDF
# =========================

def find_first_pdf() -> Path:
    pdfs = sorted([p for p in RAW_DIR.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(
            "Aucun fichier PDF trouvé dans data/raw. Veuillez ajouter un rapport."
        )
    return pdfs[0]


# =========================
# Création / chargement FAISS
# =========================

def build_or_load_vectorstore(pdf_path: Path) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Si l’index existe déjà, on le recharge
    if INDEX_DIR.exists():
        console.print(
            f"[green]✅ Chargement de l’index FAISS existant depuis {INDEX_DIR}[/green]"
        )
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    console.print(
        f"[yellow]⏳ Construction de l’index FAISS à partir de {pdf_path.name}...[/yellow]"
    )

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    console.print(
        f"[green]✅ Index créé : {len(chunks)} segments sauvegardés dans {INDEX_DIR}[/green]"
    )

    return vectorstore


# =========================
# Formatage des sources
# =========================

def format_sources(docs) -> str:
    lignes = []

    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "inconnu")).name
        page = doc.metadata.get("page", None)

        localisation = source
        if isinstance(page, int):
            localisation += f" (page {page + 1})"

        extrait = doc.page_content.strip().replace("\n", " ")
        if len(extrait) > 260:
            extrait = extrait[:260] + "…"

        lignes.append(f"[{i}] {localisation}\n    \"{extrait}\"")

    return "\n".join(lignes)


# =========================
# Recherche sémantique (RAG)
# =========================

def retrieve_context(vectorstore: FAISS, question: str, k: int = 8):
    docs = vectorstore.similarity_search(question, k=k)

    contexte = "\n\n".join([d.page_content for d in docs])
    sources = format_sources(docs)

    return contexte, sources


# =========================
# Appel OpenAI (génération)
# =========================

def run_openai(prompt: str) -> str:
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Tu es un consultant en cybersécurité utilisant la méthodologie FAIR.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


# =========================
# Fonction principale
# =========================

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = find_first_pdf()
    console.print(f"[bold]PDF utilisé :[/bold] {pdf_path}")

    vectorstore = build_or_load_vectorstore(pdf_path)

    console.print(
        "\n[bold]Entrez votre question (ou appuyez sur Entrée pour utiliser la question par défaut) :[/bold]"
    )

    question = input("> ").strip()
    if not question:
        question = (
            "Quels sont les impacts financiers d’un ransomware "
            "dans le secteur de la santé ?"
        )

    contexte, sources = retrieve_context(vectorstore, question, k=8)

    console.print("\n[bold]===== SOURCES RETROUVÉES =====[/bold]")
    console.print(sources)

    prompt_final = (
        f"{SYSTEM_RULES}\n\n"
        + FAIR_PROMPT.format(context=contexte, sources=sources)
    )

    console.print("\n[bold]===== RÉPONSE LLM (OpenAI + RAG) =====[/bold]")
    reponse = run_openai(prompt_final)
    console.print(reponse)


# =========================
# Point d’entrée
# =========================

if __name__ == "__main__":
    main()

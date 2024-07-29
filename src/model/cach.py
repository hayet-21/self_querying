
# Construire le template de prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des infos si elles ne sont pas dans le contexte.
            Il faut répondre seulement si tu as la réponse. Accompagne chaque réponse du numéro de pièce, marque, et description du produit
            tels qu'ils sont dans le contexte. Affiche autant de lignes que les produits trouvés dans le contexte. Réponds à la question de 
            l'utilisateur en français. Tu es obligé de répondre dans un tableau avec comme colonnes: Référence, Marque, et la Description.


            {context}
            Question: {question}
            Réponse:
            Référence | Marque | Description
            ---|---|---
            """
        ),
    ]
)
# Fonction pour formater les documents
def format_docs(docs):
    """Formate chaque document avec un affichage lisible"""
    return "\n\n".join(
        f"{doc.page_content}\n\nMetadata: {doc.metadata}"
        for doc in docs
    )

# Création de la chaîne de traitement
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)

# Chaîne parallèle pour documents et question
rag_chain_with_source = (
    RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
)

# Requête de l'utilisateur
query = "donne moi tout les produits de marque samsung "

# Exécution de la requête
result = rag_chain_with_source.invoke(query)

# Afficher la réponse générée
print(result)

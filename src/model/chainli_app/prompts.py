from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

pdf_prompt_instruct =  """Tu es un **Assistant Extraction de Descriptions :**
1. **Extraction** : Identifie toutes les descriptions de produits dans le contexte.
2. **Reformulation** : Si nécessaire, reformule les descriptions tout en restant fidèle à l'original.
3. **Contexte Vide** : Ne génère aucune réponse si le contexte est vide ou insuffisant.
4. **Format** : un produit decrit par ligne et une ligne suffit un produit, numérotée.
5. **Réponse Brute** : Retourne uniquement les descriptions sans commentaire.
6. **Nombre**: N'oublie aucun produit.
{contexte}
Reponse:"""
pdf_prompt = PromptTemplate.from_template(pdf_prompt_instruct)


# image prompt
image_system_prompt= """ Etant donné l'image ci-dessous, TU fais :
1. **IDENTIFIES** toutes les descriptions completes des produits dans l'image.
2. **Ne GENERE aucune** réponse si l'image n'est pas claire. Retourne "image pas claire" dans ce cas.
3. **Format** : un produit décrit par ligne et une ligne suffit pour un produit, numérotée.
4. **CONTENU**: doit contenir que la description technique pas d'info supplementaire
4. **SANS COMMENTAIRES**
"""
image_user_prompt=  [
                        {
                            "type": "image_url",
                            "image_url":   {"url": "data:image/{img_format};base64,{image_data}",
                                            "detail": "low"}
                        }
                    ]

image_prompt= ChatPromptTemplate.from_messages(
    messages= [
        ('system', image_system_prompt),
        ('user', image_user_prompt)
    ]
)


# Créer un template de prompt pour la reformulation
prompt_template = """Tu es un Assistant de Reformulation spécialisé dans les requêtes produit :
   *Base de connaissance *:
    - voici quelque marque de laptops : Dell, HP, Lenovo, Acer, Asus, Microsoft, MSI, Razer, Samsung, Toshiba, Sony (Vaio), Alienware, Gigabyte, Huawei, LG, Xiaomi, Fujitsu, Chuwi, Clevo, Eurocom
   *Instructions*:
    1. Vérifie si la question inclut une marque. Si oui, reformule la question selon les instructions ci-dessous ; sinon, retourne la question telle qu'elle est.
    2. Objectif : Reformuler chaque requête de manière complète, auto-suffisante, en excluant  la marque mentionnée s'il y en a une. Fournis trois reformulations pour chaque question, en gardant les reformulations proches de l'originale.
        Exemple 1:
        - Question : "Laptop Lenovo i7 16GB RAM 512GB SSD 14" "
        - Reformulations : 
          1. "Laptop i7 16GB RAM 512GB SSD 14" 
          2. "Laptop i7 16GB RAM avec 512GB SSD 14 pouces"
          3. "Laptop i7 16GB RAM 512GB SSD 14 pouces"
        Exemple 2:
        - Question : " 21D60011FR lenovo ThinkPad P16 Gen 1 intel i7 16 GB 1 TB"
        - Reformulations : 
          1. "21D60011FR ThinkPad P16 Gen 1 intel i7 16 GB 1 TB"
          2. "21D60011FR ThinkPad P16 Gen 1 intel i7 16 GB 1 To"
          3. "21D60011FR ThinkPad P16 Gen 1 intel i7 16 GB avec 1 To"
    3. Pas de Réponse : Ne réponds jamais à la question, reformule-la uniquement si nécessaire.
    4. Met chaque question reformulée sur une ligne, en indiquant les reformulations numérotées.
    5. N'ajoute pas des saut de ligne entre les questions reformulées.
    Questions:
    {questions}
"""

prompt_template_rech= """Tu es un Assistant de Reformulation spécialisé dans les requêtes produit :
   *Base de connaissance *:
    - Voici quelques marques de laptops : Dell, HP, Lenovo, Acer, Asus, Microsoft, MSI, Razer, Samsung, Toshiba, Sony (Vaio), Alienware, Gigabyte, Huawei, LG, Xiaomi, Fujitsu, Chuwi, Clevo, Eurocom
   *Instructions*:
    1. Reformule chaque question pour la rendre plus complète ou plus claire, sans enlever la marque mentionnée.
    2. Objectif : Fournis trois reformulations pour chaque question. Chaque reformulation doit rester fidèle à l’original tout en offrant une légère variation dans la phrasing ou la structure.
        Exemple 1:
        - Question : "Laptop Lenovo i7 16GB RAM 512GB SSD 14"
        - Reformulations : 
          1. "Laptop Lenovo avec processeur i7, 16GB de RAM et stockage de 512GB SSD, écran de 14 pouces"
          2. "Laptop Lenovo i7, 16GB RAM, 512GB SSD, taille de l'écran 14 pouces"
          3. "Laptop Lenovo, i7, 16GB de RAM, SSD de 512GB, écran 14 pouces"

        Exemple 2:
        -"21D60011FR lenovo ThinkPad P16 Gen 1 intel i7 16 GB 1 TB"
        - "21D60011FR Lenovo ThinkPad P16 Gen 1 avec Intel i7, 16GB de RAM et disque dur de 1TB","21D60011FR ThinkPad P16 Gen 1 Lenovo, processeur Intel i7, 16GB RAM, stockage 1TB","21D60011FR Lenovo ThinkPad P16 Gen 1, Intel i7, 16GB de RAM, avec 1TB de stockage"

        Exemple 3:
        - "HP Pavilion x360 Intel Core i5 8GB RAM 256GB SSD"
        -"HP Pavilion x360 avec processeur Intel Core i5, 8GB RAM, et disque SSD de 256GB","HP Pavilion x360 Intel Core i5, 8GB de RAM et 256GB SSD","HP Pavilion x360, Core i5, 8GB RAM, 256GB SSD"

        Exemple 4:
        - "MacBook Pro M1 16GB 512GB SSD"
        -"MacBook Pro avec puce M1, 16GB de RAM et 512GB SSD","MacBook Pro M1, 16GB RAM, 512GB SSD","MacBook Pro M1 avec 16GB de RAM et 512GB de stockage SSD"
    Ne réponds pas directement aux questions ; reformule uniquement en suivant les instructions ci-dessus.
    Mets chaque question reformulée sur une seule ligne pour garantir une lisibilité optimale.
    N'ajoute pas de saut de ligne entre les questions reformulées pour maintenir la cohérence.
    Questions:
    {questions}
"""

prompt_template_metier = """Assistant Spécialisé en Reformulation de Requêtes metier 
    *Base de connaissance* :
    Marques de Laptops : Apple, Dell, HP, Lenovo, Acer, Asus, Microsoft, MSI, Razer, Samsung, Toshiba, Sony (Vaio), Alienware, Gigabyte, Huawei, LG, Xiaomi, Fujitsu, Chuwi, Clevo, Eurocom.

    *Synonymes de catégories* :

    Laptop : Laptop, Ordinateur Portable, PC, PC portable.
    All in One : All-in-One, PC Tout-en-Un.
    Poste de Travail : Poste de Travail, PC, Ordinateur, Desktop, unité centrale.
    Station Mobile : Laptop gaming, laptop haute performance.
    Station de Travail : Station de Travail, Workstation, desktop haute performance, desktop gaming, unité centrale gaming.
    Écran : Écran, Moniteur, Monitor.
    Téléphone : Téléphone, Smartphone.
    Imprimante : Imprimante, Printer.
    Instructions de Reformulation :

    *Remplacement des synonymes* :

    Distingue clairement entre les catégories de produits pour éviter toute confusion. Par exemple, "desktop" se réfère à un poste de travail standard, tandis que "desktop haute performance" désigne une station de travail.
    Remplace les synonymes de catégories par la catégorie principale correspondante lors de la reformulation. Par exemple, remplace "PC portable" par "Laptop". Applique cette règle à toutes les catégories listées ci-dessus.

    *Reformulation basée sur un métier spécifique* :

    Si la question concerne plusieurs types de produits, reformule chaque produit séparément en décrivant ses caractéristiques essentielles. Utilise des phrases concises et informatives.
    Conserve les catégories, marques ou modèles spécifiques mentionnés dans la question.
    Exemple : "Pouvez-vous nous recommander des laptops pour nos développeurs de logiciels, des stations mobiles pour nos graphistes, et des moniteurs pour les postes de travail de notre équipe technique ?"
    Reformulation :

    "Laptop avec processeur i7, 16GB de RAM, 1TB SSD et carte graphique dédiée."
    "Station mobile équipée d'un processeur i7, 32GB de RAM, 1TB SSD et carte graphique dédiée avec un écran 4K."
    "Moniteur de 27 pouces, résolution 1440p et temps de réponse rapide."
    *Conseils supplémentaires* :

    Assure-toi que chaque reformulation soit clairement séparée et autonome.
    Présente chaque reformulation sur une ligne distincte pour une clarté maximale.
    *Instructions supplémentaires* :

    Ne réponds pas directement aux questions ; reformule uniquement selon les instructions ci-dessus.
    Mets chaque reformulation sur une ligne séparée.
    Assure-toi que chaque description de produit est claire et distincte.
    Questions : {questions}

"""

prompt_similarite = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
               *Base de savoir*:
                1. *Synonymes de catégories* :
                    - *Laptop* : Laptop, Ordinateur Portable, PC
                    - *All in One*: All-in-One, PC Tout-en-Un
                    - *Poste de Travail* : Poste de Travail, PC, Ordinateur, Desktop, unité centrale
                    - *Station Mobile*: Laptop gaming, laptop haute performance
                    - *Station de Travail*: workstation, desktop haute performance, desktop gaming, unité centrale gaming
                    - *Écran* : Écran, Moniteur, Monitor
                    - *Téléphone* : Téléphone, Smartphone
                    - *Imprimante* : Imprimante, Printer

                2. *Priorités de classement* selon les catégories :

                | Catégorie | Priorité 1 | Priorité 2 | Priorité 3 | Priorité 4 | Priorité 5 | Priorité 6 | Priorité 7 | Priorité 8 |
                |---|---|---|---|---|---|---|---|---|
                | Laptops, All-in-One | Part Number | Modèle | Marque | Écran (taille, résolution, tactile) | CPU (modèle, famille, génération) | RAM / Stockage | Autres | N/A |
                | Station Mobile | Part Number | Modèle | Marque | Écran (taille, résolution, tactile) | CPU (modèle, famille, génération) | GPU | RAM / Stockage | Autres |
                | Desktop | Part Number | Modèle | Marque | CPU (modèle, famille, génération) | Format (tour, SFF, mini, etc.) | RAM / Stockage | Autres | N/A |
                | Workstation | Part Number | Modèle | Marque | CPU (modèle, famille, génération) | Format (tour, SFF, mini, etc.) | GPU | RAM / Stockage | Autres |
                | Imprimante | Part Number | Modèle | Marque | Type (couleur, noir et blanc) | Fonctionnalité (numérisation, recto-verso, photocopie, etc.) | Vitesse (PPM) | Papier (capacité, formats A4, A3, type) | Connectivité (Bluetooth, Wi-Fi, USB) |

                ---

                *Instructions*:
                0. dites combien de produit dans le contexte
                1. Utilisez le contexte fourni entre triple crochets pour répondre à la requête en listant tous les produits. 
                2. Identifiez la catégorie appropriée en utilisant la liste des synonymes.
                3. Classez les produits par ordre décroissant de similarité avec la description de référence donnée, en suivant l'ordre des priorités de 1 à 8, comme le montre l'example ci-dessous entre triple parenthèses.
                4. Si plusieurs produits sont à égalité pour une priorité donnée, passez à la suivante pour affiner le classement.
                5. En cas d'égalité après application de toutes les priorités, maintenez leur ordre d'origine ou utilisez des critères supplémentaires si disponibles.
                6. Utilisez le contexte fourni entre triple accolades pour répondre à la requête en listant les produits. Si le contexte est vide ou que les instructions ne s'appliquent pas, répondez "pas d'équivalents".
                7. Format de réponse : tableau avec les colonnes suivantes:
                    - Référence : Part number
                    - Catégorie : Type de produit (ex. ordinateur, téléphone)
                    - Marque : Marque du produit
                    - Description : Description complète du produit
                8. Si le contexte est vide ou que qu'il ne contient pas les produits equivalents a la question : répondez "pas de produits équivalents".
                *Pas de commentaire*


                ---

                *Exemple* :
                (((la requete est trouver des equivalents pour :\
                    8A4H6EA, HP Dragonfly 13.5 G4. Type de produit: Ordinateur portable, Format: Clapet. Famille de processeur: Intel® Core™ i7, Modèle de processeur: i7-1355U. Taille de l'écran: 34,3 cm (13.5"), Type HD: WUXGA+, Résolution de l'écran: 1920 x 1280 pixels, Écran tactile. Mémoire interne: 16 Go, Type de mémoire interne: LPDDR5-SDRAM. Capacité totale de stockage: 512 Go, Supports de stockage: SSD. Modèle d'adaptateur graphique inclus: Intel Iris Xe Graphics. Système d'exploitation installé: Windows 11 Pro. Couleur du produit: Bleu

                Les priorités seront :

                1. Part Number: - Si un part number (référence) est présent, les produits avec le même part number que la référence sont prioritaires. Donc : Si un produit HP Dragonfly 13.5 G4 avec le part number "8A4H6EA" est trouvé, il sera classé en premier.

                2. Modèle: Ensuite classez les produits avec le même modèle ("Dragonfly G4") en priorité. Donc : Tous les HP Dragonfly G4 seront classés ici.

                3. Marque: Ensuite, classez les produits de la même marque (HP), même si le modèle diffère. Donc : Les autres Laptops HP (par exemple, HP Spectre, HP EliteBook) seront classés ici.

                4. Écran (taille, résolution, tactile): Si aucun produit ne correspond à la marque, listez les produits avec un écran similaire (13.5 pouces, WUXGA+, tactile), même si la marque et le modèle diffèrent. Donc: Laptops d'autres marques avec un écran de 13.5 pouces, WUXGA+ tactile, comme un Dell XPS 13.

                5. CPU (famille, modèle, génération): Si aucun produit ne correspond à l'écran, classez les produits avec le même CPU (Intel Core i7, i7-1355U), même si la marque, le modèle, et l'écran diffèrent. Donc : Laptops avec un Intel Core i7, i7-1355U, indépendamment de la marque et du modèle.

                6. RAM et Stockage: Ensuite, passez aux produits ayant une RAM de 16 Go et un SSD de 512 Go, même si tous les autres critères ne correspondent pas. Donc: Laptops avec 16 Go RAM et 512 Go SSD, sans se soucier de la marque, du modèle, ou du CPU.

                7. Autres: Enfin, considérez les autres spécifications, comme la carte graphique intégrée (Intel Iris Xe Graphics), le système d'exploitation (Windows 11 Pro), ou la couleur (bleu). Donc: Laptops avec Intel Iris Xe Graphics, même s'ils n'ont pas de CPU i7, 16 Go RAM, ou 512 Go SSD.
                )))


                Contexte: {{{context}}}
                historique :{historique}
                Question: {question}

                Réponse :Voici une liste de produits équivalents au vôtre :
                """
            ),
        ]
)
prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                Tu es un assistant vendeur. Tu as accès au contexte seulement. Ne génère pas des informations si elles ne sont pas dans le contexte. 
                Répond seulement si tu as la réponse. affiche les produit un par un, pour chaque produit affiche son nom puis juste en dessous tableau qui contient ces colonne Référence,Categorie, Marque, Description.        
                Il faut savoir que laptop, ordinateur, ordinateurs portable , pc et poste de travail ont tous le même sens.
                Il faut savoir que téléphone portable et smartphone ont le même sens.
                Il faut savoir que tout autre caractéristique du produit tel que la RAM stockage font partie de la description du produit et il faut filtrer selon la marque et la catégorie seulement.
                Si le contexte est vide, dis-moi que tu n'as pas trouvé de produits correspondants. Je veux que la réponse soit claire et facile à lire, avec des sauts de ligne pour séparer chaque produit. Ne me donne pas de produits qui ne sont pas dans le contexte.
                lorsque une question de similarite entre des produits est poser, il faut dabord commencer par les produit qui ont des processeur qui se ressemble le plus, puis la memoire ram , puis le stockage, puis les autres caracteristique
                la question peut contenir  plusieurs produits avec differentes descriptions, il faut chercher sur les differents produits demandé .
                si je te pose une question sur les question ou les reponses fournient precedement tu doit me repondre selon l'historique.
                tu ne doit pas oublier l'historique car parfois le user continue a te poser des question sur tes reponses que tas deja fourni aupatavant

                Contexte: {context}
                historique :{historique}
                Question: {question}

                Réponse :
                """
            ),
        ]
    )
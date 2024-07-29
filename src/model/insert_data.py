from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
import traceback
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Charger les variables d'environnement
load_dotenv()  # Charge les variables d'environnement depuis un fichier .env

HF_TOKEN = os.getenv('API_TOKEN')
CHROMA_PATH = os.path.abspath(f"../{os.getenv('CHROMA_PATH')}")
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))  # Ajustez cette valeur selon votre système
DATA_PATH_CSV = os.getenv('DATA_PATH_CSV')
COLLECTION_CSV = os.getenv('COLLECTION_CSV')
HF_TOKEN = os.getenv('API_TOKEN')

embedding_function = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="intfloat/multilingual-e5-large"
        )
docs = [
    Document(
        page_content="FSP:GD5S60Z00DESV6,macle,SP 5Y OS 9X5 4H RT,fujitsu,1584.0,50",
        metadata={"description":"SP 5Y OS 9X5 4H RT","part": "FSP:GD5S60Z00DESV6", "fournisseur": "macle", "marque": "fujitsu", "prix": 1584.0, "quantity": 50},
    ),
    Document(
        page_content="TS-464-8G,Ingram,TS-464-8G 4BAY 8GBDDR4 2X2.5GBE,qnap,630.74,1",
        metadata={"description":"TS-464-8G 4BAY 8GBDDR4 2X2.5GBE","part": "TS-464-8G", "fournisseur": "Ingram", "marque": "qnap", "prix": 630.74, "quantity": 1},
    ),
    Document(
        page_content="D-DDR4-4GB-007,convena,ProXtend 4GB DDR4 PC4-21300 2666MHz,proxtend,22.07,3",
        metadata={"description":"ProXtend 4GB DDR4 PC4-21300 2666MHz","part": "D-DDR4-4GB-007", "fournisseur": "convena", "marque": "proxtend", "prix": 22.07, "quantity": 3},
    ),
    Document(
        page_content="46372,PCA,Fibre optique Duplex LC / LC OM3 3m,lindy,7.05,1",
        metadata={"description":"Fibre optique Duplex LC / LC OM3 3m","part": "46372", "fournisseur": "PCA", "marque": "lindy", "prix": 7.05, "quantity": 1},
    ),
    Document(
        page_content="21.15.3941,Secomp,Cordon ROLINE Data Center SLIM, UTP Cat6A/Cl.EA, LSOH, bleu, 0,3m,roline,0.86,749",
        metadata={"description":"Cordon ROLINE Data Center SLIM, UTP Cat6A/Cl.EA, LSOH, bleu, 0,3m","part": "21.15.3941", "fournisseur": "Secomp", "marque": "roline", "prix": 0.86, "quantity": 749},
    ),
    Document(
        page_content="HL2P5E,Also,HPE Aruba Foundation Care 5 Years Next Business Day Exchange Hardware Only 6200F 48G 740 POE Switch Service,hewlett packard enterprise,386.25,100",
        metadata={"description":"HPE Aruba Foundation Care 5 Years Next Business Day Exchange Hardware Only 6200F 48G 740 POE Switch Service","part": "HL2P5E", "fournisseur": "Also", "marque": "hewlett packard enterprise", "prix": 386.25, "quantity": 100},
    ),
    Document(
        page_content="C13T619100,West Cost,EPSON BAC RECUP ENCRE,epson large format,12.0,4",
        metadata={"description":"EPSON BAC RECUP ENCRE","part": "C13T619100", "fournisseur": "West Cost", "marque": "epson large format", "prix": 12.0, "quantity": 4},
    ),
    Document(
        page_content="01ET977,Also,LENOVO PW 1YR Parts Delivered NBD,lenovo,271.68,100",
        metadata={"description":"LENOVO PW 1YR Parts Delivered NBD","part": "01ET977", "fournisseur": "Also", "marque": "lenovo", "prix": 271.68, "quantity": 100},
    ),
    Document(
        page_content="SU515A,techdata,Samsung CLT-Y506L - À rendement élevé - jaune - original - cartouche de toner (SU515A) - pour Samsung CLP-680DW, CLP-680ND, CLX-6260FD, CLX-6260FR, CLX-6260FW, CLX-6260ND,hp,145.68,4",
        metadata={"description":"Samsung CLT-Y506L - À rendement élevé - jaune - original - cartouche de toner (SU515A) - pour Samsung CLP-680DW, CLP-680ND, CLX-6260FD, CLX-6260FR, CLX-6260FW, CLX-6260ND","part": "SU515A", "fournisseur": "techdata", "marque": "hp", "prix": 145.68, "quantity": 4},
    ),
    Document(
        page_content="220-80E-00003,Jarltech,Zebra 220Xi4, 8 pts/mm (203 dpi), RTC, ZPLII, multi-IF,zebra,5995.83,17",
        metadata={"description":"Zebra 220Xi4, 8 pts/mm (203 dpi), RTC, ZPLII, multi-IF","part": "220-80E-00003", "fournisseur": "Jarltech", "marque": "zebra", "prix": 5995.83, "quantity": 17},
    ),
    Document(
        page_content="U7PV5E,macle,EPACK 1YR NBD WCDMR 513024G4SF,hewlett packard enterprise,227.0,50",
        metadata={"description":"EPACK 1YR NBD WCDMR 513024G4SF","part": "U7PV5E", "fournisseur": "macle", "marque": "hewlett packard enterprise", "prix": 227.0, "quantity": 50},
    ),
    Document(
        page_content="21L30031MB,Copaco,L16 G1 T\\16_WUXGA_AG_300N\\CORE_ULT7_155U_1.7G_12C_14T\\16GB_DDR5_5600_SODIMM\\512GB_SSD_M.2_2280_G4_TLC_OP\\W11_P64-WE_ML AZERTY BE,lenovo,1501.36,39",
        metadata={"description":"L16 G1 T\\16_WUXGA_AG_300N\\CORE_ULT7_155U_1.7G_12C_14T\\16GB_DDR5_5600_SODIMM\\512GB_SSD_M.2_2280_G4_TLC_OP\\W11_P64-WE_ML AZERTY BE","part": "21L30031MB", "fournisseur": "Copaco", "marque": "lenovo", "prix": 1501.36, "quantity": 39},
    ),
    Document(
        page_content="P86550314,Also,HP Poly 3yr Poly+ Onsite G7500 4k Codec-Wireless Presentation Sys Touch Cntrl EEIV-4x cam mic remote,hp inc.,2684.84,100",
        metadata={"description":"HP Poly 3yr Poly+ Onsite G7500 4k Codec-Wireless Presentation Sys Touch Cntrl EEIV-4x cam mic remote","part": "P86550314", "fournisseur": "Also", "marque": "hp inc.", "prix": 2684.84, "quantity": 100},
    ),
    Document(
        page_content="42W2820,macle,LENOVO FAN FOR TP T61,lenovo,47.0,5",
        metadata={"description":"LENOVO FAN FOR TP T61","part": "42W2820", "fournisseur": "macle", "marque": "lenovo", "prix": 47.0, "quantity": 5},
    ),
    Document(
        page_content="DELL-WD22TB4,Siewert & Kau,Dell notebook docking station WD22TB4 Thunderbolt,dell,218.31,17",
        metadata={"description":"Dell notebook docking station WD22TB4 Thunderbolt","part": "DELL-WD22TB4", "fournisseur": "Siewert & Kau", "marque": "dell", "prix": 218.31, "quantity": 17},
    ),
    Document(
        page_content="MSP2630,EET,Paper Pickup Roller,coreparts,9.3,17",
        metadata={"description":"Paper Pickup Roller","part": "MSP2630", "fournisseur": "EET", "marque": "coreparts", "prix": 9.3, "quantity": 17},
    ),
    Document(
        page_content="DS70S-950WH1,Siewert & Kau,Neomounts DS70S-950WH1 mounting kit - full-motion - for monitor - white,neomounts,163.78,20",
        metadata={"description":"Neomounts DS70S-950WH1 mounting kit - full-motion - for monitor - white","part": "DS70S-950WH1", "fournisseur": "Siewert & Kau", "marque": "neomounts", "prix": 163.78, "quantity": 20},
    ),
    Document(
        page_content="H2ZT2E,macle,EPACK 1YR FC 4H EXCH 7005 CTR,hewlett packard enterprise,490.0,50",
        metadata={"description":"EPACK 1YR FC 4H EXCH 7005 CTR","part": "H2ZT2E", "fournisseur": "macle", "marque": "hewlett packard enterprise", "prix": 490.0, "quantity": 50},
    ),
    Document(
        page_content="1.445-330.0,Difox,KÃ¤rcher PGS 4-18 Scie,kärcher,58.01,1",
        metadata={"description":"Kärcher PGS 4-18 Scie","part": "1.445-330.0", "fournisseur": "Difox", "marque": "kärcher", "prix": 58.01, "quantity": 1},
    ),
    Document(
        page_content="0-HDMI200ETHNYROTCAV,MGF,C ble HDMI 1.4 M/M, fiche or 2 mètres , nylon tressé, Ehternet 1 c té 90 c fixe,,heden,4.92,104",
        metadata={"description":"C ble HDMI 1.4 M/M, fiche or 2 mètres , nylon tressé, Ehternet 1 c té 90 c fixe","part": "0-HDMI200ETHNYROTCAV", "fournisseur": "MGF", "marque": "heden", "prix": 4.92, "quantity": 104},
    ),
    Document(
        page_content="404673-001-RFB,EET,Systemboard,hewlett packard enterprise,173.7,1",
        metadata={"description":"Systemboard","part": "404673-001-RFB", "fournisseur": "EET", "marque": "hewlett packard enterprise", "prix": 173.7, "quantity": 1},
    ),
    Document(
        page_content="MT103ZM/A,techdata,Apple - Coque de protection pour téléphone portable - compatibilité avec MagSafe - silicone - noir - pour iPhone 15 Plus,apple,49.17,32",
        metadata={"description":"Apple - Coque de protection pour téléphone portable - compatibilité avec MagSafe - silicone - noir - pour iPhone 15 Plus","part": "MT103ZM/A", "fournisseur": "techdata", "marque": "apple", "prix": 49.17, "quantity": 32},
    ),
    Document(
        page_content="TL-SG1005P,PICATA,TL-SG1005P - 5 Ports Gigabit dont 4 PoE,tp-link,33.35,1",
        metadata={"description":"TL-SG1005P - 5 Ports Gigabit dont 4 PoE","part": "TL-SG1005P", "fournisseur": "PICATA", "marque": "tp-link", "prix": 33.35, "quantity": 1},
    ),
    Document(
        page_content="H8CQ1E,techdata,HPE Foundation Care Next Business Day Exchange Service - Contrat de maintenance prolongé - remplacement - 4 années - expédition - 9x5 - temps de réponse : NBD - universitaire, pour les particuliers - pour P/N: JX957AR,hewlett packard enterprise,38.0,9876",
        metadata={"description":"HPE Foundation Care Next Business Day Exchange Service - Contrat de maintenance prolongé - remplacement - 4 années - expédition - 9x5 - temps de réponse : NBD - universitaire, pour les particuliers - pour P/N: JX957AR","part": "H8CQ1E", "fournisseur": "techdata", "marque": "hewlett packard enterprise", "prix": 38.0, "quantity": 9876},
    ),
    Document(
        page_content="TB4CDOCKUE,Ingram,THUNDERBOLT 4 DOCK 96W DUAL 4K ,startech,245.5,3",
        metadata={"description":"THUNDERBOLT 4 DOCK 96W DUAL 4K","part": "TB4CDOCKUE", "fournisseur": "Ingram", "marque": "startech", "prix": 245.5, "quantity": 3},
    ),
    Document(
        page_content="MZ-V9E1T0BW,Also,SAMSUNG SSD 990 EVO 1To M.2 NVMe PCIe,samsung,114.57,100",
        metadata={"description":"SAMSUNG SSD 990 EVO 1To M.2 NVMe PCIe","part": "MZ-V9E1T0BW", "fournisseur": "Also", "marque": "samsung", "prix": 114.57, "quantity": 100},
    ),
    Document(
        page_content="U3WZ9E,macle,EPACK 3YR NBD 66/88XX FIREWALL,hewlett packard enterprise,4596.0,50",
        metadata={"description":"EPACK 3YR NBD 66/88XX FIREWALL","part": "U3WZ9E", "fournisseur": "macle", "marque": "hewlett packard enterprise", "prix": 4596.0, "quantity": 50},
    ),
    Document(
        page_content="CL-P039-AL12BL-A,PICATA,Contac Silent 12 CPU Cooler ,thermaltake,21.31,1",
        metadata={"description":"Contac Silent 12 CPU Cooler ","part": "CL-P039-AL12BL-A", "fournisseur": "PICATA", "marque": "thermaltake", "prix": 21.31, "quantity": 1},
    ),
    Document(
        page_content="22450121,Ingram,RP-F10-K27J1-3 10819 BLK EU,seiko,176.96,22",
        metadata={"description":"RP-F10-K27J1-3 10819 BLK EU","part": "22450121", "fournisseur": "Ingram", "marque": "seiko", "prix": 176.96, "quantity": 22},
    )
]

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
chunks = text_splitter.split_documents(docs)
chroma_instance = Chroma(embedding_function= embedding_function, persist_directory=CHROMA_PATH, collection_name=COLLECTION_CSV)
# chroma_instance.delete(chroma_instance.get()['ids'])
chroma_instance.add_documents(chunks)
chroma_instance.persist()
print("There are", chroma_instance._collection.count(), "documents in the collection")
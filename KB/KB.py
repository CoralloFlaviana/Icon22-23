import pandas as pd
from pyswip import Prolog

import pandas as pd

# Carica il dataset heart.csv
dataset = pd.read_csv("icon\dataset\heart.csv")

# Apri il file KB.pl in modalità scrittura
with open("KB.pl", "w") as prolog_file:
    prolog_file.write(":- discontiguous cp/1.")
    # Itera attraverso le righe del dataset
    for index, row in dataset.iterrows():
        age = row[0]  # Il primo valore
        tipo = row[2]  # Il terzo valore

        # Scrivi il fatto Prolog nel file KB.pl con age e tipo
        prolog_fact = f'age({age}).\n'
        prolog_file.write(prolog_fact)

        prolog_fact = f'cp({tipo}).\n'
        prolog_file.write(prolog_fact)

# Regole Prolog da scrivere nel file
prolog_rules = """
% Regola 1: per determinare se una persona può avere un attacco cardiaco.
puo_avere_attacco_cardiaco(Eta, TipoDolore, AnginaEsercizio, Pendenza, NumeroVasi, RisultatoThallium, FrequenzaCardiaca, PiccoPrecedente, PuoAvereAttacco) :-
    % Condizioni per non avere un attacco cardiaco
    not((
        TipoDolore == 0, % Angina tipica
        AnginaEsercizio == 1, % Angina pectoris causata dall'esercizio
        (Pendenza == 0; Pendenza == 1), % Pendenza pari a 0 o 1
        NumeroVasi =< 3, % Numero di grandi vasi minore o uguale a 3
        RisultatoThallium == 3, % Risultato del test al thallium uguale a 3
        (Eta > 50; Eta < 30), % Età maggiore di 50 o minore di 30
        FrequenzaCardiaca < 130, % Frequenza cardiaca massima minore di 130
        PiccoPrecedente > 2.0 % Picco precedente alto
    )),
    % Se non soddisfa le condizioni, allora può avere un attacco cardiaco
    PuoAvereAttacco = no.

% Regola 2: per determinare se una persona può avere un attacco
% cardiaco, usando una somma di valori.
puo_avere_attacco_cardiaco_prob(Eta, TipoDolore, AnginaEsercizio, Pendenza, NumeroVasi, RisultatoThallium, FrequenzaCardiaca, PiccoPrecedente, PuoAvereAttacco) :-
    % Condizioni per non avere un attacco cardiaco
    (
        (TipoDolore == 0 -> ValoreCondizione1 = 0; ValoreCondizione1 = 1), % Angina tipica
        (AnginaEsercizio == 1 -> ValoreCondizione2 = 0; ValoreCondizione2 = 1), % Angina pectoris causata dall'esercizio
        ((Pendenza == 0; Pendenza == 1) -> ValoreCondizione3 = 0; ValoreCondizione3 = 1), % Pendenza pari a 0 o 1
        (NumeroVasi =< 3 -> ValoreCondizione4 = 0; ValoreCondizione4 = 1), % Numero di grandi vasi minore o uguale a 3
        (RisultatoThallium == 3 -> ValoreCondizione5 = 0; ValoreCondizione5 = 1), % Risultato del test al thallium uguale a 3
        ((Eta > 50; Eta < 30) -> ValoreCondizione6 = 0; ValoreCondizione6 = 1), % Età maggiore di 50 o minore di 30
        (FrequenzaCardiaca < 130 -> ValoreCondizione7 = 0; ValoreCondizione7 = 1), % Frequenza cardiaca massima minore di 130
        (PiccoPrecedente > 2.0 -> ValoreCondizione8 = 0; ValoreCondizione8 = 1) % Picco precedente alto
    ),
    % Somma dei valori delle condizioni
    SommaCondizioni is ValoreCondizione1 + ValoreCondizione2 + ValoreCondizione3 + ValoreCondizione4 + ValoreCondizione5 + ValoreCondizione6 + ValoreCondizione7 + ValoreCondizione8,
    % Se la somma è maggiore di 4, allora può avere un attacco cardiaco
    (SommaCondizioni > 4 -> PuoAvereAttacco = si; PuoAvereAttacco = no).



% Regola 3: per calcolare l'età media delle persone con attacco cardiaco
eta_media_persone_attacco_cardiaco(MediaEtà) :-
    findall(Age, (age(Age), Age > 0), ListeEtà), % Estrai le età delle persone con attacco cardiaco
    length(ListeEtà, NumeroPersone), % Conta il numero di persone con attacco cardiaco
    sum_list(ListeEtà, SommaEtà), % Somma le età
    MediaEtà is SommaEtà / NumeroPersone. % Calcola l'età media


% Regola 4: per determinare il tipo di dolore toracico più comune
tipo_dolore_piu_comune(TipoDolorePiuComune) :-
    findall(Tipo, (cp(Tipo), Tipo >= 0, Tipo =< 3), TipiDolore), % Estrai i tipi di dolore toracico
    list_to_set(TipiDolore, TipiUnici), % Rimuovi duplicati
    conta_tipi_dolore(TipiUnici, TipiDolore, TipoDoloreConteggi), % Conta quanti individui hanno ciascun tipo di dolore
    trova_tipo_piu_comune(TipoDoloreConteggi, TipoDolorePiuComune). % Trova il tipo di dolore più comune

% Regola 5: per contare quanti individui hanno ciascun tipo di dolore toracico
conta_tipi_dolore([], _, []).
conta_tipi_dolore([Tipo|Resto], TipiDolore, [(Tipo, Conteggio)|RestoConteggi]) :-
    conta_occorrenze_tipo(Tipo, TipiDolore, Conteggio),
    conta_tipi_dolore(Resto, TipiDolore, RestoConteggi).

% Regola 6: per contare quante volte un tipo di dolore appare nella lista
conta_occorrenze_tipo(_, [], 0).
conta_occorrenze_tipo(Tipo, [Tipo|Resto], Conteggio) :-
    conta_occorrenze_tipo(Tipo, Resto, ConteggioResto),
    Conteggio is ConteggioResto + 1.
conta_occorrenze_tipo(Tipo, [_|Resto], Conteggio) :-
    conta_occorrenze_tipo(Tipo, Resto, Conteggio).

% Regola 7: per trovare il tipo di dolore più comune
trova_tipo_piu_comune([(Tipo, Conteggio)], Tipo).
trova_tipo_piu_comune([(Tipo1, Conteggio1), (Tipo2, Conteggio2)|Resto], TipoPiuComune) :-
    (Conteggio1 >= Conteggio2 -> trova_tipo_piu_comune([(Tipo1, Conteggio1)|Resto], TipoPiuComune)
    ; trova_tipo_piu_comune([(Tipo2, Conteggio2)|Resto], TipoPiuComune)).



"""

# Apri il file KB.pl in modalità append e scrivi le regole Prolog
with open("KB.pl", "a") as prolog_file:
    prolog_file.write(prolog_rules)


# Chiudi il file KB.pl
prolog_file.close()


from pyswip import Prolog

# Crea un oggetto Prolog
prolog = Prolog()

# Carica il file KB.pl
prolog.consult("KB.pl")

# Esegui la query
print(list(prolog.query("eta_media_persone_attacco_cardiaco(MediaEtà)")))

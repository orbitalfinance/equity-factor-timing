# Metodologia

Sintesi tecnica della pipeline di **Equity Factor Timing**. Per il dettaglio
completo vedi la relazione in [MANCUSO_Santo_PW.pdf](MANCUSO_Santo_PW.pdf); per
l'esecuzione passo-passo vedi il notebook in `Code/`.

## Schema della pipeline

```
                 Parte 1                     Parte 2                    Parte 3
        +-----------------------+   +----------------------+   +---------------------+
Dati -> | Regimi di mercato     |-> | Selezione fattori    |-> | Simulazione         |
        | (2 stadi ML)          |   | (per regime)         |   | di portafoglio      |
        +-----------------------+   +----------------------+   +---------------------+
          |                           |                          |
   K-means su drawdown         Random Forest per            Pesi = probabilita' dei
   + classificatore su         regime "normale" e           fattori; ribilancio
   macro/turbolenza            "correzione"                 mensile; metriche vs
                                                            benchmark
```

## Parte 1: regimi di rischio del mercato

Approccio a **due stadi**:

1. **Etichettatura non supervisionata.** Si calcola il drawdown rolling a 3 mesi
   dell'S&P 500 e lo si aggrega a frequenza mensile. Un **K-means** (numero di
   cluster scelto con *elbow method* e *silhouette score*) separa i mesi in
   regimi. Nella configurazione principale i regimi sono due: **normale** e
   **correzione**.
2. **Classificazione supervisionata.** Un classificatore impara a prevedere il
   regime del mese a partire dalle feature descritte sotto. Si confrontano
   **Random Forest**, **Gaussian Naive Bayes** e **SVC**, combinati in uno
   **StackingClassifier** i cui iper-parametri sono ottimizzati con **HyperOpt**
   (validazione temporale con `TimeSeriesSplit`). Le classi sbilanciate sono
   gestite con **SMOTE**.

### Feature del classificatore di regime

- **Turbolenza finanziaria** (indice di Kritzman-Li): distanza di Mahalanobis del
  vettore di rendimenti settoriali/treasury rispetto alla media storica. Vedi
  `src/turbulence.py`.
- **Variabili macroeconomiche** (da `Data/Data_Macro.xlsx`), rese stazionarie
  dove necessario e verificate con il test **Augmented Dickey-Fuller**.
- I dati sono scalati con **Min-Max** e la selezione delle feature usa
  **Random Forest importance** e **mutual information**.

## Parte 2: selezione dei fattori

Sei fattori azionari S&P 500: **value, growth, momentum, low-volatility,
quality, small-cap**. Per ogni mese il fattore "vincente" e' quello a rendimento
massimo; l'etichetta e' **one-hot**. Si addestrano **due Random Forest separati**,
uno per il regime *normale* e uno per il regime di *correzione*, ciascuno
produce le **probabilita'** che ogni fattore sia il vincente del mese successivo.
Lo split e' temporale (train fino al 2010, test dal 2010).

## Parte 3: simulazione di portafoglio

Le probabilita' dei fattori diventano i **pesi** di un portafoglio ribilanciato
mensilmente (le probabilita' sono ritardate di un periodo per evitare
look-ahead). Si costruiscono e confrontano piu' strategie:

- Portafoglio **factor-timing** guidato dalle probabilita' (6 fattori).
- **Equally-weighted** sui fattori.
- **Risk-parity** (via `Riskfolio-Lib`).
- Benchmark **S&P 500** buy-and-hold.

### Metriche di valutazione

Calcolate su base mensile e giornaliera da `src/metrics.py`:

| Metrica | Significato |
| --- | --- |
| `AnnRet` | Rendimento annualizzato |
| `AnnVol` | Volatilita' annualizzata |
| `Sharpe` | Rendimento per unita' di rischio |
| `MaxDD` | Massima perdita da picco a valle |
| `Calmar` | Rendimento annualizzato su massimo drawdown |

I valori numerici per ciascuna strategia sono prodotti dalle celle della Parte 3
del notebook.

## Note di riproducibilita'

- I modelli addestrati sono salvati in `Code/*.pkl` e ricaricati dal notebook;
  le celle di training pesanti sono disattivate dal magic `%%skip`.
- Riaddestrando (rimuovendo `%%skip`) i risultati possono variare leggermente per
  seed e versioni delle librerie. Vedi `requirements.txt` per le versioni usate.

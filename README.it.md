[English](README.md) | **Italiano**

# **Equity Factor Timing**: allocazione dinamica tra fattori azionari S&P 500 guidata da algoritmi di Machine Learning

[![CI](https://github.com/orbitalfinance/equity-factor-timing/actions/workflows/ci.yml/badge.svg)](https://github.com/orbitalfinance/equity-factor-timing/actions/workflows/ci.yml)

Un sistema che riconosce il **regime di rischio** del mercato e, in base ad esso, sceglie **quale fattore azionario** (value, growth, momentum, low-volatility, quality, small-cap) sovrappesare, costruendo un portafoglio ribilanciato mensilmente e confrontato con benchmark classici.

## A cosa serve

- **Timing dei fattori**: capire quale stile azionario tende a vincere in ciascuna fase di mercato, invece di puntare su un solo fattore statico.
- **Rilevamento dei regimi di mercato**: distinguere fasi *normali* da fasi di *correzione* a partire dai drawdown e da variabili macro.
- **Backtest di strategie**: confrontare un portafoglio *factor-timing* con S&P 500 buy-and-hold, equally-weighted e risk-parity, con metriche di performance standardizzate.
- **Base di ricerca / didattica**: pipeline end-to-end (dati → clustering → classificazione → allocazione → backtest) riutilizzabile e ispezionabile in un unico notebook.

## Idea centrale (leggere prima)

Il progetto poggia su un'**architettura ML a due stadi**. *Primo stadio*: un modello di **clustering non supervisionato** (K-means sui drawdown mensili a 3 mesi dell'S&P 500) etichetta ogni mese come regime *normale* o di *correzione*; un **classificatore supervisionato** (Random Forest / Naive Bayes / SVC combinati in uno **stacking** ottimizzato con **HyperOpt**) impara poi a prevedere quel regime a partire da variabili **macroeconomiche** e dalla **turbolenza finanziaria** di Kritzman-Li. *Secondo stadio*: per ciascun regime un **Random Forest dedicato** stima la probabilità che ogni fattore sia il *vincente* del mese successivo. Queste probabilità diventano i **pesi** di un portafoglio ribilanciato mensilmente, la cui equity curve viene misurata con rendimento annuo, volatilità, **Sharpe**, **max drawdown** e **Calmar** e confrontata con i benchmark. In breve: *prima capisci in che mondo sei, poi scegli lo stile giusto per quel mondo.*

## Come usarlo

1. **Clona il repository e posizionati nella cartella.**
   ```bash
   git clone https://github.com/orbitalfinance/equity-factor-timing.git
   cd equity-factor-timing
   ```

2. **Crea l'ambiente e installa le dipendenze** (Python 3.12).
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Avvia Jupyter ed esegui il notebook principale.** Il notebook individua da solo la radice del progetto (risalendo le cartelle finché trova `.git`/`requirements.txt`/`pyproject.toml`), quindi puoi lanciarlo da qualsiasi directory interna al repository.
   ```bash
   jupyter lab Code/equity_factor_timing_final.ipynb
   ```
   Esegui le celle in ordine: le sezioni di *training* più pesanti sono disattivate dal magic `%%skip` e i modelli già addestrati vengono ricaricati dai file `.pkl`. Per riaddestrare da zero, rimuovi `%%skip` dalle celle interessate. I dati necessari sono già inclusi in `Data/`.

4. **(Opzionale) Sviluppo: test, lint e hook anti-segreto.**
   ```bash
   pip install -e ".[dev]"
   pytest -q                 # esegue la suite in tests/
   ruff check src tests      # lint + ordinamento import
   pre-commit install        # blocca segreti e file troppo grandi a ogni commit
   ```

### Mappa del repository

| File / cartella | Scopo |
| --- | --- |
| `Code/equity_factor_timing_final.ipynb` | **Notebook principale**: pipeline completa in 3 parti (regimi → fattori → portafoglio). Entry point del progetto. |
| `src/paths.py` | Radici del progetto (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`) indipendenti dalla cartella di avvio. |
| `src/metrics.py` | Metriche di performance **dai rendimenti** (importato dal notebook come `pm`). È il modulo che l'analisi usa davvero. |
| `src/equity_metrics.py` | Le stesse metriche calcolate **dall'equity curve**. Port fedele del codice originale del Project Work, conservato come riferimento — **non usato dal notebook**. |
| `src/turbulence.py` | Indice di **turbolenza finanziaria** di Kritzman-Li (feature del classificatore di regime). |
| `tests/` | Suite `pytest` per metriche e turbolenza. |
| `Code/best_ho_model_new2.pkl` | Modello di **stadio 1** (stacking ottimizzato con HyperOpt) per la classificazione del regime di mercato. |
| `Code/best_rf_model_new2.pkl` | Random Forest di riferimento per la classificazione del regime. |
| `Code/best_normal_factors_new2.pkl` | Modello di **stadio 2** per la selezione del fattore vincente nel regime *normale*. |
| `Code/best_correction_factors_new2.pkl` | Modello di **stadio 2** per la selezione del fattore vincente nel regime di *correzione*. |
| `Data/SP500.xlsx` | Serie storica prezzi dell'indice S&P 500. |
| `Data/S&P_factors.xlsx` | Serie storiche dei 6 indici fattoriali S&P 500. |
| `Data/S&P_sectors_bb.xlsx` | Serie storiche settoriali S&P (input per la turbolenza finanziaria). |
| `Data/Data_Treasuries.xlsx` | Rendimenti Treasury USA a 1, 2 e 10 anni. |
| `Data/Data_Macro.xlsx` | Variabili macroeconomiche usate come feature del classificatore di regime. |
| `docs/METHODOLOGY.it.md` | Sintesi tecnica della metodologia (schema pipeline, feature, metriche). |
| `docs/MANCUSO_Santo_PW.pdf` | Relazione completa del Project Work. |
| `docs/MANCUSO_Santo_Discussione_15min.pptx` | Presentazione di discussione (15 min). |
| `requirements.txt` | Dipendenze Python pinnate alle versioni usate per i risultati. |
| `pyproject.toml` | Metadati del progetto e configurazione di `ruff`/`pytest`. |
| `.pre-commit-config.yaml` | Hook anti-segreto e di lint eseguiti prima di ogni commit. |
| `.github/workflows/ci.yml` | CI GitHub Actions: lint e test a ogni push/PR. |

## Note / Vincoli

- **Ordine di esecuzione**: il notebook è **stateful**. Esegui le celle dall'alto verso il basso; saltarne alcune lascia variabili non definite.
- **Percorsi**: gestiti da `src/paths.py` in modo indipendente dalla cartella di avvio; i dati restano in `Data/`, i modelli `.pkl` in `Code/`. Il notebook aggiunge automaticamente la radice del progetto a `sys.path` per importare `src`.
- **Dati con licenza**: i file in `Data/` derivano da **Refinitiv Eikon / S&P / Bloomberg**. Sono inclusi per riproducibilità didattica; verifica i termini di licenza prima di riutilizzarli o ridistribuirli in altri contesti.
- **Ambiente**: ricrea l'ambiente con `requirements.txt`; non versionare il virtualenv (è già escluso in `.gitignore`).
- **Riproducibilità dei modelli**: i `.pkl` riflettono l'addestramento originale. Riaddestrando (rimuovendo `%%skip`) i risultati numerici possono variare leggermente per via di seed e versioni delle librerie.

## Paper di riferimento

L'architettura a due stadi implementata qui segue l'approccio proposto in:

> DiCiurcio, K. J., Wu, B., Xu, F., Rodemer, S., e Wang, Q. (2024).
> "Equity Factor Timing: A Two-Stage Machine Learning Approach."
> *The Journal of Portfolio Management*, 50 (3).
> https://www.pm-research.com/content/iijpormgmt/50/3

L'articolo è ad accesso a pagamento e la sua licenza non ne consente la
ridistribuzione: **non** è incluso in questo repository, va richiesto all'editore.

## Crediti / Provenienza

- **Autore**: Santo Mancuso.
- **Contesto**: Project Work del **Master in Finance**, Politecnico di Milano, Graduate School of Management (GSOM).
- **Anno**: 2024.
- **Dati**: Refinitiv Eikon, indici S&P 500 (prezzo, fattori, settori), Treasury USA, variabili macroeconomiche.
- **Licenza**: codice rilasciato sotto licenza **MIT** (vedi `LICENSE`). I dati di mercato restano soggetti alle licenze dei rispettivi fornitori.

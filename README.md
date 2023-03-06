# Forschungsprojekt
Disaggregation von Lastprofilen durch Zeitreihenanalyse

## Repository-Struktur
Das Repository ist so strukturiert, dass jeder Branch eine separate Methode aus dem Bericht abbildet. Das Mapping sieht wie folgt aus:
[Branch : Kapitel im Bericht]
- master : 3.1
- SumProfiles : 3.2.1
- Undersampled_CNN : 3.2.2 (nur CNN)
- ConcatModel : 3.2.2 (CNN+MLP)
- GAF : 3.3

Der Branch kann lokal mittels `git switch <branchName>` gewechselt werden. Aufgrund der Dateigrößen sind keine Datensätze im Repository hinterlegt. Die Dateien sind also nur zum Anschauen brauchbar.

## Dateien im Projekt
- Das `DataSourcing.ipynb` Notebook minimiert die Rohdaten auf die ausgewählten Zielgeräte und bringt sie in ein uniformes Format.
- In `Main.ipynb` passiert das restliche Preprocessing sowie das Training und die Evaluation der Modelle
- `Evaluation.ipynb` ist nur für den master-Branch relevant. Dort wurde das in `Main.ipynb` trainierte Modell auf dem Datensatz aus dem Branch SumProfiles getestet. Auf allen anderen Branches ist die Datei unbenutzt.
- `utils.py` enthält wiederverwendete Funktionen wie den Code um die windows aus der Zeitreihe zu generieren.

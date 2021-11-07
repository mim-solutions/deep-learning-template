# Code style

## Formatowanie i lintery

Trzymamy się [PEP 8](https://www.python.org/dev/peps/pep-0008/), oprócz długości linii. Do docstringów nie mamy ściśle ustalonego stylu, ale trzymamy się [PEP 257](https://www.python.org/dev/peps/pep-0257/).

Używamy typów i `mypy`, oraz lintera `flake8` z default'ową konfiguracja (z drobnymi poprawkami które pobierają z `setup.cfg`).

W VS Code wystarczy je włączyć w ustawieniach (Ctrl+Shift+P, Open Setting (globalnie) lub Open Workspace Settings, w UI lub JSON (jak kto woli)): `python.linting.mypyEnabled`, `python.linting.flake8Enabled` i po każdym zapisie pliku uruchamiają się na tym pliku (warto sobie sprawdzić czy dla złych typów widzimy ostrzeżenie w "Problems" po zapisie pliku).
Jeśli VS Code poprosi o `pip install mypy flake8` – zaakceptuj.
Wbudowane wsparcie działa coraz lepiej i wtyczki VS Code nie są już potrzebne, a mogą powodować błędy (poza domyślnie zainstalowanymi wtyczkami `Python` i `Pylance`).
W ustawieniu `python.linting.ignorePatterns` można sobie dodać do listy `**/stripped/**`.
Więcej szczegółów [tu](https://code.visualstudio.com/docs/python/linting).

W konsoli można odpalić po prostu `mypy .` i `flake8` z root repo (żeby pobrały stamtąd `setup.cfg`), najlepiej z wewnątrz środowiska conda/pip (bo w systemie może być np. starsza wersja `mypy`).

W razie wątpliwości/braku cierpliwości co do formatowania, najprościej użyć autoformatera, np. `autopep8` (poprawia tylko whitespace); w VS Code Ctrl+Shift+P i szukamy "Format document"/"Format selection", szczegóły [tu](https://code.visualstudio.com/docs/python/editing#_formatting). Alternatywnie `black` poprawia wszystko (ale potrafi też trochę zepsuć i nie potrafi "Format selection"; wymaga ustawienia `python.formatting.provider`).

# Nasze konwencje

* do dat i czasów używamy obiektów `date`, `time` i `datetime` z modułu [datetime](https://docs.python.org/3/library/datetime.html).
Chyba że w dataframe'ach: tam używamy `datetime64` z pandas/numpy zamiast `date` i `datetime` (`time` pozostaje bez zmian).
Żeby zaznaczyć że coś jest datą bez czasu można pisać `datetime64[D]` w `astype()` oraz w komentarzach, ale `dtype` będzie
się nadal wypisywał jako `datetime64[ns]`.
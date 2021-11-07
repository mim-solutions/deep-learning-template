Opisujemy zawartość katalogu `hydra_lightning` oraz jak z niej korzystać.

# Taski

Katalog `tasks` zawiera podkatalogi dla konkretnych tasków. Każdy task to generalnie: 
- konkretny zbiór testowy 
- konkertne metryki do policzenia na tym zbiorze.

Dla każdego taska chcemy mieć więc spójny sposób: 
- generowania zbioru testowego (wraz z transformacjami)
- liczenia konkretnych metryk na tym zbiorze
- logowania wyników (w wandb)

Rzeczy zmienne w tasku to:
- Modele
- Hiperparametry modeli
- Sposoby przejścia przez modele, w tym regularyzacje modeli
- Zbiory treningowe i walidacyjne
- Transformacje i augmentacje zbioru treningowego

## DataModule

Sposób generowania zbioru testowego powinien być zdefiniowany za pomocą DataModule - jest to Lightningowe opakowanie na PyTorch-owy `Dataset`. DataModuły są zdefiniowane w katalogu `datamodules` dla każdego taska. Możemy mieć różne DataModuły gdybyśmy chcieli np. generować nieco inny zbiór treningowy. Jednak zbiory testowe powinny być takie same we wszystkich DataModułach w danym tasku. Możliwe, że w praktyce zazwyczaj będzie tylko jeden DataModuł per task (`DefaultDataModule`).

DataModuł ma zawsze pewną określoną strukturę, zadaną przez `Lightning` i dodatkowo wyodrębnioną przez naszą klasę `shared.datamodules.base.BaseDataModule`. Dziedziczenie z `BaseDataModule` wymaga nadpisania metod `prepare_data` oraz `setup`. Czasochłonne operacje powinny dziać się w `prepare_data`, natomiast w `setup` powinniśmy tworzyć atrybuty takie jak `self.datasets`. Ma to znaczenie przy `DDP` - `prepare_data` odpalamy raz w głównym wątku, natomiast `setup` odpala się osobno w każdym procesie.

## Module

Drugi główny katalog w tasku to `modules`. Znajdują się w nim `Modules` - Lightningowe opakowania na PyTorch-owe modele. Nałożyliśmy dodatkowe konwencje na `LightningModule` - por. `shared.modules.base.BaseModule`. Jeśli dziedziczymy z tego Modułu, musimy podać konfigurację dla modelu `model_config` oraz słowniki `optimizer_spec` i `scheduler_spec`. Klasa `BaseModule` tworzy na podstawie tych słowników optimizery i schedulery zgodnie z konwencją z `robbytorch`, natomiast `model_config` użytkownik może zdefiniować samodzielnie i wykorzystać go w metodzie `initialize_model` - parametr ten będzie dostępny jako atrybut `self.hparams.model_config` (jest to efekt wywołania `self.save_hyperparameters()` w `__init__` - `self.hparams` zachowuje parametry inicjalizacji Modułu, które zachowują się także w checkpointach zapisywanych przez Lightning). Model przypisujemy do Modułu jako atrybut `self.model`.

Uwagi:
- Generalnie Lightning spodziewa się 3 osobnych metod (`training_step`, `validation_step` oraz `test_step`); mogą się one pokrywać, jednak często chcemy liczyć dodatkowe metryki na etapie walidacji albo dodatkowe loss-y na etapie treningu;
- `self.hparams.model_config` jest typu `DictConfig` - jest to opakowanie na config przekazany za pomocą Hydry (por. niżej). Mamy pełną dowolność w definiowaniu kształtu tego configa, dzięki czemu prototypowanie nowych modeli nie trwa dłużej, niż w jupyterze.

## Shared

Katalog, w którym przechowujemy m.in. nadklasy dla Modułów i DataModułów.

## Logowanie do WandB

Zadanie sprowadza się do podania konfiguracji `configs/logger/wandb.yaml` - tworzymy obiekt klasy `WandbLogger` z paczki PytorchLightning.

## Hydra - tworzenie konfiguracji

Dzięki paczce [Hydra](https://hydra.cc/) możemy tworzyć `robust` konfiguracje. Hydra ma dość dobre [docsy](https://hydra.cc/docs/intro), warto przejść przez [Basic Tutorial](https://hydra.cc/docs/tutorials/intro).

Ogólna struktura jest taka - w katalogu `configs` mamy drzewo konfiguracji. Na każdym poziomie drzewa mamy różne parametry do wyboru, w tym różne podrzewa. Root-em jest zawsze plik `configs/config.yaml` - ten plik zawsze odpalamy, jednak w trakcie wywołania możemy dowolnie zmieniać wybór podkonfiguracji. Możemy to zrobić za pomocą:
- command-line argumentów - przy wywołaniu w terminalu
- argumentu `overrides` w funkcji `train.load_config` - przy wywołaniu w Pythonie
- w pliku `configs/experiment/<nazwa_taska>/<nazwa_eksperymentu>.yaml`

Generalnie drzewo konfiguracji powinno być względnie stałe, w szczególności nie powinno zależeć od liczby tasków, modeli ani eskperymentów - z wyjątkiem katalogu `configs/experiment`. W tym katalogu trzymamy konkretne konfiguracje tasków, które z jakichś powodów chcemy zatrzymać w repo.

Ważne elementy Hydry:

### Variable interpolation

Hydra pod spodem używa paczki [OmegaConf](https://omegaconf.readthedocs.io/en/2.0_branch/index.html), która w szczególności umożliwia [variable interpolation](https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#variable-interpolation). Dzięki temu możemy w wielu miejscach konfiguracji odwoływać się do tej samej wartości bez duplikacji.

### Common patterns

Ważne patterny, które używamy w projekcie:
- [Packages/Config groups](https://hydra.cc/docs/advanced/overriding_packages) - najważniejsza koncepcja, czyli konfigi są ułożone w config grupy albo paczki, dzięki którym można składać konfigi modularnie w drzewo konfigów.
- [Defaults List](https://hydra.cc/docs/advanced/defaults_list/) - można definiować domyślne podkonfigi oraz je nadpisywać
- [Extending configs](https://hydra.cc/docs/patterns/extending_configs) - najprostszy schemat - można nadpisywać konkretne liście podconfigów
- [Experiment](https://hydra.cc/docs/patterns/configuring_experiments) - bardzo ważny schemat, dzięki któremu możemy tworzyć konfigi eksperymentów, zachowując przy tym stałe rozmiary drzewa configów (poza katalogiem `configs/experiment`). Aby załadować eksperyment wystarczy nadpisać klucz `experiment`, np.:
`experiment=left_right_ovary/robust_backbone`, por. `notebooks/basic_left_right-lightning`.
- [Specjalizacja configów](https://hydra.cc/docs/patterns/specializing_config) - dzięki temu schematowi możemy ograniczyć liczbę konfigów - np. jeśli mamy ogólny model, to nie musimy tworzyć osobnego configa dla tego modelu dla każdego taska - wystarczy wykorzystać variable interpolation, aby wskazać odpowiedni plik, por. `configs/module.robust_backbone` i klucz `_target_: tasks.${task.name}.modules.robust_backbone.RobustBackbone`.

### Parameter search

Hydra umożliwia konfigurację grid search parametrów za pomocą [sweepów](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).

## Spięcie wszystkiego w całość i potencjalnie zabawa w notatniku
 
Przykładowe użycie znajduje się w [notatniku](notebooks/example_task.ipynb). Config ładujemy za pomocą funkcji `train.load_config`, a następnie ładujemy obiekty `trainer, module, datamodule` za pomocą funkcji `train.load_from_config`. Mając te 3 obiekty możemy odpalić trenowanie za pomocą `trainer.fit(model=module, datamodule=datamodule)`. Możemy też debugować lub prototypować nasz model, odwołując się do atrybutu `module.model`.

Parametry configu możemy zmieniać, przekazując do funkcji `train.load_config` parametr `overrides`. Więcej informacji znajduje się w docstringu tej funkcji. Należy w szczególności zwrócić uwagę, że przy nadpisywaniu defaults list używamy slashy zamiast kropek.

# Ustawienie uprawnień

Ponieważ często na serwerach notatniki odpalamy z poziomu innego użytkownika, niż nasz, polecamy ustawić odpowiednie uprawnienia plików w repozytorium - najlepiej z poziomu katalogu zawierającego wszystkie nasze repozytoria (załóżmy, że katalog ten nosi nazwę `projects`).

Poniższe ustawienie uprawnień umożliwiają modyfikację lokalnej kopii repozytorium przez głowne konto użytkownika o nazwie $USER oraz lustrzanemu użytkownikowi w JupyterHubie o nazwie jupyter-$USER, bez konfliktu uprawniń w dostępie do plików repozytorium. W katalogu  `projects` znajdują się repozytoria/pliki modyfikowalne przez obu userów.

```
setfacl -R  -d -m u:$USER:rwX projects/
setfacl -R  -m u:$USER:rwX projects/
setfacl -R  -d -m u:jupyter-$USER:rwX projects/
setfacl -R  -m u:jupyter-$USER:rwX projects/
```
Dzięki temu użytkownik jupyter-$USER może edytować pliki użytkownika $USER i odwrotnie. 

Stan uprawnień sprawdzamy przez:

```
getfacl projects
```

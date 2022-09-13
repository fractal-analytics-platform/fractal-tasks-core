To download the example dataset from Zenodo, you can use (from this folder):
```bash
./fetch_test_data_from_zenodo.sh
```
Note that this currently requires the `zenodo-get` package. At the moment this is an optional dependency, so just use `pip install zenodo-get` if you don't have it available.
TODO: this will be fixed later, either by adding it as a mandatory (dev) dependency, or by bypassing this external package.

After this step, you may run the worfklow directly as
```
poetry run python run_worfklow.py
```

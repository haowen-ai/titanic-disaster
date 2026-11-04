# titanic-disaster

Predicts who survived the Titanic with logistic regression, once in Python and once in R. Each version runs in its own Docker container, loads the Kaggle data, cleans it, fits the model and prints every step to the terminal.

The dataset is not in this repo. Step 2 below shows how to download it.

## What you need

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux), installed and **running**
- A free [Kaggle](https://www.kaggle.com) account to download the data

Check Docker is running:

```bash
docker --version
docker info
```

`docker info` should print details about the server. If it says it cannot connect to the Docker daemon, open Docker Desktop and wait until it says it is running.

## Step 1: clone the repo

```bash
git clone https://github.com/haowen-ai/titanic-disaster.git
cd titanic-disaster
```

Run every command below from this `titanic-disaster` folder.

## Step 2: download the data

1. Sign in to Kaggle and open the competition data page: <https://www.kaggle.com/competitions/titanic/data>
2. If Kaggle asks you to join the competition or accept its rules, do that first.
3. Click **Download All**. You get `titanic.zip` with `train.csv`, `test.csv` and `gender_submission.csv`.
4. Unzip it into `src/data`:

   macOS / Linux:

   ```bash
   unzip ~/Downloads/titanic.zip -d src/data
   ```

   Windows (PowerShell):

   ```powershell
   Expand-Archive $HOME\Downloads\titanic.zip -DestinationPath src\data
   ```

   If your browser saved the file somewhere else, change the path. You can also unzip it by hand and drag `train.csv` and `test.csv` into `src/data`.

5. Check the files are in place:

   ```bash
   ls src/data
   ```

   You should see `train.csv` and `test.csv` (plus `gender_submission.csv`, which the code does not use).

Optional, with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) set up and an API token in place:

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

`src/data` is listed in `.gitignore`, so the data never gets committed.

## Step 3: run the Python version

```bash
docker build -t titanic-python -f src/python/Dockerfile .
docker run --rm titanic-python
```

The `.` at the end of the build command matters: it tells Docker to build from the repo root, so the image can copy both the code and `src/data`. The first build downloads the Python image and packages and takes a minute or two.

## Step 4: run the R version

```bash
docker build -t titanic-r -f src/r/Dockerfile .
docker run --rm titanic-r
```

The first build downloads the R image and installs `dplyr` (fixed versions from a dated CRAN snapshot), which takes a few minutes.

## Optional: get the Kaggle submission files

Both scripts save their test predictions as a Kaggle submission file (`PassengerId,Survived`): `submission_python.csv` and `submission_r.csv`. With the plain `docker run` above the file is deleted along with the container when it exits. To keep it, share `src/data` with the container when you run it:

macOS / Linux:

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-r
```

Windows (PowerShell):

```powershell
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-python
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-r
```

The files show up in `src/data` (ignored by git). To score one, open <https://www.kaggle.com/competitions/titanic/submit>, upload the file and click **Submit**.

## What the output shows

Both scripts print the same steps:

1. **Load `train.csv`**: 891 rows, 12 columns.
2. **Explore**: column types, first rows, missing values (Age 177, Cabin 687, Embarked 2), survival rate overall and by sex, class and port.
3. **Clean and add features**, printing each change:
   - `Title` taken from the name (Mr, Mrs, Miss, Master, Rare)
   - missing `Age` filled with the median age of each title
   - missing `Embarked` filled with the most common port, missing `Fare` with the median fare
   - new `FamilySize`, `IsAlone`, `HasCabin`
   - `PassengerId`, `Name`, `Ticket`, `Cabin`, `SibSp`, `Parch` dropped
   - `Pclass`, `Sex`, `Embarked`, `Title` turned into dummy columns
   All fill values come from the training data and are reused as-is on the test data.
   The R model leaves out `Sex`: `Title` already separates men from women (only the few `Rare` titles mix both), and plain `glm` (no penalty) gives huge, unstable coefficients when both are in. The Python model keeps `Sex` because scikit-learn's logistic regression is regularized by default.
4. **Holdout check**: 80% of `train.csv` is used to fit the model and 20% is kept aside. Prints accuracy on both parts.
5. **Fit on all of `train.csv`**: prints the training accuracy and the model coefficients.
6. **Load `test.csv` and predict**: 418 rows; prints how many passengers are predicted to survive and the first 10 predictions.
7. **Test accuracy**: see below.

### Results

| | Python | R |
|---|---|---|
| Accuracy on the 80% used to fit | 0.837 | 0.839 |
| Holdout accuracy (20%) | 0.838 | 0.843 |
| Training accuracy (all 891 rows) | 0.838 | 0.838 |
| Test passengers predicted to survive | 168 of 418 | 170 of 418 |
| **Test accuracy (Kaggle score)** | **0.770** | **0.770** |

The two holdout sets are not the same passengers (each language splits with its own random number generator), so the numbers differ slightly.

### About test accuracy

Kaggle's `test.csv` does not include the `Survived` column; Kaggle keeps those answers to score submissions. So the scripts cannot compute accuracy on `test.csv` themselves and say so in the output. To get the real test accuracy, I uploaded both submission files to Kaggle: each scored **0.77033** (322 of 418 passengers right). The two models disagree on only 2 passengers and end up with the same number correct.

The test score is lower than the holdout score (about 0.84). That gap is common on this competition: the holdout set is only about 180 passengers, so its accuracy moves a few points depending on which passengers land in it.

If you run the scripts with a `test.csv` that does have a `Survived` column, they print the test accuracy directly.

## Repo layout

```
.
├── .gitignore
├── CODEOWNERS
├── README.md
└── src
    ├── data/                  # train.csv and test.csv go here; submission files are saved here (not committed)
    ├── python
    │   ├── Dockerfile
    │   ├── requirements.txt   # pinned package versions
    │   └── titanic.py
    └── r
        ├── Dockerfile
        ├── install_packages.R # pinned package versions
        └── titanic.R
```

## Troubleshooting

- **`train.csv not found`**: the files are not in `src/data`, or you added them after building. Put them in `src/data` and run the `docker build` command again.
- **`Cannot connect to the Docker daemon`**: Docker Desktop is not running. Start it and try again.
- **`failed to read dockerfile` / `not found`**: you are not in the repo root, or the `.` at the end of `docker build` is missing.
- **Linux `permission denied` on the Docker socket**: put `sudo` in front of the docker commands, or add your user to the `docker` group.
- Apple Silicon and Intel machines both work; the base images support both.

## Clean up

```bash
docker rmi titanic-python titanic-r
```

# Revelio-Labs-Assessment


## Setup Instructions

#### Move into top-level directory
```
cd Revelio-Labs-Assessment
```

#### Install environment
```
conda env create -f environment.yml
```

#### Activate environment
```
conda activate rlabs
```

#### Install package
```
pip install -e src/rlabs
```

Including the optional -e flag will install the package in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.

#### Fetch data
```
python -m rlabs fetch
```

#### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter kernel to run notebooks.

### Preparing data for proteus

In the Proteus' root folder you will find a script to convert pipe-separated files to our binary inputs. The usual process is as follows:

#### Generating the data 
Clone the generator. As an example greenlion's ssb-dbgen:
```sh
git clone https://github.com/greenlion/ssb-dbgen
cd ssb-dbgen
sed s/=MAC/=LINUX/g -i makefile # for Linux only
make
cd .. # Return to the project root folder
```


Create a folder under `tests/inputs/` with an identifier for the dataset and a subfolder `raw/` for the pipe-separated files:

```sh
cd proteus/tests/inputs/
mkdir -p ssb100/raw
```

Enter this subfolder and generate your data:

```sh
cd ssb100/raw
cp ../../../../ssb-dbgen/dists.dss .
../../../../ssb-dbgen/dbgen -s 100 -T a
cd ..
```

#### Converting the data to the proteus binary format
Return to the *parent* of the raw folder and run the python script from the root of the folder:
```sh
python3 $proteus_root/tools/prepare_data.py --schema-name ssb100 --ddl ssb
```
This script both converts the pipe-separated files to the Proteus binary format and generates a `catalog.json` describing the dataset. 

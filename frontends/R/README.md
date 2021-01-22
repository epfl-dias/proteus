# PROJECT_NAME

## Introduction

This is a template for projects, containing a default license, as well as
examples for the documentation.

In order to use these templates, please:
 1. Update the year in the license file
 2. Adapt this README file
 3. Adapt `docs/_config` so that the links points to the correct URLs,
    also update the project name and short description.
 4. As needed, write documentation, using the templates in `docs`

## Requirements

### Hardware

 * **Processor:** 1GHz CPU
 * **RAM:** 1 MB
 * **Available storage space:** 1 MB

### Software

 * Ubuntu
 * LLVM

## Quick start

## Building from sources

To build this project, you will need to run the following:

```sh
# make
R CMD build R
```

### Installation

To install the software on the system you can use:

```sh
# make install
R -e "install.packages(c('DBI', 'lazyeval', 'jsonlite', 'dbplyr', 'dplyr', 'rlang', 'RJDBC', 'purrr', 'stringi', 'Rcpp'), repos='http://cran.rstudio.com/')"
R -e "install.packages('ViDaR_0.1.5.tar.gz', dependencies = TRUE)"
```

### Usage

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin vehicula pretium
quam sit amet facilisis. Class aptent taciti sociosqu ad litora torquent per
conubia nostra, per inceptos himenaeos. Curabitur metus sapien, rhoncus vitae
eleifend nec, convallis vel nunc. Nulla metus mauris, porta eu porta eu,
vulputate et est. Suspendisse lacinia leo vel auctor aliquet. Maecenas non arcu
libero. Nulla ut eleifend dui. Cras bibendum pharetra facilisis. Proin mattis
libero non pharetra tristique. Nam massa nulla, ultrices pharetra quam a,
fermentum placerat dolor. Nullam mollis libero et neque lobortis, id dignissim
lectus dignissim. Maecenas ligula enim, congue in ornare vel, volutpat ut ante.

```sh
<command to run the project>
```

Quisque id velit erat. Quisque gravida posuere nisi, quis vestibulum urna
egestas in. Curabitur accumsan eget elit quis molestie. Nunc vestibulum, dolor
nec interdum iaculis, enim mauris auctor ligula, ac suscipit velit ex eu lectus.

## Documentation

For more information, please refer to the [documentation](https://epfl-dias.github.io/PROJECT_NAME).

## Acknowledgements

Quisque id velit erat. Quisque gravida posuere nisi, quis vestibulum urna
egestas in. Curabitur accumsan eget elit quis molestie. Nunc vestibulum, dolor
nec interdum iaculis, enim mauris auctor ligula, ac suscipit velit ex eu lectus.

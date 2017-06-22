#!/bin/bash

base=noise_decorrelation

pdflatex ${base}
bibtex ${base}
pdflatex ${base}
pdflatex ${base}

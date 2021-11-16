# shrinkage
version 1.0.0 <br>
author: Nicolas Honnorat <br>
date: November 16, 2021

# Overview

This project provides the most important methods of the original code associated with the publication: 

> Nicolas Honnorat, Mohamad Habes. "Covariance Shrinkage for Functional Connectomes." submitted to NeuroImage

Please cite the last version of the article if/when it will be published. 

# Usage

The usage of the python scripts can be printed by executed them with the -h option.

# Example

The following command line should create a chart corresponding to functional connectivity matrices (correlation matrices) of dimension 116, and display in red the matrices corresponding to the set of time series listed in the file "list.txt." The output chart will be shown in the file "OAS_chart.png" and should be similar to "OAS_chart_example.png"  

> python OASchart.py -d 116 -li list.txt



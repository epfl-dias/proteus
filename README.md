# SQLPlanner

Utility to obtain the query plan for a given SQL query.

The code invokes the Apache Calcite Optimizer to parse, validate, and optimize the input SQL query.
The code currently uses a series of heuristic rewrites, listed in QueryToPlan.scala.

Validation proceeds based on the db schema described in the 'resources/schema.json' file. 

The file lists a number of folders that contain single-line CSV files, each of which contains the schema of a table.

Extending the existing schema requires adding the new information in schema.json and adding a CSV files per new table.

## Use
* Type sbt run
* Type a SQL query based on the input schema corresponding to your CSV files

__Output:__ The plan produced by Calcite, and a serialization of it in JSON

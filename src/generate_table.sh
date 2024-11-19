#!/bin/sh

cat table1.html > ../table.html
julia table.jl >> ../table.html
cat table2.html >> ../table.html
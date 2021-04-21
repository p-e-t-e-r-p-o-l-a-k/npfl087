set term png 
set title 'Recall'
set ylabel 'Recall'
set xlabel 'treshold'
set output "recall.png"
set key outside
plot for [col=4:12:4] 'results' using 1:col with lines title columnheader

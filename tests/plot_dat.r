##
# plot_dat
#
# Plots data 
#
library(ggplot2)
tr <- read.csv( "../build/train.txt" )
ts <- read.csv( "../build/test.txt" )

ggplot() +
     geom_ribbon( data=ts, aes( x=x, ymin=f-2*s , ymax=f+2*s ), fill= "gray90" )+
     geom_point( data=tr, aes(x=x,y=y) , shape=21 , fill="#E69F00" , color = "#000000" ) +
     geom_line( data=ts, aes(x=x,y=y) , colour="gray90" , size = 0.5 ) +
     geom_line( data=ts, aes(x=x,y=f) , colour="#000000" , size = 0.5 ) +
     geom_line( data=ts, aes(x=x,y=f-2*s) , colour="#000000" , size =0.5) +
     geom_line( data=ts, aes(x=x,y=f+2*s) , colour="#000000" , size = 0.5) +
     labs( x = NULL , y = NULL ) + 
     theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggsave( "f-plot.pdf" )

ggplot() +
     geom_ribbon( data=ts, aes( x=x, ymin=g-2*v , ymax=g+2*v ), fill= "gray90" )+
     geom_point( data=tr, aes(x=x,y=z) , shape=21 , fill="#E69F00" , color = "#000000" ) +
     geom_line( data=ts, aes(x=x,y=z) , colour="gray90" , size = 0.5 ) +
     geom_line( data=ts, aes(x=x,y=g) , colour="#000000" , size = 0.5 ) +
     geom_line( data=ts, aes(x=x,y=g-2*v) , colour="#000000" , size =0.5) +
     geom_line( data=ts, aes(x=x,y=g+2*v) , colour="#000000" , size = 0.5) +
     labs( x = NULL , y = NULL ) + 
     theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

ggsave( "g-plot.pdf" )

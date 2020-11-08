neg_sample = c(20, 30, 40, 50, 60, 70, 80, 90, 100)
precision_2 = c(0.436, 0.414, 0.410, 0.397, 0.384, 0.375, 0.366, 0.358, 0.353)
ndcg = c(0.840, 0.814, 0.796, 0.778, 0.766, 0.752, 0.743, 0.735, 0.717)
map = c(0.893, 0.858, 0.834, 0.815, 0.797, 0.779, 0.768, 0.754, 0.733)
mrr_2 = c(0.603, 0.584, 0.574, 0.562, 0.552, 0.541, 0.532, 0.526, 0.514)


df <- data.frame(neg=neg_sample,
                 Precision=precision_2, NDCG=ndcg, MAP=map, MRR=mrr_2)
head(df)
library(reshape)
library(ggplot)
library("ggsci")
df_1 = melt(df, id="neg")
names(df_1)[2] <- "Metric"

ggplot(data=df_1, aes(x=neg, y=value, col=Metric))+
  geom_line(size=1.3)+
  geom_point()+
  labs(x="Num. NR Docs per Query", y="Ranking Metrics")+  
  ylim(0.33, 1) +
  theme_bw() +
  theme(legend.position = c(0.87, 0.83), legend.text=element_text(size = 20)) +
  theme(axis.title.x=element_text(face="italic", size=20)) +
  theme(axis.title.y=element_text(size=20))+
  theme(text = element_text(size=25))+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank())+
  scale_color_npg()





library(ggplot2)
# Basic line plot with points
 ggplot(data=df, aes(x=neg)) +
  geom_line(aes(y=p),color="darkblue")+
  geom_line(aes(y=n),color="darkred")+
  geom_line(aes(y=map),color="orange")+
  geom_line(aes(y=mrr),color="black")+
   scale_colour_manual("Loss Function",
                         values = c("darkblue", "darkred","orange","black")) +
  labs(x="Num. NR Docs per Query", y="Ranking Metrics") +theme(legend.position="top")+
   theme(legend.text=element_text(size = 8.5)) +
   theme(plot.title=element_text(size=10, face="bold")) +
   theme(axis.title.x=element_text(face="italic", size=12)) +
   theme(axis.title.y=element_text(size=12))+
 ggtitle("Comparsion of different Loss Function") 
   
   theme_bw()

  
  
 scale_linetype_manual(values=c("twodash", "dotted"))+
  scale_color_manual(values=c('#999999','#E69F00'))+
  scale_size_manual(values=c(1, 1.5))+
  theme(legend.position="top")



# Change the line type
p2 <- ggplot(data=df, aes(x=neg, y=n, group=1)) +
  geom_line(color="blue")+
  geom_point()+
  labs(x="Negative Sample", y="NDCG@5") 

# Change the color
p3 <- ggplot(data=df, aes(x=neg, y=map, group=1)) +
  geom_line(color="blue")+
  geom_point()+
  labs(x="Negative Sample", y="MAP") 

p4 <- ggplot(data=df, aes(x=neg, y=mrr, group=1)) +
  geom_line(color="blue")+
  geom_point()+
  labs(x="Negative Sample", y="MRR") 



grid.arrange(p1)


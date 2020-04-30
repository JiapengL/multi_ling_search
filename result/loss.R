sigmoid = function(x) {
  1 / (1 + exp(-x))}

laplace <- function(x){
  return(ifelse(x>=0 ,1-0.5*exp(-x), 0.5*exp(x)))}

theta0 = 0
theta1 = 0.5
pos <- function(x){
  -log(sigmoid(theta1 - x)) - log(1 - sigmoid(theta0 - x)) } 
pos_r <- function(x){pos(x) - pos(0.25)}

pol <- function(x){
  -log(laplace(theta1 - x)) - log(1 - laplace(theta0 - x)) }
pol_r <- function(x){pol(x) - pol(0.25)}

im <- function(x){
  result = ifelse(x < theta0, theta0 - x, 
            ifelse(x > theta1, x - theta1, 0))
  return(result)}

optimize(pol, interval = c(-1,1))

library(ggplot2)
library(gridExtra)


 ggplot(data.frame(x=c(-1, 1)), aes(x=x)) + 
  geom_path(aes(colour="darkblue"), stat="function", fun=pol_r) +
  geom_path(aes(colour="purple"), stat="function", fun=pos_r) +
  geom_path(aes(colour="red"), stat="function", fun=im) +
  scale_colour_identity("Loss Function", guide="legend", 
                        labels = c("PO Laplace", "PO Sigmoid","Immediate Threshold")) +
  labs(x="x", y="Loss") +
  ylim(0,1.1) +
   theme_bw() +
   theme(legend.position = c(0.75, 0.83), legend.text=element_text(size = 15)) +
   theme(axis.title.x=element_text(face="italic", size=15)) +
   theme(axis.title.y=element_text(size=15))+
   theme(text = element_text(size=15))



 ggplot(data=df_1, aes(x=neg, y=value, col=Metric))+
   geom_line()+
   labs(x="Num. NR Docs per Query", y="Ranking Metrics")+  
   ylim(0.33, 0.98) +
   theme_bw() +
   theme(legend.position = c(0.87, 0.83), legend.text=element_text(size = 15)) +
   theme(axis.title.x=element_text(face="italic", size=15)) +
   theme(axis.title.y=element_text(size=15))+
   theme(text = element_text(size=15))
 










pos <- function(x){
  -log(sigmoid(theta1 - x)) - log(1 - sigmoid(theta0 - x)) } 
pos_r <- function(x){pos(x) - pos(0.25)}

pol <- function(x){
  -log(laplace(theta1 - x)) - log(1 - laplace(theta0 - x)) }
pol_r <- function(x){pol(x) - pol(0.25)}

im <- function(x){
  result = ifelse(x < theta0, theta0 - x, 
                  ifelse(x > theta1, x - theta1, 0))
  return(result)}

optimize(pol, interval = c(-1,1))

library(ggplot2)
library(gridExtra)


ggplot(data.frame(x=c(-1, 1)), aes(x=x)) + 
  geom_path(aes(colour="darkblue"), stat="function", fun=pol_r) +
  geom_path(aes(colour="purple"), stat="function", fun=pos_r) +
  geom_path(aes(colour="red"), stat="function", fun=im) +
  scale_colour_identity("Loss Function", guide="legend", 
                        labels = c("PO Laplace", "PO Sigmoid","Immediate Threshold")) +
  labs(x="x", y="h(x)") +
  ylim(0,1.1) +
  ggtitle("Comparsion of different Loss Function") +
  theme(legend.text=element_text(size = 8.5)) +
  theme(plot.title=element_text(size=10, face="bold")) +
  theme(axis.title.x=element_text(face="italic", size=9.5)) +
  theme(axis.title.y=element_text(size=9.5))

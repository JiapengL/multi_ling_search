sigmoid = function(x) {
  1 / (1 + exp(-x))}

laplace <- function(x){
  return(ifelse(x>=0 ,1-0.5*exp(-x), 0.5*exp(x)))}

l2  <- function(x){
  return(ifelse(x>=0 ,1-0.5*exp(-x^2), 0.5*exp(-x^2)))}

reg <- function(x){
  return((x-(theta0+theta1)/2)^2)
}

theta0 = 0
theta1 = 0.5
pos <- function(x){
  -log(sigmoid(theta1 - x)) - log(1 - sigmoid(theta0 - x)) } 
pos_r <- function(x){pos(x) - pos(0.25)}

pol <- function(x){
  -log(laplace(theta1 - x)) - log(1 - laplace(theta0 - x)) }
pol_r <- function(x){pol(x) - pol(0.25)}

pol2 <- function(x){
  -log(l2(theta1 - x)) - log(1 - l2(theta0 - x)) }
pol2_r <- function(x){pol2(x) - pol2(0.25)}


im <- function(x){
  result = ifelse(x < theta0, theta0 - x, 
            ifelse(x > theta1, x - theta1, 0))
  return(result)}

im2 <- function(x){
  result = ifelse(x < theta0, (theta0 - x)^2, 
                  ifelse(x > theta1, (x - theta1)^2, 0))
  return(result)}

optimize(pol, interval = c(-1,1))

library(ggplot2)
library(gridExtra)


ggplot(data.frame(x=c(-1, 1)), aes(x=x)) + 
   geom_path(aes(colour="darkblue"), stat="function", fun=reg, size=1.3) +
#  geom_path(aes(colour="darkgreen"), stat="function", fun=pos_r, size=1.3) +
  geom_path(aes(colour="red"), stat="function", fun=im2, size=1.3) +
 # scale_colour_identity("Loss Function", guide="legend", 
#                        labels = c(expression(MSE), expression(PO), expression(SOSL))) +
   scale_colour_identity("Loss Function", guide="legend", 
                         labels = c(expression(MSE), expression(SOSL))) +
  labs(x="r", y="Loss") +
  ylim(0,1.1) +
   geom_vline(xintercept=c(0, 0.5),
              linetype="dashed")+
   theme_bw() +
   theme(legend.position = c(0.75, 0.75), legend.text=element_text(size = 20)) + 
   theme(panel.border = element_blank()) +
   theme(axis.title.x=element_text(face="italic", size=20)) +
   theme(axis.title.y=element_text(size=20))+
   theme(text = element_text(size=22))+
 theme(axis.line = element_line(colour = "black"),
       panel.grid.major = element_blank(),
       panel.grid.minor = element_blank(),
       panel.border = element_blank()) 


 
 
 
theta0 = 0
theta1 = 0.5 
reg1 <- function(x){
  return((x-(theta0-1)/2)^2)
}
reg2 <- function(x){
  return((x-(theta0+theta1)/2)^2)
}
reg3 <- function(x){
  return((x-(1+theta1)/2)^2)
}


im1 <- function(x){
  result = ifelse(x > theta0, (theta0 - x)^2, 0)
  return(result)}

im2 <- function(x){
  result = ifelse(x < theta0, (theta0 - x)^2, 
                  ifelse(x > theta1, (x - theta1)^2, 0))
  return(result)}
im3 <- function(x){
  result = ifelse(x < theta1, (theta1 - x)^2, 0)
  return(result)}

optimize(pol, interval = c(-1,1))

library(ggplot2)
library(gridExtra)


p1<- ggplot(data.frame(x=c(-1, 1)), aes(x=x)) + 
  geom_path(aes(colour="darkblue"), stat="function", fun=reg1, size=1.3) +
  #  geom_path(aes(colour="darkgreen"), stat="function", fun=pos_r, size=1.3) +
  geom_path(aes(colour="red"), stat="function", fun=im1, size=1.3) +
  # scale_colour_identity("Loss Function", guide="legend", 
  #                        labels = c(expression(MSE), expression(PO), expression(SOSL))) +
  scale_colour_identity("Loss Function", guide="legend", 
                        labels = c(expression(MSE), expression(SOSL))) +
  geom_text(x=-0.5, y=0.5, label="y=1", size=8)+
  labs(x="r", y="Loss") +
  ylim(0,1.1) +
  geom_vline(xintercept=c(0, 0.5),
             linetype="dashed")+
  theme_bw() +
  theme(legend.position = c(0.35, 0.75), legend.text=element_text(size = 18)) + 
  theme(panel.border = element_blank()) +
  theme(axis.title.x=element_text(face="italic", size=20)) +
  theme(axis.title.y=element_text(size=20))+
  theme(text = element_text(size=18))+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) 

p2<- ggplot(data.frame(x=c(-1, 1)), aes(x=x)) + 
  geom_path(aes(colour="darkblue"), stat="function", fun=reg2, size=1.3) +
  #  geom_path(aes(colour="darkgreen"), stat="function", fun=pos_r, size=1.3) +
  geom_path(aes(colour="red"), stat="function", fun=im2, size=1.3) +
  # scale_colour_identity("Loss Function", guide="legend", 
  #                        labels = c(expression(MSE), expression(PO), expression(SOSL))) +
  scale_colour_identity("Loss Function", guide="legend", 
                        labels = c(expression(MSE), expression(SOSL))) +
  geom_text(x=0.25, y=0.5, label="y=2", size=8)+
  labs(x="r", y="Loss") +
  ylim(0,1.1) +
  geom_vline(xintercept=c(0, 0.5),
             linetype="dashed")+
  theme_bw() +
  theme(legend.position = c(0.75, 0.75), legend.text=element_text(size = 18)) + 
  theme(panel.border = element_blank()) +
  theme(axis.title.x=element_text(face="italic", size=20)) +
  theme(axis.title.y=element_text(size=20))+
  theme(text = element_text(size=18))+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) 



p3<- ggplot(data.frame(x=c(-1, 1.1)), aes(x=x)) + 
  geom_path(aes(colour="darkblue"), stat="function", fun=reg3, size=1.3) +
  #  geom_path(aes(colour="darkgreen"), stat="function", fun=pos_r, size=1.3) +
  geom_path(aes(colour="red"), stat="function", fun=im3, size=1.3) +
  # scale_colour_identity("Loss Function", guide="legend", 
  #                        labels = c(expression(MSE), expression(PO), expression(SOSL))) +
  scale_colour_identity("Loss Function", guide="legend", 
                        labels = c(expression(MSE), expression(SOSL))) +
  geom_text(x=0.5, y=0.5, label="y=3", size=8)+
  labs(x="r", y="Loss") +
  ylim(0,1.1) +
  geom_vline(xintercept=c(0, 0.5),
             linetype="dashed")+
  theme_bw() +
  theme(legend.position = c(0.75, 0.75), legend.text=element_text(size = 18)) + 
  theme(panel.border = element_blank()) +
  theme(axis.title.x=element_text(face="italic", size=20)) +
  theme(axis.title.y=element_text(size=20))+
  theme(text = element_text(size=18))+
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()) 


grid.arrange(p1, p2, p3,nrow = 1)



## 3-c


s=0
arr=c()
for (i in seq(from = 0, to = 10000, by =2))
{
  s=s+dpois(x=i,lambda = 20)
  arr=c(arr,s)
}
s

plot(arr,type='l',xlab = 'iteration',ylab='incremental probability'
     ,main = expression(Sigma (frac(e^-lambda *lambda^-(2*n), 2*n*factorial))))


## 8


library(ggplot2)
df=read.csv('~/Documents/assignments/SI/HW2/Foods.csv')




## a

plot(df$pricePerServing)
outlier_ind=which(df$pricePerServing>25000)
outlier_ind
df2 <- df[-c(1429), ]


## b
ggplot(df2, aes(x = pricePerServing)) +
  geom_histogram( 
    binwidth = 5, color = "black", fill = "yellow")+
  theme_bw(base_size = 11) +
  geom_density(aes(y = 3*..count..), color = "red",lwd=1)+
  ggtitle(" probability distribution of pricePerServing")+theme(plot.title = element_text(hjust = 0.5))



ggplot(df,aes(x = healthScore,y=readyInMinutes))+stat_density_2d(aes(fill = ..level..), geom = "polygon")+
  ggtitle(" probability distribution of healthScore and readyInMinutes")+theme(plot.title = element_text(hjust = 0.5))



## c

types=unique(df$dishType)
counts=c()
for (i in types)
{
  counter=sum(df$dishType==i)
  counts=c(counts,counter)
}
df_bar=data.frame(dishType=types,
                  frequency=counts)
df_bar


p<-ggplot(data=df_bar, aes(x=dishType, y=frequency)) +
  geom_bar(stat="identity",fill=c("#eb8334", "#34ebe5", "#eb34cc",'#34eb56','#ebd334','#4334eb'))



p + coord_flip()+ggtitle("Bar plot of dish types")+theme(plot.title = element_text(hjust = 0.5))


## d

ggplot(data=df, aes(y= healthScore, x=dishType
) )+ geom_boxplot(fill=c("#eb8334", "#34ebe5", "#eb34cc",'#34eb56','#ebd334','#4334eb'))+
  ggtitle("Box plot of dish types and healthScore")+theme(plot.title = element_text(hjust = 0.5))

## e

library(ggmosaic)
ggplot(data=df)+geom_mosaic( aes(x=product(dairyFree,veryHealthy), fill=dairyFree,))
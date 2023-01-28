## 8
scores=c(57,66,72,78,79,79,81,81,82,83,84,87,88,88,89,90,91,92,94,95)
scores
print('mean')
mean(scores)
print('median')
median(scores)
print("variance")
var(scores)
print("standard div")
sd(scores)


getmode <- function(v){
  uni <- unique(v)
  
  uni[which.max(tabulate(match(v,uni)))]
}
print('mode')
getmode(scores)



lower_bound=quantile(scores,0.025)
lower_bound


outlier_ind=which(scores<lower_bound |scores>upper_bound)
scores[outlier_ind]



boxplot(scores)



h<-hist(scores, breaks=10, col="blue",
        main="Histogram with Normal Curve")
xfit<-seq(min(scores),max(scores),length=90)
yfit<-dnorm(xfit,mean=mean(scores),sd=sd(scores))

lines(xfit, 120*yfit, col="red", lwd=2) 



## 9


df=read.csv('~/Documents/assignments/SI/HW1/imdb.csv')


plot(table(df$year),xlab = 'years',ylab = 'number of movies', main='number of movies per year')




hist(df$USA_gross_income)




boxplot(x = df$duration,ylab = 'duration',main='movie duration')



uni=unique(df$tomatometer_status)


boxplot(df['duration'][df['tomatometer_status']=='Rotten']
        ,df['duration'][df['tomatometer_status']=='Certified-Fresh']
        ,df['duration'][df['tomatometer_status']=='Fresh'],ylab = 'duration',xlab ='Class',xaxt="n"
        ,main='side by side box plot of duration per rotten tomato class' )
xticks=c('Rotten','Certified-Fresh','Fresh')
axis(side=1, at=c(1,2,3), labels = FALSE)
text(x=c(1,2,3),  par("usr")[3], 
     labels = xticks, srt = 0, pos = 1, xpd = TRUE)



print('Rotten')
x=df['duration'][df['tomatometer_status']=='Rotten']
lower_bound=quantile(x,0.025)
upper_bound=quantile(x,0.975)
outlier_ind=which(x<lower_bound |x>upper_bound)
x[outlier_ind]

print('Certified-Fresh')
x=df['duration'][df['tomatometer_status']=='Certified-Fresh']
lower_bound=quantile(x,0.025)
upper_bound=quantile(x,0.975)
outlier_ind=which(x<lower_bound |x>upper_bound)
x[outlier_ind]

print('Fresh')
x=df['duration'][df['tomatometer_status']=='Fresh']
lower_bound=quantile(x,0.025)
upper_bound=quantile(x,0.975)
outlier_ind=which(x<lower_bound |x>upper_bound)
x[outlier_ind]





vals=c(sum(df$duration>200),sum(df$duration<=200 &df$duration>150),sum(df$duration<=150 &df$duration>100),sum(df$duration<=80))
labs =vals*100/sum(vals)
labels = c('very long','long','standard','short')
labs

p=pie(vals,labs,main='pie chart of movie duration',col=c('red','green','blue','yellow'))
legend('topleft',legend = c('very long','long','standard','short'),fill = c('red','green','blue','yellow'),col=c('red','green','blue','yellow'))



plot(df$USA_gross_income,df$worldwide_gross_income,main='worldwide_gross_income per USA_gross_income')

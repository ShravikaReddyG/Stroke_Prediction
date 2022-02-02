library(tidyverse)
library(naniar)
library(dbplyr)
library(caret)
library(rpart.plot)
library(Hmisc)
library(ggplot2)
library(glmnet)
library(Amelia)
library(boot)
library(pROC)
library(xgboost)
library(ranger)


##Download the dataset
healthcare_raw1 <- read.csv("healthcare-dataset-stroke-data.csv")
janitor::clean_names()

##rename the new set

brain_stroke_data <- healthcare_raw1


summary(brain_stroke_data)
#Clean the dataset - Removing the ID and changing N/A & Unknown to NA
brain_stroke_data[brain_stroke_data =="N/A"] <-NA 
brain_stroke_data[brain_stroke_data =="Unknown"] <-NA  
brain_stroke_data$id <- NULL


## Check what is the structure of the dataset
str(brain_stroke_data)

#Change the string variable to integer/number for an appropriate format
## Change variable worktype into categorical 0,1,2,3,4
brain_stroke_data$work_type <- factor(brain_stroke_data$work_type, levels= c("Never_worked","children","Govt_job","Private", "Self-employed"), labels=c(0:4))

## Change Variable ever_married to categorical (0,1)
brain_stroke_data$ever_married <- factor(brain_stroke_data$ever_married, levels =c("No","Yes"), labels = c(0,1))

## Change variable bmi to numeric value
brain_stroke_data$bmi <- as.numeric(brain_stroke_data$bmi)

## Change variable smoking_status to categorical(0,1,2)
brain_stroke_data$smoking_status <- factor(brain_stroke_data$smoking_status, levels=c("never smoked", "formerly smoked", "smokes"), labels=c(0:2))

##Change variable stroke into labels 'yes' or 'no'
brain_stroke_data$stroke<- factor(brain_stroke_data$stroke, levels=c("1", "0"), labels = c("yes","no"))

##Changing variable- gender to a categorical integer
brain_stroke_data$gender <- factor(brain_stroke_data$gender,levels =c("Male", "Female","Other"), labels=c(0:2))

##keeping the below variables as is 
brain_stroke_data$hypertension <- as.factor(brain_stroke_data$hypertension)

brain_stroke_data$heart_disease <- as.factor(brain_stroke_data$heart_disease)

#str(brain_stroke_data)
summary(brain_stroke_data$stroke)
vis_miss(brain_stroke_data,cluster = TRUE) 

###Imputing the values of data
brain_stroke_data$bmi[is.na(brain_stroke_data$bmi)] <- mean(brain_stroke_data$bmi,na.rm = TRUE)

brain_stroke_data$avg_glucose_level[is.na(brain_stroke_data$avg_glucose_level)] <- mean(brain_stroke_data$avg_glucose_level,na.rm = TRUE)

brain_stroke_data$age[is.na(brain_stroke_data$age)] <- mean(brain_stroke_data$age,na.rm = TRUE)

brain_stroke_data$smoking_status[is.na(brain_stroke_data$smoking_status)] <- 0



#missing data.Plot the data 
vis_miss(brain_stroke_data,cluster = TRUE) 

#brain_stroke_data <-na.omit(brain_stroke_data) #3426



###Taking all the cases which were stroke positive--
summary(brain_stroke_data)
stroke_tab =
  brain_stroke_data %>%
  filter(stroke == "yes")

###PIE CHARTS
#distribution of brain_stroke_data by hypertension

hyp_tab <- table(stroke_tab$hypertension) 
hyp_per <-round(100*hyp_tab/sum(hyp_tab))
hyp_lab <- paste(names(hyp_tab), 
                 "\n", hyp_per, "%", sep="")
pie(hyp_tab, labels=hyp_lab, col = c("light pink","black"),main="People who had a stroke by hypertension")

#distribution of brain_stroke_data by gender
gen_tab <- table(stroke_tab$gender) 

gen_per <-round(100*gen_tab/sum(gen_tab))

gen_lab <- paste(names(gen_tab), 
                 "\n", gen_per, "%", sep="")
pie(gen_tab, labels=gen_lab, col = c("light yellow", "light blue"),
    main="People who had a stroke by gender")


#distribution of brain_stroke_data by smoking status
smoking_tab <- table(stroke_tab$smoking_status) 
smoke_percent<-round(100*smoking_tab/sum(smoking_tab))
smoking_lab <- paste(names(smoking_tab), 
                     "\n", smoke_percent, "%", sep="")
pie(smoking_tab, labels=smoking_lab, col = c("light green", "light blue", "sky blue"),  main="People who had a stroke by Smoking status")



##ggplots

#distribution of brain_stroke_data by bmi
stroke_tab%>% 
  ggplot()+ 
  geom_point(mapping = aes(x = age,y = bmi),color="Maroon")+ 
  labs(x= "age", y="BMI", title = "Distribution of Stroke according to BMI")+
  ylim(10,60)+
  theme_linedraw()

#distribution of brain_stroke_data by age
stroke_tab %>%
  
  ggplot(aes(x=age)) +
  geom_histogram(bins=10,color="Purple") +
  labs(title = "People who had stroke from different age category")+
  viridis::scale_color_viridis(discrete = TRUE)

##converting stroke to factor so that we can use them in boxplots

##box plots ( ggplots)
brain_stroke_data$stroke <- factor(brain_stroke_data$stroke)


boxplot_1<-ggplot(brain_stroke_data, aes(x=gender,y=age,color=stroke))+geom_boxplot()

boxplot_1


boxplot_2<-ggplot(brain_stroke_data, aes(x=heart_disease,y=hypertension,color=stroke))+geom_boxplot()

boxplot_2


box_plot3 <- ggplot(brain_stroke_data, aes(x=ever_married, y = work_type,color = stroke))+geom_boxplot()

box_plot3

boxplot_4 <-ggplot(brain_stroke_data, aes(x=Residence_type, y = avg_glucose_level,color = stroke))+geom_boxplot()

boxplot_4

boxplot_5 <- ggplot(brain_stroke_data, aes(x=bmi, y = smoking_status, fill = stroke)) +geom_boxplot()

boxplot_5


###CLASSIFICATION VALUE

####DICISION TREE
##Dataset preparation
#attach(brain_stroke_data)
set.seed(1)
training.index <- sample(c(1:5110),5110*0.60)
training.df <- brain_stroke_data[training.index, ]
valid.df <- brain_stroke_data[-training.index, ]


tree_model=rpart(stroke ~., data = training.df,method = "class", control=rpart.control(cp = 0.001, maxdepth = 5))
fancyRpartPlot(tree_model, sub = "decision_tree", palettes = "YlGnBu" )
tree_predict= predict(tree_model, training.df,type="class")

confusionMatrix(tree_predict,as.factor(training.df$stroke))

#ROC
default.ct.predict.valid.roc <-predict(tree_model, valid.df,type = "prob")

r.dt <- roc(valid.df$stroke,default.ct.predict.valid.roc[,1])
plot.roc(r.dt)
auc(r.dt)


###LOGISTIC REGRESSION

###LOGISTIC REGRESSION
logit.reg.train1 <- glm(training.df$stroke ~ ., data = training.df ,family = "binomial")
options(scipen = 999)
summary(logit.reg.train1)
logit.ct.predict.train1 <-predict(logit.reg.train1, training.df,type = "response")
logit.ct.predict.train1
confusionMatrix(as.factor(ifelse(logit.ct.predict.train1 > 0.5,1,0)), as.factor(training.df$stroke))


logit.reg.valid <- glm(valid.df$stroke ~ ., data = valid.df ,family = "binomial")
logit.ct.predict.valid <-predict(logit.reg.valid, valid.df, type = "response")
logit.ct.predict.valid
confusionMatrix(as.factor(ifelse(logit.ct.predict.valid[,2] > 0.5,1,0)), as.factor(valid.df$stroke))


#### MODEL TRAINING Z#####

#Partition data for use in demonstration
set.seed(333)
train_ind<-createDataPartition(y=brain_stroke_data$stroke,p=0.60,list=FALSE)
training_data<-brain_stroke_data[train_ind,] 
testing_data<-brain_stroke_data[-train_ind,] 

##OVER SAMPLING
?upSample
trainup <-upSample(x=training_data[,-ncol(training_data)],
                   y=training_data$stroke) 
#4134

#y_train <- brain_stroke_data$brain_stroke_data[train_ind]
y_test <- brain_stroke_data$stroke[-train_ind]
y_test
##MODEL
set.seed(1)
ctrl <- trainControl(method = "cv",
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)
myGrid <- expand.grid(
  alpha = c(0,1),
  lambda = seq(0.00001, 1, length = 20)
)
glm.fit <- train(Class~.,
                 trainup,
                 method = "glmnet",
                 metric ="ROC",
                 tuneGrid = myGrid,
                 trControl = ctrl)
plot(glm.fit)
summary(glm.fit)
max(glm.fit[["results"]]$ROC)
varImp(glm.fit, scale=F)

summary(glm.fit)
### Apply to the Test Set
glm_prob <- predict(glm.fit, newdata = testing_data)
summary()
glm_prob_roc <- predict(glm.fit, newdata = testing_data, type="prob")[,2]
glm_final <- confusionMatrix(glm_prob, y_test,positive = "yes")
confusionMatrix(glm_prob, y_test,positive = "yes")

glm_roc <- roc(y_test,glm_prob_roc,plot = TRUE, print.auc = TRUE)





########### RANDOM FOREST #################

set.seed(222)
rf.grid <- data.frame(
  .mtry = 0:10,
  .splitrule = "gini",
  .min.node.size = 5)

rf.fit <- train(Class~.,
                trainup,
                method = "ranger",
                metric ="ROC",
                tuneGrid = rf.grid,
                trControl = ctrl)
plot(rf.fit)
rf2.final.per <- ranger(Class~.,
                        trainup,
                        mtry = rf.fit$bestTune[[1]], 
                        min.node.size = rf.fit$bestTune[[3]], 
                        splitrule = "gini",
                        importance = "permutation", 
                        scale.permutation.importance = TRUE)
barplot(sort(ranger::importance(rf2.final.per), decreasing = FALSE), las = 2, horiz = TRUE, cex.names = 0.7, col = colorRampPalette(colors = c("cyan","blue"))(10))

### Test Set
rf.pred <-predict(rf.fit, newdata = testing_data)
rf.pred.roc <- predict(rf.fit, newdata = testing_data, type="prob")[,1]
rf.prob.roc <- predict(rf.fit, newdata = testing_data, type="prob")[,2]
confusionMatrix(rf.pred, y_test,positive = "yes")
rf.roc <- roc(y_test,rf.prob.roc,plot = TRUE, print.auc = TRUE)


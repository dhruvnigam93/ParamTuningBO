library(mlr)
library(xgboost)

train = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" , stringsAsFactors = F , header = F)
test = read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" , stringsAsFactors = F , header = F)
setcol <- c("age","workclass","fnlwgt","education","education_num","marital_status","occupation",
            "relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","target")

names(train) = names(test) = setcol

for(i in 1:ncol(train)){ 
  if(is.character(train[,i]) ){
    train[,i] = factor(train[,i])
  }
}


traintask <- makeClassifTask (data = createDummyFeatures(obj = train ,  target = "target") ,target = "target")


lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree")), 
                        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))


rdesc <- makeResampleDesc("CV",stratify = T,iters=3)

ctrl <- makeTuneControlRandom(maxit = 10L)

mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

xgmodel <- train(learner = lrn_tune,task = traintask)

mat <- xgb.importance(feature_names = model$feature_names,model = model)


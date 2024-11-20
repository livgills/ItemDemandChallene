library(vroom)
library(timetk)
library(tidyverse)
library(patchwork)
library(tidymodels)
library(forecast)


ID_train <- vroom("./train.csv") 
ID_test <- vroom("./test.csv") 



store1_item3 <- ID_train %>% 
  filter(store ==1 & item ==3)

store5_item11 <- ID_train %>% 
  filter(store ==5 & item ==11)

store7_item13 <- ID_train %>% 
  filter(store ==7 & item ==13)

store10_item25 <- ID_train %>% 
  filter(store ==10 & item ==25)

plot1 <- store1_item3 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

plot2 <- store5_item11 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

plot3 <- store7_item13 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

plot4 <- store10_item25 %>%
  pull(sales) %>% 
  forecast::ggAcf(.,lag.max= 2*365)

(plot1 + plot2)/(plot3+plot4)


##############
storeItem <- ID_train %>% 
  filter(store == 3, item == 7)

my_recipe <- recipe(sales~., data = storeItem) %>% 
  step_date(date, features =c("dow","month","year"))



my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 600) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


rf_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,6)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(storeItem, v= 5, repeats = 1)


CV_results <- rf_workflow %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

bestTune <- CV_results %>% 
  select_best()

bestMetric <- collect_metrics(CV_results) %>% 
  filter(mtry == bestTune$mtry, min_n == bestTune$min_n) %>% 
  pull(mean)

bestMetric

final_wf <-
  RF_amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)


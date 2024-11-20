library(vroom)
library(timetk)
library(tidyverse)
library(patchwork)
library(tidymodels)
library(forecast)
library(modeltime)


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

#####################
#Arima models 

arima_recipe <- recipe(sales~., data = train) %>% 
  step_date(date, features =c("dow","month","year"))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5, 
                         seasonal_ar = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>% 
  set_engine("auto_arima")

train <- ID_train %>% filter(store==3, item==17)
cv_split <- time_series_split(train, assess="3 months", cumulative = TRUE)
cv_split %>%
tk_time_series_cv_plan() %>% 
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
test <- ID_test %>% filter(store == 3, item == 17)
cv_split <- time_series_split(test, assess= "3 months", cumulative = TRUE)

arima_wf <- workflow() %>% 
  add_recipe(arima_recipe) %>% 
  add_model(arima_model) %>% 
  fit(data=training(cv_split))


cv_results <- modeltime_calibrate(arima_wf, 
                                  new_data = testing(cv_split))
p1<- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = FALSE)


arima_fullfit <- cv_results %>% 
  modeltime_refit(data = train)

arima_preds <- arima_fullfit %>% 
  modeltime_forecast(new_data=train) %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p2<- arima_fullfit %>% 
  modeltime_forecast(new_data= train, actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE)


arima_wf <- workflow() %>% 
  add_recipe(arima_recipe) %>% 
  add_model(arima_model) %>% 
  fit(data=training(cv_split))


cv_results <- modeltime_calibrate(arima_wf, 
                                  new_data = testing(cv_split))
p3 <- cv_results %>% 
  modeltime_forecast(new_data = testing(cv_split),
                     actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

cv_results %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = FALSE)

arima_fullfit <- cv_results %>% 
  modeltime_refit(data = train)

arima_preds <- arima_fullfit %>% 
  modeltime_forecast(new_data=train) %>% 
  rename(date= .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y= ID_test, by= "date") %>% 
  select(id, sales)

p4 <- arima_fullfit %>% 
  modeltime_forecast(new_data= train, actual_data = train) %>% 
  plot_modeltime_forecast(.interactive = FALSE, .legend_show = FALSE)

plotly::subplot(p1,p3, p2,p4, nrows = 2)  

library(dplyr)
library(tidyr)
library(flextable)

source('classification/classification.utils.R')

dir.create('tables', showWarnings = F, recursive = T)

results.dir = 'cv_results'
classifiers = c('glmnet', 'randomForest', 'svmRadialWeights', 'xgbTree')


metrics_tibble = function(params, criteria) {

  dataset = params$dataset
  include.vars = params$include.vars[[1]]
  include.freesurfer = params$include.freesurfer
  freesurfer.measures = params$freesurfer.measures[[1]]
  average.lr = params$average.lr
  hamd = params$hamd
  classifier = params$classifier
  include.vbm = params$include.vbm
  
  file.suffix = get_suffix(
    include.vars = include.vars,
    include.freesurfer = include.freesurfer,
    average.lr = average.lr,
    freesurfer.measures = freesurfer.measures,
    include.vbm = params$include.vbm
  )
  
  out.rds = file.path('data', dataset, results.dir, 
    paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))        
  print(out.rds)
  
  df = readRDS(file = out.rds)
  metrics = bind_rows(df$cv_results %>% lapply(evaluate_metrics))
  
  metrics$balanced_accuracy = metrics$balanced_accuracy * 100
  metrics$sensitivity = metrics$sensitivity * 100
  metrics$specificity = metrics$specificity * 100
  
  if (classifier == 'glmnet') classifier.str = 'Elastic Net'
  if (classifier == 'svmRadialWeights') classifier.str = 'SVM'
  if (classifier == 'randomForest') classifier.str = 'Random Forest'
  if (classifier == 'xgbTree') classifier.str = 'Boosted Trees'
  
  # Output metrics as table
  tibble(
    Classifier = classifier.str,
    AUC = sprintf('%0.2f (%0.2f)', mean(metrics$auc), sd(metrics$auc)),
    `Balanced Accuracy` = sprintf('%0.1f (%0.1f)', mean(metrics$balanced_accuracy), sd(metrics$balanced_accuracy)),
    Sensitivity = sprintf('%0.1f (%0.1f)', mean(metrics$sensitivity), sd(metrics$sensitivity)),
    Specificity = sprintf('%0.1f (%0.1f)', mean(metrics$specificity), sd(metrics$specificity))
  )
  
}


study.params =
  tibble(include.freesurfer = F) %>%
  full_join(tibble(dataset = c('NP1')), by = character()) %>%
  mutate(include.vbm = T) %>%
  mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
  mutate(hamd = 'hamd6') %>%
  full_join(tibble(classifier = classifiers), by = character()) %>%
  group_split(row_number(), .keep = FALSE)

df.tbl = left_join(
  lapply(study.params, metrics_tibble, 'response') %>% bind_rows(),
  lapply(study.params, metrics_tibble, 'remission') %>% bind_rows(),
  by = 'Classifier'
)

df.tbl %>% flextable() %>%
set_header_labels(
  AUC.x = 'AUC', AUC.y = 'AUC',
  `Balanced Accuracy.x` = 'Balanced Accuracy', `Balanced Accuracy.y` = 'Balanced Accuracy',
  Sensitivity.x = 'Sensitivity', Sensitivity.y = 'Sensitivity',
  Specificity.x = 'Specificity', Specificity.y = 'Specificity'
) %>%
add_header_row(values = c('', 'Response', 'Remission'), colwidths = c(1, 4, 4)) %>%
flextable::save_as_docx(path = file.path('tables', 'Table_S9.docx'))

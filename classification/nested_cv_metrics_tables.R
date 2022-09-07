library(dplyr)
library(tidyr)
library(flextable)

source('classification/classification.utils.R')

classifiers = c('glmnet', 'randomForest', 'svmRadialWeights', 'xgbTree')

results.dir = 'cv_results'
dir.create('tables', showWarnings = F, recursive = T)

metrics_tibble = function(params, criteria, include.pval=T) {

  dataset = params$dataset
  include.vars = params$include.vars[[1]]
  include.freesurfer = params$include.freesurfer
  freesurfer.measures = params$freesurfer.measures[[1]]
  average.lr = params$average.lr
  hamd = params$hamd
  classifier = params$classifier
  
  file.suffix = get_suffix(
    include.vars = include.vars,
    include.freesurfer = include.freesurfer,
    average.lr = average.lr,
    freesurfer.measures = freesurfer.measures
  )
  
  out.rds = file.path('data', dataset, results.dir, 
    paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))        
  print(out.rds)
  
  df = readRDS(file = out.rds)
  metrics = bind_rows(df$cv_results %>% lapply(evaluate_metrics))
  
  # Print out p-values if permutations are evaluated
  
  perm.dir = file.path('data', dataset, results.dir, 
    paste('permutations', classifier, criteria, hamd, file.suffix, sep = '.'))
  
  if (dir.exists(perm.dir)) {
    
    get_mean_aucs = function(n_perm) {
      out.rds = file.path(perm.dir, sprintf('permutation-%i.rds', n_perm))
      # print(out.rds)
      df = readRDS(file = out.rds)
      metrics = bind_rows(df$cv_results %>% lapply(evaluate_metrics))
      mean(metrics$auc)
    }
    
    if (classifier %in% c('glmnet', 'randomForest')) {
      n_permutations = 1000
    }
    
    if (classifier %in% c('svmRadialWeights', 'xgbTree')) {
      n_permutations = 100
    }
    
    perm.aucs = unlist(lapply(1:n_permutations, get_mean_aucs))
    print(sprintf('Mean perm auc = %f', mean(perm.aucs)))
    
    p.val = sum(perm.aucs > mean(metrics$auc))/length(perm.aucs)
    
    print(sprintf('%s - %s - %s', dataset, classifier, criteria))
    print(sprintf('P-value: %f', p.val))
    
    if (p.val < 0.001) {
      p.val.str = '0.001'
    } else {
      p.val.str = sprintf('%0.3f', p.val)
    }
    
  } else {
    p.val.str = 'NA'
  }
  
  # Format as percentage
  metrics$balanced_accuracy = metrics$balanced_accuracy * 100
  metrics$sensitivity = metrics$sensitivity * 100
  metrics$specificity = metrics$specificity * 100
  
  if (classifier == 'glmnet') classifier.str = 'Elastic Net'
  if (classifier == 'svmRadialWeights') classifier.str = 'SVM'
  if (classifier == 'randomForest') classifier.str = 'Random Forest'
  if (classifier == 'xgbTree') classifier.str = 'Boosted Trees'
  
  # Output metrics as table
  df.tbl = tibble(
    Classifier = classifier.str,
    AUC = sprintf('%0.2f (%0.2f)', mean(metrics$auc), sd(metrics$auc)),
    `Balanced Accuracy` = sprintf('%0.1f (%0.1f)', mean(metrics$balanced_accuracy), sd(metrics$balanced_accuracy)),
    Sensitivity = sprintf('%0.1f (%0.1f)', mean(metrics$sensitivity), sd(metrics$sensitivity)),
    Specificity = sprintf('%0.1f (%0.1f)', mean(metrics$specificity), sd(metrics$specificity))
  )
  
  if (include.pval) {
    df.tbl = df.tbl %>% bind_cols(tibble(`AUC p-value` = p.val.str))
  }
  
  df.tbl
  
}


# NeuroPharm

study.params =
  tibble(include.freesurfer = T) %>%
  full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
  full_join(tibble(average.lr = F), by = character()) %>%
  full_join(tibble(dataset = c('NP1')), by = character()) %>%
  mutate(hamd = 'hamd6') %>%
  mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
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
  Specificity.x = 'Specificity', Specificity.y = 'Specificity',
  `AUC p-value.x` = 'AUC p-value', `AUC p-value.y` = 'AUC p-value'
) %>%
add_header_row(values = c('', 'Response', 'Remission'), colwidths = c(1, 5, 5)) %>%
flextable::save_as_docx(path = file.path('tables', 'Table_3.docx'))


# EMBARC

study.params =
  tibble(include.freesurfer = T) %>%
  full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
  full_join(tibble(average.lr = F), by = character()) %>%
  full_join(tibble(dataset = c('EMBARC')), by = character()) %>%
  mutate(hamd = 'hamd6') %>%
  mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
  full_join(tibble(classifier = classifiers), by = character()) %>%
  group_split(row_number(), .keep = FALSE)

df.tbl = left_join(
  lapply(study.params, metrics_tibble, 'response', include.pval=F) %>% bind_rows(),
  lapply(study.params, metrics_tibble, 'remission', include.pval=F) %>% bind_rows(),
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
flextable::save_as_docx(path = file.path('tables', 'Table_S8.docx'))

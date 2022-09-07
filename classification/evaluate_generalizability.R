library(dplyr)
library(tidyr)
library(flextable)
library(sva)

source('classification/classification.utils.R')

dir.create('tables', showWarnings = F, recursive = T)

# Evaluation parameters
evaluation.dataset = 'EMBARC'
results.dir = 'final_models'
compute.permutations = T


get_model_metrics = function(fit, X, y, permute = F, seed = 42) {
  
  # Permute if required, used to estimate null
  if (permute) {
    set.seed(seed)
    y = sample(y, replace = F)
  }
  
  # Override preProcess
  preproc = fit$preProcess
  fit$preProcess = NULL
  
  X_train = fit$trainingData %>%
    select_if(!names(.) %in% c('.weights', '.outcome'))
  X_test = X
  
  # Remove any left over factors
  if (!is.matrix(X_test)) {
    X_train = X_train %>% select_if(function(x) !is.factor(x))
    X_factors = X_test %>% select_if(is.factor)
    X_test = X_test %>% select_if(function(x) !is.factor(x))
  }
  
  # Apply preprocessing
  if (!is.null(preproc)) {
    X_train = predict(preproc, X_train)
    X_test = predict(preproc, X_test)
  }
  
  # Remove batch effect using ComBat
  edata = t(rbind(X_test, X_train))
  batch = c(rep(1, nrow(X_test)), rep(2, nrow(X_train)))
  combat_edata = invisible(  # suppress messages
    ComBat(dat = edata, batch = batch, par.prior = TRUE, ref.batch = 2)
  )
  X_test_combat = t(combat_edata[,1:nrow(X_test)])
  
  # Add back factors, if any
  if (exists('X_factors')) {
    X_test_combat = X_test_combat %>% cbind(X_factors)
  }
  
  # Using predict() directly results in the following error
  # predict(fit, newdata = X)
  # no applicable method for 'predict' applied to an object of class "character"
  
  # extractPrediction results in NA values
  # extractPrediction(list(fit), unkX = X)
  # obs pred  model dataType  object
  # 1  <NA> <NA> glmnet  Unknown Object1
  # 2  <NA> <NA> glmnet  Unknown Object1
  
  prob = extractProb(list(fit), unkX = X_test_combat, unkOnly = T)[,'case']
  pred = caret::predict.train(fit, newdata = X_test_combat)
  
  # Save out results
  metrics = tibble(
    pred = list(pred),
    prob = list(prob),
    y = list(y)
  ) %>%
    evaluate_metrics
  
}


metrics_tibble = function(params, criteria) {
  
  training.dataset = params$dataset
  include.vars = params$include.vars[[1]]
  include.freesurfer = params$include.freesurfer
  freesurfer.measures = params$freesurfer.measures[[1]]
  average.lr = params$average.lr
  hamd = params$hamd
  classifier = params$classifier
  
  # Load and format data
  data = load_data(
    dataset = evaluation.dataset,
    criteria = criteria,
    hamd=hamd,
    include.vars = include.vars,
    include.freesurfer = include.freesurfer,
    freesurfer.measures = freesurfer.measures,
    average.lr = average.lr
  )
  
  y = data$group
  
  # Transform factors to dummy variables for classifiers to require it
  if (classifier != c('randomForest')) {
    X = model.matrix(group ~ ., data = data)
    X = X[,-1]  # remove intercept
  } else {
    X = data %>% select(-group)
  }
  
  file.suffix = get_suffix(
    include.vars = include.vars,
    include.freesurfer = include.freesurfer,
    average.lr = average.lr,
    freesurfer.measures = freesurfer.measures
  )
  
  # Load model trained on training dataset
  model.rds = file.path('data', training.dataset, results.dir, 
                        paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))
  print(model.rds)
  mdls = readRDS(file = model.rds)
  
  metrics = bind_rows(mcmapply(
    get_model_metrics,                  
    mdls, list(X), list(y),
    mc.cores = n_jobs,
    SIMPLIFY = F
  ))
  
  if (compute.permutations) {
    
    # Estimate metrics    
    perm.metrics = bind_rows(mcmapply(
      get_model_metrics,                  
      mdls, list(X), list(y),
      permute = T,
      seed = 1:length(mdls),
      mc.cores = n_jobs,
      SIMPLIFY = F
    ))
    
    p.val = sum(perm.metrics$auc > mean(metrics$auc))/length(perm.metrics$auc)
    
    print(sprintf('P-value: %f', p.val))
    
  } else {
    p.val = NA
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
  tibble(
    Classifier = classifier.str,
    AUC = sprintf('%0.2f (%0.2f)', mean(metrics$auc), sd(metrics$auc)),
    `Balanced Accuracy` = sprintf('%0.1f (%0.1f)', mean(metrics$balanced_accuracy), sd(metrics$balanced_accuracy)),
    Sensitivity = sprintf('%0.1f (%0.1f)', mean(metrics$sensitivity), sd(metrics$sensitivity)),
    Specificity = sprintf('%0.1f (%0.1f)', mean(metrics$specificity), sd(metrics$specificity)),
    `AUC p-value` = sprintf('%0.2f', p.val)
  )
  
}

classifiers = c('randomForest', 'glmnet')

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
flextable::save_as_docx(path = file.path('tables', 'Table_generalization.docx'))

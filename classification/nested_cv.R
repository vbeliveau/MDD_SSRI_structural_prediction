# Perform nested cross-validation

source('classification/classification.utils.R')

# Plotting options
do.plot.roc = T

# Evaluation parameters
n_repetitions = 25
n_folds = 5
compute.permutations = T
use.class.weight = T

# Define datasets to process

classifiers = c('glmnet', 'randomForest', 'xgbTree', 'svmRadialWeights')

study.params =
  tibble(include.freesurfer = T) %>%
  full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
  full_join(tibble(average.lr = F), by = character()) %>%
  full_join(tibble(dataset = c('NP1', 'EMBARC')), by = character()) %>%
  full_join(tibble(criteria = c('response', 'remission')), by = character()) %>%
  mutate(hamd = 'hamd6') %>%
  mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
  full_join(tibble(classifier = classifiers), by = character()) %>%
  group_split(row_number(), .keep = FALSE)

train_nested_cv(
  study.params = study.params,
  classifiers = classifiers,
  compute.permutations = F,
  do.plot.roc = do.plot.roc,
  n_repetitions = n_repetitions,
  n_folds = n_folds,
  use.class.weight = use.class.weight,
  n_jobs = n_jobs
)

if (compute.permutations) {
  
  classifiers = c('glmnet', 'randomForest')
  n_permutations = 1000
  plot.permutations = T
  
  study.params =
    tibble(include.freesurfer = T) %>%
    full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
    full_join(tibble(average.lr = F), by = character()) %>%
    full_join(tibble(dataset = c('NP1')), by = character()) %>%
    full_join(tibble(criteria = c('response', 'remission')), by = character()) %>%
    mutate(hamd = 'hamd6') %>%
    mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
    full_join(tibble(classifier = classifiers), by = character()) %>%
    group_split(row_number(), .keep = FALSE)
  
  train_nested_cv(
    study.params = study.params,
    classifiers = classifiers,
    compute.permutations = T,
    n_permutations = n_permutations,
    do.plot.roc = do.plot.roc,
    plot.permutations = plot.permutations,
    n_repetitions = n_repetitions,
    n_folds = n_folds,
    use.class.weight = use.class.weight,
    n_jobs = n_jobs
  )
  
  classifiers = c('xgbTree', 'svmRadialWeights')
  n_permutations = 100

  study.params =
    tibble(include.freesurfer = T) %>%
    full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
    full_join(tibble(average.lr = F), by = character()) %>%
    full_join(tibble(dataset = c('NP1')), by = character()) %>%
    full_join(tibble(criteria = c('response', 'remission')), by = character()) %>%
    mutate(hamd = 'hamd6') %>%
    mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
    full_join(tibble(classifier = classifiers), by = character()) %>%
    group_split(row_number(), .keep = FALSE)

  train_nested_cv(
    study.params = study.params,
    classifiers = classifiers,
    compute.permutations = compute.permutations,
    n_permutations = n_permutations,
    do.plot.roc = do.plot.roc,
    plot.permutations = plot.permutations,
    n_repetitions = n_repetitions,
    n_folds = n_folds,
    use.class.weight = use.class.weight,
    n_jobs = n_jobs
  )

}

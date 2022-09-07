source('classification/classification.utils.R')

# Plotting options
do.plot.roc = T

# Evaluation parameters
fs_version = 'fastsurfer'
n_repetitions = 25
n_folds = 5
compute.permutations = F
use.class.weight = T
cv.checkpoint = T

classifiers = c('glmnet', 'randomForest', 'xgbTree', 'svmRadialWeights')

# Define datasets to process
study.params =
  tibble(include.freesurfer = F) %>%
  full_join(tibble(dataset = c('NP1')), by = character()) %>%
  full_join(tibble(criteria = c('response', 'remission')), by = character()) %>%
  mutate(include.vbm = T) %>%
  mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
  mutate(hamd = 'hamd6') %>%
  full_join(tibble(classifier = classifiers), by = character()) %>%
  group_split(row_number(), .keep = FALSE)

train_nested_cv(
  study.params = study.params,
  classifiers = classifiers,
  compute.permutations = compute.permutations,
  do.plot.roc = do.plot.roc,
  n_repetitions = n_repetitions,
  n_folds = n_folds,
  use.class.weight = use.class.weight,
  n_jobs = n_jobs,
  cv.checkpoint = cv.checkpoint
)

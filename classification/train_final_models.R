source('classification/classification.utils.R')

# Train final models using all available data

# Evaluation parameters
n_bootstrap = 1000
use.class.weight = T

study.params =
  tibble(include.freesurfer = T) %>%
  full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
  full_join(tibble(average.lr = F), by = character()) %>%
  full_join(tibble(dataset = c('NP1')), by = character()) %>%
  full_join(tibble(criteria = c('response', 'remission')), by = character()) %>%
  mutate(hamd = 'hamd6') %>%
  mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
  full_join(tibble(classifier = c('randomForest', 'glmnet')), by = character()) %>%
  group_split(row_number(), .keep = FALSE)

train_final_models(
  study.params = study.params,
  classifiers = classifiers,
  n_bootstrap = n_bootstrap,
  use.class.weight = use.class.weight,
  n_jobs = n_jobs
)

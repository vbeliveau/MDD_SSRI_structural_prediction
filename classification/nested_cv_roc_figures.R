source('classification/classification.utils.R')

dir.create('figures', showWarnings = F, recursive = T)

results.dir = 'cv_results'
classifiers = c('glmnet', 'randomForest', 'svmRadialWeights', 'xgbTree')

# Aggregate data

extract_classifier_roc = function(params) {
  
  dataset = params$dataset
  include.vars = params$include.vars[[1]]
  include.freesurfer = params$include.freesurfer
  freesurfer.measures = params$freesurfer.measures[[1]]
  average.lr = params$average.lr
  criteria = params$criteria
  hamd = params$hamd
  classifier = params$classifier

  file.suffix = get_suffix(
    include.vars = include.vars,
    include.freesurfer = include.freesurfer,
    average.lr = average.lr,
    freesurfer.measures = freesurfer.measures
  )
  
  cv.rds = file.path('data', dataset, results.dir, 
     paste(classifier, criteria, hamd, file.suffix, 'rds', sep = '.'))   
  
  # Read data
  df = readRDS(file = cv.rds)
  
  # Create ROC
  df.roc = bind_rows(df$cv_results %>% lapply(create_roc))
  df.roc$group = classifier
  
  df.roc
  
}


roc_plot_wrapper  = function(criteria, dataset) {
  
  study.params =
    tibble(include.freesurfer = T) %>%
    full_join(tibble(freesurfer.measures = list(c('thickness', 'volume'))), by = character()) %>%
    full_join(tibble(average.lr = F), by = character()) %>%
    full_join(tibble(dataset = dataset), by = character()) %>%
    full_join(tibble(criteria = c(criteria)), by = character()) %>%
    mutate(hamd = 'hamd6') %>%
    mutate(include.vars = list(c('age', 'sex', 'hamd6.base', 'single.recurrent'))) %>%
    full_join(tibble(classifier = classifiers), by = character()) %>%
    group_split(row_number(), .keep = FALSE)

  df.roc =
    lapply(study.params, extract_classifier_roc) %>%
    bind_rows() %>%
    mutate(group =
      recode(group, 
        glmnet = 'Elastic Net',
        svmRadialWeights = 'SVM',
        randomForest = 'Random Forest',
        xgbTree = 'Boosted Trees'
      )
    )
  
  # Plot
  plot_cv_roc(
    df.roc,
    title = str_to_title(criteria),
    group = 'group',
    plot.sd = F,
    plot.color = T
  )
  
}

# NeuroPharm

plot.list = lapply(c('response', 'remission'), roc_plot_wrapper, 'NP1')

ggarrange(
  plotlist = plot.list,
  ncol = 2,
  common.legend = T,
  labels='AUTO',
  legend = 'bottom'
)

ggsave(
  file = file.path('figures', 'Fig2.pdf'),
  width = 180, height = 100,
  dpi = 600,
  units = 'mm',
  bg = 'white'
)

ggsave(
  file = file.path('figures', 'Fig2.png'),
  width=180, height=100,
  dpi=600,
  units = 'mm',
  bg = 'white'
)


# EMBARC

plot.list = lapply(c('response', 'remission'), roc_plot_wrapper, 'EMBARC')

ggarrange(
  plotlist = plot.list,
  ncol = 2,
  common.legend = T,
  labels='AUTO',
  legend = 'bottom'
)

ggsave(
  file = file.path('figures', 'Fig_S1.pdf'),
  width = 180, height = 100,
  dpi = 600,
  units = 'mm',
  bg = 'white'
)

ggsave(
  file = file.path('figures', 'Fig_S1.png'),
  width=180, height=100,
  dpi=600,
  units = 'mm',
  bg = 'white'
)


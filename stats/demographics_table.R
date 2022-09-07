library(dplyr)
library(tidyr)
library(gtsummary)

source('classification/classification.utils.R')

dir.create('tables', showWarnings = F, recursive = T)

datasets = c('NP1', 'EMBARC')

tbls = list()

for (dataset in datasets) {
  
  # Load data
  df = load_data(
    dataset,
    criteria = 'response',  # used to assign `group`, will not be used
    hamd = 'hamd6',
    include.freesurfer = F,
    include.vars = c(
      'responder.hamd6',
      'remitter.hamd6',
      'age',
      'sex',
      'hamd6.base',
      'hamd6.week8',
      'single.recurrent'
    )
  )
    
  # Create contingency table
  print(table(df$responder.hamd6, df$remitter.hamd6))
  
  # HAMD-6
  
  # Responder
  
  df$single.recurrent = df$single.recurrent %>% recode(Single = 'First-episode')
  
  tab.responder = df %>%
    tbl_summary(label = list(
      age ~ 'Age',
      sex ~ 'Sex',
      single.recurrent ~ 'Recurrence status',
      hamd6.base ~ 'HAMD-6 - Baseline',
      hamd6.week8 ~ 'HAMD-6 - Week 8'),
      type = hamd6.base ~ 'continuous',
      by = responder.hamd6) %>%
    add_p() 
  
  # Remitter

  tab.remitter = df %>%
    tbl_summary(label = list(
      age ~ 'Age',
      sex ~ 'Sex',
      single.recurrent ~ 'Recurrence status',
      hamd6.base ~ 'HAMD-6 - Baseline',
      hamd6.week8 ~ 'HAMD-6 - Week 8'),
      type = hamd6.base ~ 'continuous',
      by = remitter.hamd6) %>%
    add_p()
  
    tbl_merge(
      tbls=list(tab.responder, tab.remitter),
      tab_spanner = c("**Response**", "**Remission**")
    ) %>%
    as_flex_table() %>%
    flextable::save_as_docx(path = file.path('tables', paste0('demographics.', dataset, '.docx')))
    
}


# Identify the number of individuals who switched to duloxetine before end of week 8
df = read_excel(file.path('lists', 'MR_NP1_HC_DBPROJECT_Vincent.xlsx'))
sum(df$`Week for switch from escitalopram to duloxetine` < 8, na.rm = T)

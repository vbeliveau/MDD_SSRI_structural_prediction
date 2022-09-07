library(dplyr)
library(tidyr)
library(gtsummary)

source('classification/classification.utils.R')

tbls = list()

# Load data
NP1 = load_data(
  'NP1',
  criteria = 'response',  # needed to assign `group`, will not be used
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

# Load data
EMBARC = load_data(
  'EMBARC',
  criteria = 'response',  # needed to assign `group`, will not be used
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

# Perform tests

# Age
wilcox.test(NP1$age, EMBARC$age)

# Sex
df = tibble(sex = NP1$sex, dataset = 'NP1') %>% add_row(
    tibble(sex = EMBARC$sex, dataset = 'EMBARC')
  ) %>%
  group_by(dataset)

chisq.test(df$dataset, df$sex)

# Recurrence status
df = tibble(single.recurrent = NP1$single.recurrent, dataset = 'NP1') %>% add_row(
  tibble(single.recurrent = EMBARC$single.recurrent, dataset = 'EMBARC')
) %>%
group_by(dataset)

chisq.test(df$dataset, df$single.recurrent)

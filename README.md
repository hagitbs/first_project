
# we split the training dataset into 3 equal parts, randomly. We train on 1/3 of the train data and then end up labeling and training on an additional 10 percent of the previously unlabelled data randomly.
# %% test1 baseline Performance (40)
TEST_ID = 1  
train_percent=0.4 

# %% test2 Minimum Performance (30)
TEST_ID=2 
train_percent=0.3

# %% test3 Maximum Performance (50)
TEST_ID=3
train_percent=0.5

# %% test4 JSD Performance (30+10)
TEST_ID=4
train_percent=0.3
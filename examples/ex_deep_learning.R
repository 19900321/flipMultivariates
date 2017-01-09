data(iris)
m.iris <- DeepLearning(Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, data=iris)
print(m.iris)

# Performance of Glass data set here: http://www.is.umk.pl/projects/datasets.html#Other
# Around 70% accuracy by other methods - similar to our default parameters
# Just increasing the training time will increase accuracy to around 80%
# Increasing network size and training time we got accuracy of 94.86%
# But running time was around 100 seconds - close to 2 minute limit
data(Glass, package="mlbench")
system.time(m.glass <- DeepLearning(Type~RI+Na+Mg+Al+Si+K+Ca+Ba+Fe, data=Glass, hidden=c(50,50,50), epochs=5000))
print(m.glass)

# This dataset is available from CRAN
# Slightly larger than the mlbench datasets - 4177 instances
# R-sq = 0.57, running time 94 seconds
# Increasing training time and network size does not help
data(abalone, package="AppliedPredictiveModeling")
system.time(m.abalone <- DeepLearning(Rings~Type+LongestShell+Diameter+Height+WholeWeight+ShuckedWeight+VisceraWeight+ShellWeight, data=abalone, hidden=c(200,200,200), epochs=5000))
print(m.abalone)

# Very simple - total accuracy
data(Zoo)
system.time(m.zoo <- DeepLearning(type~hair+feathers+eggs+milk+airborne+aquatic+predator+toothed+backbone+breathes+venomous+fins+legs+tail+domestic, data=Zoo))

# Default parameter give R-sq of ~60%
# High accuracy model took 220 seconds to train
data("concrete", package="AppliedPredictiveModeling")
system.time(m.concrete <- DeepLearning(CompressiveStrength~Cement+BlastFurnaceSlag+FlyAsh+Water+Superplasticizer+CoarseAggregate+FineAggregate+Age, data=concrete, hidden=c(100,100,100), epochs=2000))

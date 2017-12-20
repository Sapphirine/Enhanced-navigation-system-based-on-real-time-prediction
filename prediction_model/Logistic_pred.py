from os.path import expanduser, join, abspath
from pyspark.mllib.classification import SVMWithSGD,SVMModel,LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from pyspark import SparkConf
conf = SparkConf().setAppName("JimYang").setMaster("local").set("spark.driver.memory", "2G").set("spark.executor.memory", "1G")
sc = SparkContext(conf=conf)

def parsePoint(line):
  values = [float(x) for x in line.split(",")]
  return LabeledPoint(values[0],values[1:])

data_train = sc.textFile("./traffic_grad_3.csv")

parseData_train = data_train.map(parsePoint)

training, test = parseData_train.randomSplit([0.8, 0.2])

# Create SVM Model
model = LogisticRegressionWithLBFGS.train(training,iterations=3000, regParam=0.0,regType='l2',intercept=True, corrections=10, tolerance=1e-6, numClasses = 3)

# Evaluate Model
labelsAndPoints = test.map(lambda p:(p.label,model.predict(p.features)))
trainAccurate = labelsAndPoints.filter(lambda (v, p) : v == p).count() / float(test.count())

model.save(sc, "./logistc_3.model")

print "Accuracy:", trainAccurate
